#!/usr/bin/env python
# Packaged implementation; code/eval_all_methods.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
eval_all_methods.py — 一条命令把 5 个方法放进同一张表 + 同一组图
============================================================================
bilinear / bicubic / UNet / ViT / SCD(EDM 集合) 在同一天循环里各自产出物理预测,
全部喂给 eval_common.MultiMethodEval, 于是:
  * 一张总表(RMSE/MAE/bias/corr/CRPS, 含集合成员数 N)
  * 一组图: 逐变量功率谱(各方法 + truth) / 年平均地图(truth vs 各方法 + 偏差) /
            rank histogram(仅 SCD 集合)
口径与 train_scd.py 完全一致(都用 eval_common)。

前置: 先各自训练好 ckpt
  train_unet.py --out runs/unet   -> runs/unet/ckpt.pt
  train_vit.py  --out runs/vit    -> runs/vit/ckpt.pt
  train_scd.py  --stage both --out runs/scd -> runs/scd/{corrector,generator}.pt
然后:
  python eval_all_methods.py --unet-dir runs/unet --vit-dir runs/vit --scd-dir runs/scd --out runs/compare_all
缺哪个 ckpt 就自动跳过哪个方法(插值永远可用)。

  python eval_all_methods.py --smoke      # 合成数据自测(无 torch 时只跑插值)
============================================================================
"""
import argparse
import os
import sys
import time

import numpy as np

from era5_daymet.data import downscale_baseline as DB
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.evaluation import eval_common as EC
from era5_daymet.training import train_downscale as TD
from era5_daymet import contract as C
from era5_daymet.data import dataset as DS
from era5_daymet.models.unet import UNet
from era5_daymet.models import tiled_inference as TI
from era5_daymet.training import train_scd as SCD
from era5_daymet.training import train_vit as VM
from era5_daymet.paths import PROJECT_ROOT

try:
    import torch
except Exception:
    torch = None

INTERP = ("bilinear", "bicubic")
ALL = ["bilinear", "bicubic", "bcsd", "unet", "vit", "scd"]


def _load_ckpt_args(path):
    return torch.load(path, map_location="cpu").get("args", {})


def _resolve_ckpt(method, args):
    """新布局 runs/{unet,vit}/ckpt.pt 优先; 兼容旧 runs/base/{unet,vit}.pt。"""
    if method == "unet":
        cands = [os.path.join(args.unet_dir, "ckpt.pt"), os.path.join(args.base_dir, "unet.pt")]
    else:
        cands = [os.path.join(args.vit_dir, "ckpt.pt"), os.path.join(args.base_dir, "vit.pt")]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def build_predictors(methods, test, stats, args, device):
    """返回 {method: fn(cond, t) -> members (N,Cout,H,W) 物理量}。"""
    y = args.test_year; out_vars = args.out_vars; Cout = len(out_vars)
    Cin = C.cond_channels(args.in_vars, out_vars, getattr(args, "use_clim", True))
    dstd = stats.d_std[:, None, None]; dmean = stats.d_mean[:, None, None]
    up_bl = DB.make_bilinear(test.Hl, test.Wl, C.FACTOR)
    up_bc = DB.make_bicubic(test.Hl, test.Wl, C.FACTOR)
    preds = {}

    def denorm(o):                                            # (N,Cout,H,W) 归一 -> 物理
        ph = o * dstd[None] + dmean[None]
        if C.PRECIP in out_vars:
            pi = out_vars.index(C.PRECIP)
            ph[:, pi] = (C.precip_inv(ph[:, pi], stats.precip_scale)
                         if stats.precip_log else np.maximum(ph[:, pi], 0.0))
        return ph

    if "bilinear" in methods:
        preds["bilinear"] = lambda cond, t: np.stack(
            [up_bl(test.lr[y][v][t].astype(np.float32)) for v in out_vars], 0)[None]
    if "bicubic" in methods:
        preds["bicubic"] = lambda cond, t: np.stack(
            [up_bc(test.lr[y][v][t].astype(np.float32)) for v in out_vars], 0)[None]

    if "bcsd" in methods:
        # 逐像素线性回归, 系数由 fit_bcsd_coefs.py 预先拟合并存盘(38 年, stride=1)。
        # ★各变量在自己的空间里拟合★: 温度=恒等空间(K), 降水=log1p(mm)。
        # 降水预测完必须 expm1 反变换回 m/day, 否则与其它方法不可通约。
        coefs = {}
        for v in out_vars:
            f = os.path.join(args.bcsd_coef_dir, f"{v}.npz")
            d = np.load(f, allow_pickle=True)
            coefs[v] = (d["a"], d["b"], str(d["space"]))

        def f_bcsd(cond, t):
            outs = []
            for v in out_vars:
                a_, b_, sp = coefs[v]
                x = test.lr[y][v][t].astype(np.float32)
                if v == C.PRECIP:                                # log1p(mm) 空间
                    p = a_ * up_bl(np.log1p(np.maximum(x, 0) * 1000.0)) + b_
                    p = np.maximum(np.expm1(p), 0.0) / 1000.0     # -> m/day
                else:                                             # 恒等空间(K)
                    p = a_ * up_bl(x) + b_
                outs.append(p.astype(np.float32))
            return np.stack(outs, 0)[None]
        preds["bcsd"] = f_bcsd

    if "unet" in methods:
        ck = _resolve_ckpt("unet", args); a = _load_ckpt_args(ck)
        net_u = TD.build_regressor(Cin, Cout, a).to(device)
        net_u.load_state_dict(torch.load(ck, map_location=device)["model"]); net_u.eval()

        def f_unet(cond, t, _net=net_u):
            with torch.no_grad():                              # UNet 全卷积: 整帧
                o = TI.det_predict(_net, torch.from_numpy(cond[None]).float().to(device), 0, device)[None]
            return denorm(o)
        preds["unet"] = f_unet

    if "vit" in methods:
        ck = _resolve_ckpt("vit", args); a = _load_ckpt_args(ck)
        # 整幅 ckpt(full_frame): 整幅一次前向(tile=0); crop ckpt: 训练裁剪块=评测分块尺寸
        tile = 0 if a.get("full_frame") else a.get("patch", 64)
        # pos_type/head_up: 老 ckpt(无此键)回退 learned+convt 旧结构; 新整幅 ckpt 为 sincos+pixelshuffle
        net_v = VM.ViT(Cin, Cout, img=a.get("patch", 64), patch=a.get("vit_patch", 2), dim=a.get("dim", 384),
                       depth=a.get("depth", 8), heads=a.get("heads", 6), mlp=a.get("mlp", 4.0),
                       pos_type=a.get("pos_type", "learned"), head_up=a.get("head_up", "convt"),
                       full_frame=bool(a.get("full_frame", False))).to(device)
        vsd = torch.load(ck, map_location=device)["model"]
        # 兼容加 dropout 之前的旧 ckpt: MLP 第二个 Linear 从 mlp.2 迁到 mlp.3(dropout 无参数, 对新 ckpt 是 no-op)
        if not any(k.endswith("blocks.0.mlp.3.weight") for k in vsd):
            vsd = {k.replace(".mlp.2.", ".mlp.3."): x for k, x in vsd.items()}
        net_v.load_state_dict(vsd); net_v.eval()

        def f_vit(cond, t, _net=net_v, _tile=tile):
            with torch.no_grad():                              # ViT 位置编码绑定 patch: 分块重叠平均
                o = TI.det_predict(_net, torch.from_numpy(cond[None]).float().to(device), _tile, device)[None]
            return denorm(o)
        preds["vit"] = f_vit

    if "scd" in methods:
        cc = os.path.join(args.scd_dir, "corrector.pt"); gc = os.path.join(args.scd_dir, "generator.pt")
        a = _load_ckpt_args(gc)
        k = a.get("kappa_factor", C.FACTOR); use_sigma = not a.get("no_sigma_cond", False)
        corr = UNet(Cin, 2 * Cout, base=a.get("corrector_base", 48), temb=0).to(device)
        corr.load_state_dict(torch.load(cc, map_location=device)["model"]); corr.eval()
        gen_in = Cout + Cin + Cout + (Cout if use_sigma else 0)
        gen = UNet(gen_in, Cout, base=a.get("base", 64), temb=128).to(device)
        gen.load_state_dict(torch.load(gc, map_location=device)["model"]); gen.eval()
        edm = SCD.EDM(sigma_data=a.get("sigma_data", 1.0), sigma_min=a.get("sigma_min", 0.002),
                      sigma_max=a.get("sigma_max", 80.0), rho=a.get("rho", 7.0))
        steps = a.get("edm_steps", 18)

        def f_scd(cond, t):
            cb = torch.from_numpy(cond[None]).float().to(device); H, W = cond.shape[1:]
            with torch.no_grad():
                mu, logvar = SCD.split_corrector(corr(SCD.coarsen(cb, k)), Cout)
                sig = torch.exp(0.5 * logvar)
                cg = [cb, SCD.upsample(mu, (H, W))] + ([SCD.upsample(sig, (H, W))] if use_sigma else [])
                cond_g = torch.cat(cg, 1)
                mem = [edm.sample(gen, cond_g, (1, Cout, H, W), steps=steps)[0].cpu().numpy()
                       for _ in range(args.ensemble)]
            return denorm(np.stack(mem, 0))
        preds["scd"] = f_scd
    return preds


def available(methods, args):
    """按 ckpt 是否存在过滤方法(插值永远可用)。"""
    ok = []
    for m in methods:
        if m in INTERP:
            ok.append(m)
        elif m in ("unet", "vit"):
            ck = _resolve_ckpt(m, args)
            if torch is not None and os.path.exists(ck):
                ok.append(m)
            else:
                print(f"  跳过 {m}: 缺 {ck} 或无 torch", flush=True)
        elif m == "bcsd":                                   # 无需 torch, 只要有预拟合的系数
            need = [os.path.join(args.bcsd_coef_dir, f"{v}.npz") for v in args.out_vars]
            miss = [p for p in need if not os.path.exists(p)]
            if not miss:
                ok.append(m)
            else:
                print(f"  跳过 bcsd: 缺系数 {miss[0]} (先跑 fit_bcsd_coefs.py)", flush=True)
        elif m == "scd":
            need = [os.path.join(args.scd_dir, x) for x in ("corrector.pt", "generator.pt")]
            if torch is not None and all(os.path.exists(p) for p in need):
                ok.append(m)
            else:
                print(f"  跳过 scd: 缺 {args.scd_dir}/corrector.pt|generator.pt 或无 torch", flush=True)
    return ok


def run(args):
    device = "cpu"
    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = available(args.methods, args)
    if not methods:
        sys.exit("没有可评测的方法。")
    # 若有 NN/SCD ckpt, 用其训练时的 in/out vars + 气候态口径以防通道不匹配
    args.use_clim = False                                 # 无 ckpt(纯插值)时的默认(20通道口径, cond 不被插值用)
    for m in methods:
        ckpt = None
        if m in ("unet", "vit"):
            ckpt = _resolve_ckpt(m, args)
        elif m == "scd":
            ckpt = os.path.join(args.scd_dir, "generator.pt")
        if ckpt and os.path.exists(ckpt):
            a = _load_ckpt_args(ckpt)
            if a.get("in_vars"):
                args.in_vars = a["in_vars"]; args.out_vars = a["out_vars"]
            args.use_clim = a.get("use_clim", True)       # 旧 ckpt 无此键=23通道(True); 新 ckpt 存实际值
            break

    stats = DS.Stats(args.stats_dir, args.in_vars, args.out_vars)
    test = DS.DownscaleData(args.era5_dir, args.daymet_dir, [args.test_year],
                            args.in_vars, args.out_vars, stats, use_clim=args.use_clim)
    y = args.test_year; days = list(range(0, test.ndays[y], args.eval_stride))
    print(f"评测方法={methods}  test={y}  天数={len(days)}(stride={args.eval_stride})  device={device}", flush=True)

    preds = build_predictors(methods, test, stats, args, device)
    ev = EC.MultiMethodEval(methods, args.out_vars, test.H, test.W, test.mask[y],
                            precip_scale=stats.precip_scale, precip_log=stats.precip_log)
    t0 = time.time()
    for di, t in enumerate(days):
        cond, _, m, hr = test.full(y, t)
        members = {mth: preds[mth](cond, t) for mth in methods}
        ev.add_day(hr, m, members)
        if (di + 1) % 25 == 0:
            print(f"  {di+1}/{len(days)} days ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    ev.finalize(args.out, y, eval_stride=args.eval_stride, tag="all", n_total_days=test.ndays[y])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--methods", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--era5-dir", default=M.ERA5_DIR); p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default="stats_train")
    p.add_argument("--in-vars", nargs="+", default=C.DEFAULT_IN)
    p.add_argument("--out-vars", nargs="+", default=C.TARGETS)
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])
    p.add_argument("--unet-dir", default=str(PROJECT_ROOT / "runs/unet"), help="train_unet.py 的输出目录(找 ckpt.pt)")
    p.add_argument("--vit-dir", default=str(PROJECT_ROOT / "runs/vit"), help="train_vit.py 的输出目录(找 ckpt.pt)")
    p.add_argument("--base-dir", default=str(PROJECT_ROOT / "runs/base"), help="旧布局兼容: unet.pt / vit.pt 所在目录")
    p.add_argument("--scd-dir", default=str(PROJECT_ROOT / "runs/scd"), help="corrector.pt / generator.pt 所在目录")
    p.add_argument("--bcsd-coef-dir", default=str(PROJECT_ROOT / "runs/bcsd_coefs"),
                   help="BCSD 逐像素系数 {var}.npz 所在目录 (由 fit_bcsd_coefs.py 生成)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "runs/compare_all"))
    p.add_argument("--ensemble", type=int, default=16, help="SCD 集合成员数")
    p.add_argument("--eval-stride", type=int, default=1, help="1=完整 2020")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        import tempfile
        root = tempfile.mkdtemp()
        args.era5_dir, args.daymet_dir, args.stats_dir = TD.make_synth(root)
        args.in_vars = C.TARGETS; args.out_vars = C.TARGETS
        args.test_year = 2020; args.eval_stride = 10; args.ensemble = 3
        args.out = root + "/cmp"
        if torch is None:
            args.methods = [m for m in args.methods if m in INTERP] or list(INTERP)
        print(f"[smoke] methods={args.methods} (torch={'有' if torch else '无, 仅插值'})", flush=True)

    run(args)


if __name__ == "__main__":
    main()
