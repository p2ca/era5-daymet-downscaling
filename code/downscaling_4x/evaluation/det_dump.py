#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
det_dump.py — 确定性方法的逐日场落盘(与集合方法同一布局)
============================================================================
把 UNet / CorrDiff 阶段A 的 μ / 确定性 JiT(JDA/JMA) / BCSD / 插值 的整年预测落成
逐日逐像素场, 目录布局与生成式采样落盘完全一致, 因此 dump_metrics 与 render 管线通吃:

  <out>/<目标>/ens_mean/<年>_d<日>.npy   预测场    float32 物理单位 (有效域外 NaN)
  <out>/<目标>/crps/<年>_d<日>.npy       逐像素 CRPS float32 物理单位 (有效域外 NaN)
  <out>/<目标>/crps_log/<年>_d<日>.npy   同上, log1p(mm) 空间 (仅降水)
  <out>/meta.json

确定性方法只有一个成员, 其 CRPS 恒等于 |预测 − 真值|, 因此 crps 场在这里直接给出, 下游
无需知道产生它的方法是不是集合方法。**不落 spread 与 rank**: 单成员的离散度恒为 0、名次
只有两档, 落了会让下游画出一张看似正常、实则无意义的名次直方图; 场缺失时 render 会跳过
依赖它的模块, 这正是期望行为。

NN 方法的条件模式、目标与结构一律取 checkpoint 自述, 不接受命令行覆盖; JiT 走整幅前向
与固定切块起点 offset=(0,0), 与 μ 缓存(build_mu_cache)同一约定 —— 评测看到的 JDA/JMA
与阶段 B 消费的 μ 是同一个函数。
============================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import downscale_baseline as DB
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.baselines.eval_baselines import bcsd_predict
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.metrics import precip_log_mm
from downscaling_4x.training import train_downscale as TD

NN_METHODS = ("unet", "corrdiff-a", "jda", "jma")
STAT_METHODS = ("bcsd", "bilinear", "bicubic")
# 方法名与 checkpoint 结构的强制对应。拿错目录不会有任何形状错误 —— 权重装得回去、
# 前向照跑、指标正常产出, 只是全部错标在另一个方法名下, 因此必须在入口硬校验。
METHOD_ARCH = {"unet": "unet", "corrdiff-a": "corrdiff", "jda": "jit", "jma": "jit"}
METHOD_MOE = {"jda": False, "jma": True}


def _sha_stat(path):
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def _nn_predictor(args, device):
    """按 checkpoint 还原确定性主体, 返回 (out_vars, mode, fn(cond)->归一化输出)。"""
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck = payload.get("args", {})
    # 真实 ERA5 与 oracle 输入同布局, 拿错目录前向照跑; 先核 checkpoint 自述的输入目录
    II.check(ck.get("era5_dir"), args.era5_dir, f"checkpoint {args.checkpoint}",
             allow=getattr(args, II.ALLOW_DEST, False))
    arch = ck.get("arch", "unet") or "unet"
    if arch != METHOD_ARCH[args.method]:
        raise SystemExit(f"--method {args.method} 要求 arch={METHOD_ARCH[args.method]}, "
                         f"checkpoint 是 {arch!r} —— 拿错了 checkpoint")
    if args.method in METHOD_MOE and bool(ck.get("moe")) != METHOD_MOE[args.method]:
        raise SystemExit(f"--method {args.method} 要求 moe={METHOD_MOE[args.method]}, "
                         f"checkpoint moe={bool(ck.get('moe'))} —— jda/jma 拿反了")
    if ck.get("crop"):
        raise SystemExit(f"checkpoint 是按 crop={ck['crop']} 训的(仅冒烟用), 不能整幅评测")
    mode = ck.get("mode", C.DEFAULT_MODE)
    tgt = ck.get("target", C.TARGETS[0])
    out_vars = list(C.TARGETS) if tgt == "all" else [tgt]
    net = TD.build_regressor(C.cond_channels(mode), len(out_vars), ck, hw=C.HR_SHAPE)
    net.load_state_dict(payload["model"])
    net.to(device).eval()
    for q in net.parameters():
        q.requires_grad_(False)

    def predict(cond):
        with torch.no_grad():
            t = torch.from_numpy(cond[None]).float().to(device)
            return net(t)[0].float().cpu().numpy()

    return out_vars, mode, predict


def main():
    ap = argparse.ArgumentParser(description="确定性方法逐日场落盘(布局同生成式采样落盘)")
    ap.add_argument("--method", required=True, choices=NN_METHODS + STAT_METHODS)
    ap.add_argument("--checkpoint", default="", help="NN 方法的 checkpoint")
    ap.add_argument("--bcsd-coef-dir", default="", help="bcsd: fit_bcsd_coefs.py 落的系数目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="统计方法用; NN 方法一律取 checkpoint 记录的目标")
    ap.add_argument("--max-days", type=int, default=0, help=">0 时只跑前 N 天, 冒烟用")
    II.add_arg(ap)
    a = ap.parse_args()

    device = "cpu"
    predict = None
    if a.method in NN_METHODS:
        import torch
        if not a.checkpoint:
            raise SystemExit(f"--method {a.method} 需要 --checkpoint")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out_vars, mode, predict = _nn_predictor(a, device)
        sources = [dict(_sha_stat(a.checkpoint), targets=out_vars)]
    else:
        out_vars = list(a.targets or C.TARGETS)
        mode = "baseline_21"                     # 统计方法不用条件张量, 更不需要历史帧
        sources = []
        if a.method == "bcsd":
            if not a.bcsd_coef_dir:
                raise SystemExit("--method bcsd 需要 --bcsd-coef-dir")
            sources = [dict(_sha_stat(Path(a.bcsd_coef_dir) / f"{v}.npz"), targets=[v])
                       for v in out_vars]

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    avail = sorted({a.year - 1, a.year}) if need else [a.year]
    fi = FrameIndex([a.year], avail, need, lags, split=f"dump{a.year}")
    ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
    test = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode,
                         era5_cache_years=len(ds_years))
    n_days = C.DAYS_PER_YEAR
    frames = list(range(len(fi)))
    if a.max_days:
        frames = frames[: a.max_days]
    if fi.dropped:
        print(f"[det-dump] 注意: {a.year} 有 {fi.dropped} 天因缺历史被剔除, "
              f"可评 {len(fi)}/{n_days} 天", flush=True)
    land = test.mask
    ti_list = [C.TARGETS.index(v) for v in out_vars]
    dstd = stats.d_std[ti_list][:, None, None]
    dmean = stats.d_mean[ti_list][:, None, None]
    up = DB.make_bilinear(test.Hl, test.Wl, C.FACTOR) if a.method in STAT_METHODS else None
    up_bc = DB.make_bicubic(test.Hl, test.Wl, C.FACTOR) if a.method == "bicubic" else None
    coefs = {}
    if a.method == "bcsd":
        for v in out_vars:
            cp = Path(a.bcsd_coef_dir) / f"{v}.npz"
            d = np.load(cp, allow_pickle=True)
            II.check(str(d["era5_dir"]) if "era5_dir" in d.files else None, a.era5_dir,
                     f"BCSD 系数 {cp}", allow=getattr(a, II.ALLOW_DEST, False))
            coefs[v] = (d["a"], d["b"], str(d["space"]))

    out_root = Path(a.out)
    written = {}
    for v in out_vars:
        for f in ("ens_mean", "crps") + (("crps_log",) if v == C.PRECIP else ()):
            (out_root / v / f).mkdir(parents=True, exist_ok=True)
            written[f"{v}/{f}"] = 0

    lg = lambda x: precip_log_mm(x, stats.precip_clip)  # 入参已是 mm/day; 与指标、统计基线同一条变换
    t0 = time.time()
    for done, k in enumerate(frames, 1):
        year, day = fi.frames[k]
        if predict is not None:
            cond, _, _, raw = test.full(year, day, fi.history_of(k))
            pred = predict(cond) * dstd + dmean
            if C.PRECIP in out_vars:
                pi = out_vars.index(C.PRECIP)
                pred[pi] = (C.precip_inv(pred[pi], stats.precip_scale)
                            if stats.precip_log else np.maximum(pred[pi], 0.0))
        else:
            _, raw = test.target(year, day)
            outs = []
            for v in out_vars:
                x = test._era5_year(year)[v][day].astype(np.float32)
                if a.method == "bcsd":
                    a_, b_, _ = coefs[v]
                    # 与系数拟合、eval_baselines 同一条公式: 先上采样再变换(两者不可交换)
                    q = bcsd_predict(up(x), a_, b_, v == C.PRECIP, stats.precip_clip, stats.precip_scale)
                else:
                    q = (up_bc if a.method == "bicubic" else up)(x)
                outs.append(q.astype(np.float32))
            pred = np.stack(outs, 0)
        truth = raw[ti_list]

        for i, v in enumerate(out_vars):
            # 降水在数据层是 m/day, 而落盘布局的场一律存 mm/day; 不换算的话下游会把预测
            # 当成近似 0, MAE 恰好等于真值均值, 且 corr 因量纲不变而依然正常 ——
            # 是一处不会报错的单位错。
            unit_k = stats.precip_scale if v == C.PRECIP else 1.0
            p = np.where(land, pred[i] * unit_k, np.nan).astype(np.float32)
            t = np.where(land, truth[i] * unit_k, np.nan).astype(np.float32)
            np.save(out_root / v / "ens_mean" / f"{year}_d{day}.npy", p)
            written[f"{v}/ens_mean"] += 1
            # 单成员的 CRPS 恒等于绝对误差, 下游因此不必区分确定性与集合方法
            np.save(out_root / v / "crps" / f"{year}_d{day}.npy",
                    np.abs(p - t).astype(np.float32))
            written[f"{v}/crps"] += 1
            if v == C.PRECIP:
                np.save(out_root / v / "crps_log" / f"{year}_d{day}.npy",
                        np.abs(lg(p) - lg(t)).astype(np.float32))
                written[f"{v}/crps_log"] += 1
        if done % 25 == 0 or done == len(frames):
            print(f"[det-dump] {done}/{len(frames)} days ({time.time()-t0:.0f}s)", flush=True)

    meta = {
        "kind": "deterministic_field_dump",
        "method": a.method,
        "targets": out_vars,
        "units": {v: ("mm/day" if v == C.PRECIP else "K") for v in out_vars},
        "mode": mode,
        "cond_channels": C.cond_channels(mode) if predict is not None else None,
        "inference": ("整幅单次前向; JiT 切块起点固定 offset=(0,0), 与 μ 缓存同约定"
                      if predict is not None else a.method),
        "model_sources": sources,
        "input": II.describe(a.era5_dir),
        "daymet_dir": II.normalize(a.daymet_dir),
        "years": [a.year],
        "n_days_expected": n_days,
        "n_days_written": len(frames),
        "all_days": len(frames) == n_days,
        "members": 1,
        "fields": ["ens_mean", "crps"] + (["crps_log"] if C.PRECIP in out_vars else []),
        "crps_note": "确定性方法只有一个成员, CRPS 恒等于 |预测 − 真值|",
        "crps_log_transform": f"precip_fwd: <{stats.precip_clip:g} mm 置零后 log1p (预测与真值同)",
        "no_spread_rank": "单成员的 spread 恒为 0、rank 只有两档, 故不落盘; 依赖它们的下游模块会自动跳过",
        "mask_note": "有效域 = daymet_land AND era5_valid; 场按 (H,W) 落盘, 有效域外 NaN",
        "written": written,
        "status": "done" if len(frames) == n_days else "partial",
    }
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    # 每个目标目录另写一份自己的 meta: 下游(dump_metrics / render)拿到的是 <out>/<目标>,
    # 身份与单位必须在那一层就能读到, 否则图与指标会失去出处
    for v in out_vars:
        one = dict(meta, target=v, targets=[v], unit=meta["units"][v],
                   fields=["ens_mean", "crps"] + (["crps_log"] if v == C.PRECIP else []),
                   written={k.split("/", 1)[1]: n for k, n in written.items()
                            if k.startswith(f"{v}/")})
        (out_root / v / "meta.json").write_text(json.dumps(one, ensure_ascii=False, indent=1) + "\n")
    print(f"[det-dump] 完成 -> {out_root}  ({len(frames)}/{n_days} 天)", flush=True)


if __name__ == "__main__":
    main()
