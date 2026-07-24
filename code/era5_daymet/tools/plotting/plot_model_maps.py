#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
plot_model_maps.py — 会议用: 五模型 + truth 的地图对比 (诊断口径, 不美化)
============================================================================
把 bilinear / BCSD / UNet / ViT / CorrDiff 的预测与 truth 并排画出来, 用来
★诚实展示每个方法哪里好、哪里不好, 以及数据本身的特点与难处★
(历史结论见 docs/archive/results/2026-07-15-results.md)。

产出两类图 (每个变量各一套):
  1) --mode case   单日对比。自动挑一个"强降水事件日"(最难的一类), 或 --day 指定。
     面板 = [truth, Bilinear, BCSD, UNet, ViT, CorrDiff(集合均值), CorrDiff(单成员)]
     ★单成员 vs 集合均值★ 直接展示生成式的"锐但不逐点准 / 均值准但糊"的权衡。
  2) --mode annual 2020 全年(365 天)平均对比。
     上排 = truth + 五方法的年均场; 下排 = 各方法 - truth 的偏差(暴露系统性误差)。
     年均图里 CorrDiff 用回归器 μ(= 集合均值的期望, 残差年均趋零), 省掉 365×16 采样。

温度画成 °C; 降水两版都出: 物理 mm/day(直觉/被暴雨端主导) 与 log1p(mm)(看结构)。

复用: eval_all_methods.build_predictors (bilinear/bcsd/unet/vit) + train_corrdiff 的
      回归器/生成器/分块采样。口径与归档结果文档的评测完全一致(同 ckpt 同 stats)。

用法(登录节点有 GPU, 直接跑):
  python plot_model_maps.py --mode both --out runs/exp/<id>-meeting-maps
  python plot_model_maps.py --mode case --day 200 --ensemble 16 --out ...
  python plot_model_maps.py --mode annual --annual-stride 5 --out ...   # 快速预览
============================================================================
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.evaluation import eval_all_methods as EAM
from era5_daymet.training import train_corrdiff as CD
from era5_daymet.training import train_downscale as TD

EXTENT = [-125.125, -65.125, 23.625, 53.625]         # 与学长 plot_compare.py 一致
ASPECT = 1.0 / np.cos(np.deg2rad(0.5 * (EXTENT[2] + EXTENT[3])))   # 修正美国经纬比例(≈1.28), 替代 aspect=ASPECT
PRECIP = TD.PRECIP

# 画图顺序与显示名(诊断口径, 中性)
MODELS = ["bilinear", "bcsd", "unet", "vit", "corrdiff"]
PRETTY = {"bilinear": "Bilinear", "bcsd": "BCSD", "unet": "UNet",
          "vit": "ViT", "corrdiff": "CorrDiff"}


def var_short(v):
    return {"2m_temperature_max": "tmax", "2m_temperature_min": "tmin",
            PRECIP: "precip"}.get(v, v)


def is_temp(v):
    return "temperature" in v


# ---------------------------------------------------------------------------
# 载入五个模型 -> 统一的 predict 接口
# ---------------------------------------------------------------------------
def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build] device={device}", flush=True)

    # in/out vars 以 UNet ckpt 为准(保证与 corrdiff 回归器/生成器通道一致)
    uck = os.path.join(args.unet_dir, "ckpt.pt")
    a_u = torch.load(uck, map_location="cpu").get("args", {})
    in_vars = a_u.get("in_vars", TD.DEFAULT_IN)
    out_vars = a_u.get("out_vars", TD.TARGETS)
    Cout = len(out_vars)
    Cin = len(in_vars) + 3 + Cout

    stats = TD.Stats(args.stats_dir, in_vars, out_vars)
    test = TD.DownscaleData(args.era5_dir, args.daymet_dir, [args.year],
                            in_vars, out_vars, stats)

    # --- bilinear / bcsd / unet / vit: 直接复用 eval_all_methods 的 predictors ---
    eam_args = SimpleNamespace(
        test_year=args.year, out_vars=out_vars, in_vars=in_vars,
        unet_dir=args.unet_dir, vit_dir=args.vit_dir, base_dir="runs/base",
        scd_dir="runs/scd", bcsd_coef_dir=args.bcsd_coef_dir, ensemble=1)
    det_preds = EAM.build_predictors(["bilinear", "bcsd", "unet", "vit"],
                                     test, stats, eam_args, device)

    # --- corrdiff: 回归器 μ + 生成器 + 分块采样 ---
    gck = torch.load(os.path.join(args.corrdiff_dir, "generator.pt"), map_location=device)
    ca = gck.get("args", {})
    regressor = TD.UNet(Cin, Cout, base=ca.get("reg_base", 64), temb=0).to(device)
    regressor.load_state_dict(torch.load(args.regressor_ckpt, map_location=device)["model"])
    regressor.eval()
    gen = TD.UNet(Cout + Cin + Cout, Cout, base=ca.get("base", 64), temb=128).to(device)
    gen.load_state_dict(gck["model"])
    gen.eval()
    # ★sigma_data: args 里存的是哨兵 -1(=训练时自动估计); 真值在 ckpt 顶层。
    #   用错会让 EDM preconditioning(c_skip/c_out/c_in) 全偏, 采样悄悄失真。
    sd_val = ca.get("sigma_data", 0)
    if not sd_val or sd_val <= 0:
        sd_val = float(gck.get("sigma_data", 1.0))
    edm = CD.EDM(sigma_data=sd_val, sigma_min=ca.get("sigma_min", 0.002),
                 sigma_max=ca.get("sigma_max", 80.0), rho=ca.get("rho", 7.0))
    tile = ca.get("eval_tile", 192)
    steps = ca.get("edm_steps", 18)
    dstd = stats.d_std[:, None, None]
    dmean = stats.d_mean[:, None, None]
    print(f"[build] corrdiff: sigma_data={sd_val:.4f} sigma_max={ca.get('sigma_max',80)} "
          f"tile={tile} steps={steps}", flush=True)

    def _denorm_precip(members):                       # (N,Cout,H,W) 归一残差已加回 -> 物理
        if PRECIP in out_vars:
            pi = out_vars.index(PRECIP)
            members[:, pi] = (TD.precip_inv(members[:, pi], stats.precip_scale)  # 默认钳 log<=8
                              if stats.precip_log else np.maximum(members[:, pi], 0.0))
        return members

    @torch.no_grad()
    def corrdiff_pred(cond, m, ensemble):
        """ensemble<=0 -> 只用 μ(年均图用); >0 -> 分块采样出该数量成员(单日图用)。"""
        cb = torch.from_numpy(cond[None]).float().to(device)
        mu = regressor(cb)
        if ensemble <= 0:
            members = mu.cpu().numpy() * dstd[None] + dmean[None]
            return _denorm_precip(members)
        lm = torch.from_numpy((m[0] if m.ndim == 3 else m) > 0.5).to(device)
        r = CD.tiled_sample(gen, edm, cb, mu, lm, Cout, tile, ensemble, steps, stochastic=True)
        members = (mu.cpu().numpy() + r) * dstd[None] + dmean[None]
        return _denorm_precip(members)

    return device, stats, test, out_vars, det_preds, corrdiff_pred


# ---------------------------------------------------------------------------
# 显示变换与配色(诊断口径)
# ---------------------------------------------------------------------------
def to_display(v, arr, space="physical"):
    """物理量 -> 画图量。温度->°C; 降水-> mm/day 或 log1p(mm)。"""
    if is_temp(v):
        return arr - 273.15
    if space == "log":
        return np.log1p(np.maximum(arr, 0) * 1000.0)
    return np.maximum(arr, 0) * 1000.0                 # mm/day


def field_cmap(v):
    return "RdYlBu_r" if is_temp(v) else "YlGnBu"


def mask_ocean(a2d, land):
    o = a2d.astype(np.float32).copy()
    o[~land] = np.nan
    return o


def _safe(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace("/", "")


def save_panel(a2d, title, cmap, vmin, vmax, unit, path):
    """单面板单独存一张(高清), 便于会上放大/逐张看。色标与组合图一致 -> 仍可跨方法对比。"""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(a2d, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, aspect=ASPECT)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# mode=case: 单日七面板
# ---------------------------------------------------------------------------
def pick_event_day(test, out_vars, y):
    """挑一个陆地平均降水最大的日子(最有空间结构的降水难例)。"""
    if PRECIP not in out_vars:
        return test.ndays[y] // 2
    pi = out_vars.index(PRECIP)
    best_t, best_v = 0, -1.0
    for t in range(test.ndays[y]):
        _, _, m, hr = test.full(y, t)
        land = (m[0] if m.ndim == 3 else m) > 0.5
        val = float(np.maximum(hr[pi], 0)[land].mean())
        if val > best_v:
            best_v, best_t = val, t
    print(f"[case] 自动选定事件日 t={best_t} (陆地均降水最大 = {best_v*1000:.2f} mm/day)", flush=True)
    return best_t


def plot_case(test, out_vars, det_preds, corrdiff_pred, y, t, ensemble, out_dir,
              individual=False, cd_label="CorrDiff"):
    cond, _, m, hr = test.full(y, t)
    land = (m[0] if m.ndim == 3 else m) > 0.5

    # 各方法预测(物理量)。det 方法 N=1; corrdiff 出 ensemble 个成员
    fields = {"truth": hr}
    for name, fn in det_preds.items():
        fields[name] = fn(cond, t)[0]                  # (Cout,H,W)
    cd_mem = corrdiff_pred(cond, m, ensemble)          # (ensemble,Cout,H,W)
    fields["corrdiff_mean"] = cd_mem.mean(0)
    fields["corrdiff_member"] = cd_mem[0]

    panels = [("truth", "Truth (Daymet)")] + \
             [(k, PRETTY[k]) for k in MODELS[:-1]] + \
             [("corrdiff_mean", f"{cd_label} (ens mean)"),
              ("corrdiff_member", f"{cd_label} (1 member)")]

    for vi, v in enumerate(out_vars):
        spaces = [("log", "log1p(mm)"), ("physical", "mm/day")] if v == PRECIP else [("physical", "°C")]
        for space, unit in spaces:
            disp = {k: mask_ocean(to_display(v, arr[vi], space), land) for k, arr in fields.items()}
            vmin = float(np.nanpercentile(disp["truth"], 2))
            vmax = float(np.nanpercentile(disp["truth"], 98))
            n = len(panels)
            fig, ax = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False)
            im = None
            for c, (k, title) in enumerate(panels):
                a = ax[0, c]
                im = a.imshow(disp[k], origin="lower", extent=EXTENT, cmap=field_cmap(v),
                              vmin=vmin, vmax=vmax, aspect=ASPECT)
                a.set_title(title, fontsize=9)
                a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=ax[0, :].tolist(), fraction=0.012, pad=0.01, label=unit)
            from datetime import date, timedelta
            dstr = (date(y, 1, 1) + timedelta(days=int(t))).isoformat()
            fig.suptitle(f"{var_short(v)}  single case {dstr}  [{unit}]   "
                         f"(shared color scale; member is sharper but not pointwise-accurate)",
                         fontsize=11)
            tag = f"case_{var_short(v)}" + (f"_{space}" if v == PRECIP else "")
            p = os.path.join(out_dir, f"{tag}.png")
            fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
            print(f"  -> {p}", flush=True)

            if individual:                                 # 每个面板单独存一张
                sub = os.path.join(out_dir, "individual", tag)
                os.makedirs(sub, exist_ok=True)
                for c, (k, title) in enumerate(panels):
                    save_panel(disp[k], f"{var_short(v)} {dstr} — {title}",
                               field_cmap(v), vmin, vmax, unit,
                               os.path.join(sub, f"{c:02d}_{_safe(k)}.png"))
                print(f"     单图 -> {sub}/ ({len(panels)} 张)", flush=True)


# ---------------------------------------------------------------------------
# mode=annual: 365 天平均, 上排场 + 下排偏差
# ---------------------------------------------------------------------------
def plot_annual(test, out_vars, det_preds, corrdiff_pred, y, stride, out_dir, individual=False):
    Cout = len(out_vars)
    days = list(range(0, test.ndays[y], stride))
    names = ["truth"] + MODELS
    acc = {k: np.zeros((Cout, test.H, test.W), np.float64) for k in names}
    land = None
    t0 = time.time()
    for di, t in enumerate(days):
        cond, _, m, hr = test.full(y, t)
        if land is None:
            land = (m[0] if m.ndim == 3 else m) > 0.5
        acc["truth"] += hr
        for name, fn in det_preds.items():
            acc[name] += fn(cond, t)[0]
        acc["corrdiff"] += corrdiff_pred(cond, m, ensemble=0)[0]     # μ only
        if (di + 1) % 25 == 0:
            print(f"  annual {di+1}/{len(days)} 天 ({time.time()-t0:.0f}s)", flush=True)
    mean = {k: acc[k] / len(days) for k in names}

    for vi, v in enumerate(out_vars):
        spaces = [("log", "log1p(mm)"), ("physical", "mm/day")] if v == PRECIP else [("physical", "°C")]
        for space, unit in spaces:
            disp = {k: mask_ocean(to_display(v, mean[k][vi], space), land) for k in names}
            vmin = float(np.nanpercentile(disp["truth"], 2))
            vmax = float(np.nanpercentile(disp["truth"], 98))
            diffs = {k: disp[k] - disp["truth"] for k in MODELS}
            dmax = float(np.nanpercentile(np.abs(np.stack(list(diffs.values()))), 98)) or 1.0
            ncol = 1 + len(MODELS)
            fig, ax = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4), squeeze=False)
            # 上排: 场
            top = [("truth", "Truth (Daymet)")] + [(k, PRETTY[k]) for k in MODELS]
            imf = None
            for c, (k, title) in enumerate(top):
                a = ax[0, c]
                imf = a.imshow(disp[k], origin="lower", extent=EXTENT, cmap=field_cmap(v),
                               vmin=vmin, vmax=vmax, aspect=ASPECT)
                a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(imf, ax=ax[0, :].tolist(), fraction=0.012, pad=0.01, label=unit)
            # 下排: 偏差 (truth 列留空)
            ax[1, 0].axis("off")
            imb = None
            for c, k in enumerate(MODELS):
                a = ax[1, c + 1]
                imb = a.imshow(diffs[k], origin="lower", extent=EXTENT, cmap="RdBu_r",
                               vmin=-dmax, vmax=dmax, aspect=ASPECT)
                a.set_title(f"{PRETTY[k]} - truth", fontsize=9); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(imb, ax=ax[1, 1:].tolist(), fraction=0.012, pad=0.01, label=f"Δ {unit}")
            fig.suptitle(f"{var_short(v)}  2020 annual mean ({len(days)} days, stride={stride})  [{unit}]"
                         f"   [top: fields | bottom: model - truth]", fontsize=11)
            tag = f"annual_{var_short(v)}" + (f"_{space}" if v == PRECIP else "")
            p = os.path.join(out_dir, f"{tag}.png")
            fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
            print(f"  -> {p}", flush=True)

            if individual:                                 # 场面板 + 偏差面板各自单独存
                sub = os.path.join(out_dir, "individual", tag)
                os.makedirs(sub, exist_ok=True)
                for c, (k, title) in enumerate(top):
                    save_panel(disp[k], f"{var_short(v)} annual — {title}",
                               field_cmap(v), vmin, vmax, unit,
                               os.path.join(sub, f"{c:02d}_field_{_safe(k)}.png"))
                for k in MODELS:
                    save_panel(diffs[k], f"{var_short(v)} annual — {PRETTY[k]} minus truth",
                               "RdBu_r", -dmax, dmax, f"Δ {unit}",
                               os.path.join(sub, f"bias_{_safe(k)}.png"))
                print(f"     单图 -> {sub}/ ({len(top)+len(MODELS)} 张)", flush=True)


def main():
    W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=["case", "annual", "both"], default="both")
    p.add_argument("--day", type=int, default=-1, help="单日图的日索引; -1 = 自动挑强降水事件日")
    p.add_argument("--ensemble", type=int, default=16, help="单日图 CorrDiff 集合成员数")
    p.add_argument("--annual-stride", type=int, default=1, help="年均图取样步长; 1=全年365天")
    p.add_argument("--individual", action="store_true",
                   help="除组合图外, 把每个面板单独存一张高清PNG到 individual/ 子目录")
    p.add_argument("--corrdiff-label", default="CorrDiff",
                   help="CorrDiff 面板标题(如 'CorrDiff-papernoise' 区分噪声档对照)")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--out", default=f"{W}/runs/exp/{time.strftime('%Y%m%d')}-meeting-maps")
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default=f"{W}/runs/stats/train_dayofyear")
    p.add_argument("--unet-dir", default=f"{W}/runs/exp/20260711-unet-b64")
    p.add_argument("--vit-dir", default=f"{W}/runs/exp/20260712-vit-d384-b16-ep12")
    p.add_argument("--bcsd-coef-dir", default=f"{W}/runs/bcsd_coefs")
    p.add_argument("--corrdiff-dir", default=f"{W}/runs/exp/20260714-corrdiff-b64")
    p.add_argument("--regressor-ckpt", default=f"{W}/runs/exp/20260711-unet-b64/ckpt.pt")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device, stats, test, out_vars, det_preds, corrdiff_pred = build(args)
    y = args.year

    if args.mode in ("case", "both"):
        t = args.day if args.day >= 0 else pick_event_day(test, out_vars, y)
        print(f"[case] 画单日图 t={t}, ensemble={args.ensemble}", flush=True)
        plot_case(test, out_vars, det_preds, corrdiff_pred, y, t, args.ensemble, args.out,
                  individual=args.individual, cd_label=args.corrdiff_label)
    if args.mode in ("annual", "both"):
        print(f"[annual] 画年均图 stride={args.annual_stride}", flush=True)
        plot_annual(test, out_vars, det_preds, corrdiff_pred, y, args.annual_stride, args.out,
                    individual=args.individual)
    print(f"[done] 图输出到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
