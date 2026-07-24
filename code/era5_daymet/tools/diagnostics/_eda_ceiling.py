#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""模型 vs 数据: 复杂度阶梯的细尺度 skill(能否把细节对上 truth) + 输入 vs 真值对照。
逻辑: bilinear(无细节)→ BCSD(线性+静态)→ UNet → ViT, 若 skill 上到某处就平(谁也拉不开),
     则天花板在数据不在模型。温度 vs 降水对比即回答"模型不行 or 不适配数据"。"""
import os, sys, time
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import make_bilinear
from era5_daymet.tools.plotting import plot_model_maps as PM
from era5_daymet.training import train_downscale as TD

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUTDIR = f"{W}/runs/exp/20260720-eda-ceiling"
YEAR = 2020; DAYS = list(range(0, 365, 45)); BOX = 384; SIGMA = 6
PRECIP = TD.PRECIP
os.makedirs(OUTDIR, exist_ok=True)
logmm = lambda x: np.log1p(np.maximum(x, 0.0) * 1000.0)
VARS = ["2m_temperature_max", "2m_temperature_min", PRECIP]
VS = {"2m_temperature_max": "tmax", "2m_temperature_min": "tmin", PRECIP: "precip"}
LADDER = ["bilinear", "bcsd", "unet", "vit"]
LNAME = {"bilinear": "Bilinear (input, no detail)", "bcsd": "BCSD (linear + static)", "unet": "UNet", "vit": "ViT"}
LCOL = {"bilinear": "#9aa0a6", "bcsd": "#7b4fa0", "unet": "#1f77b4", "vit": "#ff7f0e"}


def hp(f):                       # 高通 = 原场 - 高斯平滑, 取细尺度
    return f - gaussian_filter(f, SIGMA)


def skill(a, b, mask):           # 两个高通场在陆地上的相关系数
    x = a[mask]; y = b[mask]; x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x*x).sum() * (y*y).sum())
    return float((x*y).sum()/d) if d > 0 else 0.0


def to_space(v, phys):
    return logmm(phys) if v == PRECIP else phys


def main():
    t0 = time.time()
    a = SimpleNamespace(year=YEAR, stats_dir=f"{W}/runs/stats/train_dayofyear",
        era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR,
        unet_dir=f"{W}/runs/exp/20260711-unet-b64", vit_dir=f"{W}/runs/exp/20260712-vit-d384-b16-ep12",
        bcsd_coef_dir=f"{W}/runs/bcsd_coefs", corrdiff_dir=f"{W}/runs/exp/20260714-corrdiff-b64",
        regressor_ckpt=f"{W}/runs/exp/20260711-unet-b64/ckpt.pt")
    device, stats, test, out_vars, det_preds, _ = PM.build(a)
    vi = {v: out_vars.index(v) for v in VARS}

    sk = {v: {m: [] for m in LADDER} for v in VARS}
    for t in DAYS:
        cond, _, m, hr = test.full(YEAR, t)
        land = (m[0] if m.ndim == 3 else m) > 0.5
        by, bx, bs = TD.pick_land_box(land, min(BOX, test.H, test.W))
        sl = (slice(by, by+bs), slice(bx, bx+bs)); lb = land[sl]
        preds = {mm: det_preds[mm](cond, t)[0] for mm in LADDER}
        for v in VARS:
            idx = vi[v]
            tr = to_space(v, hr[idx])[sl]; htr = hp(tr)
            for mm in LADDER:
                pm = to_space(v, preds[mm][idx])[sl]
                sk[v][mm].append(skill(hp(pm), htr, lb))
        print(f"[ceil] day {t} done ({time.time()-t0:.0f}s)", flush=True)

    skm = {v: {mm: float(np.mean(sk[v][mm])) for mm in LADDER} for v in VARS}
    print("细尺度 skill (与truth高通相关):")
    for v in VARS:
        print(f"  {VS[v]:7s} " + "  ".join(f"{mm}={skm[v][mm]:.2f}" for mm in LADDER))

    # ---- 图1: 阶梯 skill 柱 ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xg = np.arange(3); bw = 0.2
    for i, mm in enumerate(LADDER):
        ax.bar(xg + (i-1.5)*bw, [skm[v][mm] for v in VARS], bw, color=LCOL[mm], label=LNAME[mm].replace("\n"," "))
    ax.set_xticks(xg); ax.set_xticklabels(["tmax", "tmin", "precip"], fontsize=12)
    ax.set_ylabel("fine-scale skill  (corr of high-pass detail vs truth)")
    ax.set_ylim(0, 1)
    ax.set_title("Complexity ladder: does adding model capacity help?\n"
                 "flat across models = data ceiling reached (not the model's fault)")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/ladder_skill.png", dpi=140, bbox_inches="tight")

    # ---- 图2: 输入 vs 真值 (降水, 说明"答案不在输入里") ----
    t, by, bx, bs = DAYS[0], *TD.pick_land_box((test.full(YEAR, DAYS[0])[2][0] > 0.5), BOX)
    cond, _, m, hr = test.full(YEAR, t); land = m[0] > 0.5
    up = make_bilinear(test.Hl, test.Wl, TD.FACTOR)
    bil = up(test.lr[YEAR][PRECIP][t].astype(np.float32))
    sl = (slice(by, by+bs), slice(bx, bx+bs))
    inp = logmm(bil)[sl]; tru = logmm(hr[vi[PRECIP]])[sl]
    prcmap = LinearSegmentedColormap.from_list("pr", ["#f7fbff","#c6dbef","#6baed6","#2171b5","#08306b","#3f007d"])
    figB, axB = plt.subplots(1, 2, figsize=(11, 4.6))
    vmax = np.percentile(tru, 99.5)
    for axx, arr, ttl in [(axB[0], inp, "ERA5 input (upsampled) — smooth blob"), (axB[1], tru, "Daymet truth — fine cellular structure")]:
        im = axx.imshow(arr, origin="lower", cmap=prcmap, vmin=0, vmax=vmax); axx.set_xticks([]); axx.set_yticks([])
        axx.set_title(ttl, fontsize=12); figB.colorbar(im, ax=axx, fraction=.046, label="log1p(mm)")
    figB.suptitle("The fine precip detail a model must add is simply not in the ERA5 input", fontsize=12)
    figB.tight_layout(); figB.savefig(f"{OUTDIR}/input_vs_truth.png", dpi=140, bbox_inches="tight")

    import json; json.dump(skm, open(f"{OUTDIR}/skill.json", "w"), indent=1)
    print(f"[ceil] DONE {time.time()-t0:.0f}s -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
