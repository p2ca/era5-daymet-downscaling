#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
统一跨方法诊断 (登录节点 GPU, 复用 plot_model_maps.build) —— 用同一把尺子横比四法。
框架: 每个变量的【功率谱】把 truth + bilinear + BCSD + UNet + ViT + CorrDiff(均值/成员) 放同一张图;
     单数字【结构保真度】= 各方法高频功率 / truth 高频功率 (k>tile尺度), 越低=越平滑。
温度用 K, 降水用 log1p(mm)。BCSD 用已存系数解析计算(a*bilinear+b, 各自空间)。
输出: runs/exp/20260720-eda-unified/{unified_spectra.png, scorecard.json}
"""
import os, sys, json, time
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import make_bilinear
from era5_daymet.tools.plotting import plot_model_maps as PM
from era5_daymet.training import train_downscale as TD

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUTDIR = f"{W}/runs/exp/20260720-eda-unified"
# DAYS 覆盖全年(汇报口径, stride=1)。★注意: 每天要做 NMEM 成员 corrdiff 集合采样, 全年很贵;
# 快速探索可临时改回 list(range(0, 365, 40)) 之类的子采样(但结果不可作为全年汇报)。
YEAR = 2020; DAYS = list(range(0, 365)); NMEM = 8; BOX = 384; FACTOR = 6
PRECIP = TD.PRECIP
TVARS = ["2m_temperature_max", "2m_temperature_min", PRECIP]
LAB = {"2m_temperature_max": "tmax (K)", "2m_temperature_min": "tmin (K)", PRECIP: "precip  log1p(mm)"}
os.makedirs(OUTDIR, exist_ok=True)
logmm = lambda x: np.log1p(np.maximum(x, 0.0) * 1000.0)

METHODS = ["bilinear", "bcsd", "unet", "vit", "cd_mean", "cd_member"]
STYLE = {  # (color, lw, ls, label)
    "truth":     ("#111111", 2.4, "-",  "Truth"),
    "bilinear":  ("#9aa0a6", 1.4, ":",  "Bilinear (ref)"),
    "bcsd":      ("#7b4fa0", 1.8, "-",  "BCSD"),
    "unet":      ("#1f77b4", 1.8, "-",  "UNet"),
    "vit":       ("#ff7f0e", 1.8, "-",  "ViT"),
    "cd_mean":   ("#2ca02c", 1.8, "--", "CorrDiff mean"),
    "cd_member": ("#d62728", 2.2, "-",  "CorrDiff member"),
}


def radial_mean(P):
    c0, c1 = np.array(P.shape) // 2; Y, X = np.indices(P.shape)
    r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def psd(img):
    f = img - img.mean(); w = np.hanning(f.shape[0])[:, None] * np.hanning(f.shape[1])[None, :]
    return radial_mean(np.abs(np.fft.fftshift(np.fft.fft2(f * w))) ** 2)


def to_space(v, phys):
    return logmm(phys) if v == PRECIP else phys       # 温度 K, 降水 log1p(mm)


def main():
    t0 = time.time()
    args = SimpleNamespace(
        year=YEAR, stats_dir=f"{W}/runs/stats/train_dayofyear",
        era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR,
        unet_dir=f"{W}/runs/exp/20260711-unet-b64",
        vit_dir=f"{W}/runs/exp/20260712-vit-d384-b16-ep12",
        bcsd_coef_dir=f"{W}/runs/bcsd_coefs",
        corrdiff_dir=f"{W}/runs/exp/20260714-corrdiff-b64",
        regressor_ckpt=f"{W}/runs/exp/20260711-unet-b64/ckpt.pt")
    device, stats, test, out_vars, det_preds, corrdiff_pred = PM.build(args)
    vi = {v: out_vars.index(v) for v in TVARS}
    up = make_bilinear(test.Hl, test.Wl, FACTOR)
    coefs = {v: np.load(f"{W}/runs/bcsd_coefs/{v}.npz") for v in TVARS}

    acc = {v: {k: None for k in ["truth"] + METHODS} for v in TVARS}
    for t in DAYS:
        cond, _, m, hr = test.full(YEAR, t)
        land = (m[0] if m.ndim == 3 else m) > 0.5
        by, bx, bs = TD.pick_land_box(land, min(BOX, test.H, test.W))
        cb = (slice(by, by+bs), slice(bx, bx+bs))
        cond_b = cond[:, by:by+bs, bx:bx+bs]
        m_b = m[:, by:by+bs, bx:bx+bs] if m.ndim == 3 else m[by:by+bs, bx:bx+bs][None]
        # 各方法全帧 bilinear / bcsd (系数在全帧位置), 再裁框
        unet = det_preds["unet"](cond_b, t)[0]
        vit = det_preds["vit"](cond_b, t)[0]
        mem = corrdiff_pred(cond_b, m_b, NMEM)          # (N,Cout,bs,bs)
        cd_mean = mem.mean(0)
        for v in TVARS:
            idx = vi[v]; a = coefs[v]["a"][cb]; b = coefs[v]["b"][cb]
            bil_full = up(test.lr[YEAR][v][t].astype(np.float32))
            bil_b = bil_full[by:by+bs, bx:bx+bs]
            bil_sp = to_space(v, bil_b)                 # bilinear in display/fit space
            bcsd_sp = a * bil_sp + b                    # BCSD = a*bilinear+b (同拟合空间)
            fields = {
                "truth":    to_space(v, hr[idx, by:by+bs, bx:bx+bs]),
                "bilinear": bil_sp,
                "bcsd":     bcsd_sp,
                "unet":     to_space(v, unet[idx]),
                "vit":      to_space(v, vit[idx]),
                "cd_mean":  to_space(v, cd_mean[idx]),
            }
            for k, arr in fields.items():
                p = psd(arr); acc[v][k] = p if acc[v][k] is None else acc[v][k] + p
            mp = np.mean([psd(to_space(v, mem[j, idx])) for j in range(NMEM)], 0)  # 成员PSD均值
            acc[v]["cd_member"] = mp if acc[v]["cd_member"] is None else acc[v]["cd_member"] + mp
        print(f"[uni] day {t} done ({time.time()-t0:.0f}s)", flush=True)

    nd = len(DAYS)
    for v in TVARS:
        for k in acc[v]:
            acc[v][k] = acc[v][k] / nd

    # 细尺度保真: 只看 <=30px(约半个tile以内)的真·细结构 -> 避开中频混淆
    FINE_PX = 30; kt = int(round(BOX / FINE_PX))        # k>=13
    fidelity = {v: {} for v in TVARS}
    for v in TVARS:
        tr_hf = acc[v]["truth"][kt:].sum()
        for k in METHODS:
            fidelity[v][k] = float(acc[v][k][kt:].sum() / max(tr_hf, 1e-12))
    # 存盘径向谱(以后调指标/出图无需重跑采样)
    np.savez_compressed(os.path.join(OUTDIR, "psd_arrays.npz"),
                        **{f"{v}__{k}": acc[v][k] for v in TVARS for k in ["truth"] + METHODS},
                        box=BOX, fine_px=FINE_PX, kt=kt)

    # ---- 图: 3 谱 + 1 结构保真柱 ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    axes = {"2m_temperature_max": ax[0, 0], "2m_temperature_min": ax[0, 1], PRECIP: ax[1, 0]}
    for v, a in axes.items():
        P = acc[v]; k = np.arange(1, len(P["truth"]))
        for key in ["truth"] + METHODS:
            c, lw, ls, lab = STYLE[key]
            a.loglog(k, P[key][1:], color=c, lw=lw, ls=ls, label=lab)
        a.axvline(BOX/60, color="green", ls=":", alpha=.5)
        a.set_title(f"power spectrum  {LAB[v]}"); a.set_xlabel("radial wavenumber k (cyc/box)")
        a.grid(True, which="both", alpha=.25); a.legend(fontsize=7.5, ncol=2)
    # 结构保真柱
    ab = ax[1, 1]; groups = ["tmax", "tmin", "precip"]; xg = np.arange(3)
    bw = 0.15
    for i, k in enumerate(METHODS):
        vals = [fidelity[v][k] for v in TVARS]
        ab.bar(xg + (i - 2) * bw, vals, bw, color=STYLE[k][0], label=STYLE[k][3])
    ab.axhline(1.0, color="#111", ls="--", lw=1, alpha=.7); ab.text(2.35, 1.02, "truth=1", fontsize=8)
    ab.set_xticks(xg); ab.set_xticklabels(groups); ab.set_ylim(0, 1.25)
    ab.set_ylabel("fine-scale fidelity  (<=30px power / truth)")
    ab.set_title("fine-scale structure fidelity  (1.0 = matches truth, lower = smoother)")
    ab.legend(fontsize=7.5, ncol=2); ab.grid(True, axis="y", alpha=.25)
    fig.suptitle(f"Unified cross-method diagnostic — power spectra & structure fidelity   "
                 f"year={YEAR}  {nd} days  box={BOX}px  (temp: K, precip: log1p(mm))", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "unified_spectra.png"), dpi=130, bbox_inches="tight")

    json.dump(dict(year=YEAR, days=DAYS, nmem=NMEM, box=BOX, structure_fidelity=fidelity),
              open(os.path.join(OUTDIR, "scorecard.json"), "w"), indent=1)
    print("\n结构保真度 (高频功率/truth):")
    print(f"{'method':14s} {'tmax':>7s} {'tmin':>7s} {'precip':>7s}")
    for k in METHODS:
        print(f"{k:14s} " + " ".join(f"{fidelity[v][k]:7.2f}" for v in TVARS))
    print(f"[uni] DONE {time.time()-t0:.0f}s -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
