#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
EDA ① 任务局部性 / 数据机制诊断  (登录节点可跑, 纯 numpy, 不碰 GPU)
------------------------------------------------------------------
问题: ViT<UNet 到底是"任务天生偏卷积(局部)"还是"我们把 ViT 切窗做残了(全局被丢)"。
方法: 对 test 年 2020 抽样若干天, 在物理空间比较
  (a) 径向功率谱  truth vs 双线性(ERA5 upsample): 二者之差 = 降尺度必须补的信号,
      看它落在哪些空间尺度(相对 60px tile / 6px ERA5 格)。
  (b) 残差 (truth - 双线性) 的径向空间自相关, 求 1/e 相关长度 ℓ。
      ℓ << 60px  => 残差是局部结构, tile 内就能覆盖, 全局注意力无用 => 任务偏卷积。
      ℓ ~ 数百px => 有大尺度结构, 切窗会伤 ViT => 协议问题, 该整帧比。
输出: runs/exp/20260720-eda-task-locality/{psd_acf.png, summary.json}
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import make_bilinear
from era5_daymet.training import train_downscale as TD

STATS   = "/lustre/orion/atm112/scratch/hjsong/downscaling/runs/stats/train_dayofyear"
OUTDIR  = "/lustre/orion/atm112/scratch/hjsong/downscaling/runs/exp/20260720-eda-task-locality"
YEAR    = 2020
STRIDE  = 1                       # 1=全年365天(汇报口径); 调大只为快速探索
                                  # (旧值 30 是"对齐 13 天约定", 那个约定已被证实是失误, 勿再沿用)
BOX     = 384                     # 陆地方框边长(px, 与 eval_spectrum 一致), 6px=1 ERA5格, 60px=tile
FACTOR  = 6
VARS    = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
LABEL   = {"2m_temperature_max": "tmax (K)", "2m_temperature_min": "tmin (K)",
           "total_precipitation_24hr": "precip (m/day)"}
os.makedirs(OUTDIR, exist_ok=True)


def radial_mean(P):
    """P 的中心径向平均, 返回 shape (rmax,) 从 r=0 起。"""
    c0, c1 = np.array(P.shape) // 2
    Y, X = np.indices(P.shape)
    r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def psd_2d(img):
    """无窗? 用 Hann 窗抑制泄漏(与 TD.radial_psd 一致), 返回 2D |FFT|^2 (fftshift)."""
    f = img - img.mean()
    w = np.hanning(f.shape[0])[:, None] * np.hanning(f.shape[1])[None, :]
    return np.abs(np.fft.fftshift(np.fft.fft2(f * w))) ** 2


def autocov_2d(img):
    """径向自协方差(Wiener-Khinchin), 归一化到 r=0 为 1。"""
    f = img - img.mean()
    F = np.fft.fft2(f)
    ac = np.fft.ifft2(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)
    ac /= ac.flat[np.argmax(ac)]                       # 峰值(r=0)归一
    return ac


def corr_length(acf_r):
    """径向 ACF 首次跌破 1/e 的半径(px), 线性插值。"""
    thr = 1.0 / np.e
    for i in range(1, len(acf_r)):
        if acf_r[i] < thr:
            a, b = acf_r[i - 1], acf_r[i]
            return (i - 1) + (a - thr) / (a - b + 1e-12)
    return float(len(acf_r) - 1)


def main():
    stats = TD.Stats(STATS, VARS, VARS)
    print(f"[eda] loading year {YEAR} ...", flush=True)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [YEAR], VARS, VARS, stats)
    up = make_bilinear(d.Hl, d.Wl, FACTOR)
    mask = d.mask[YEAR]
    by, bx, bs = TD.pick_land_box(mask, min(BOX, d.H, d.W))
    print(f"[eda] land box (y,x,sz)=({by},{bx},{bs})  all-land={mask[by:by+bs,bx:bx+bs].all()}", flush=True)
    days = list(range(0, d.ndays[YEAR], STRIDE))
    print(f"[eda] {len(days)} days: {days}", flush=True)

    res = {}
    for v in VARS:
        psd_t = psd_a = acf = None
        for t in days:
            truth = np.asarray(d._hr(YEAR, v, t)).astype(np.float32)
            bil = up(d.lr[YEAR][v][t].astype(np.float32))
            ct = truth[by:by+bs, bx:bx+bs]
            cb = bil[by:by+bs, bx:bx+bs]
            cr = ct - cb
            psd_t = radial_mean(psd_2d(ct)) if psd_t is None else psd_t + radial_mean(psd_2d(ct))
            psd_a = radial_mean(psd_2d(cb)) if psd_a is None else psd_a + radial_mean(psd_2d(cb))
            a = radial_mean(autocov_2d(cr))
            acf = a if acf is None else acf + a
        n = len(days)
        psd_t /= n; psd_a /= n; acf /= n
        ell = corr_length(acf)
        # tile(60px)对应波数 k = bs/60; ERA5格(6px): k = bs/6
        res[v] = dict(psd_truth=psd_t.tolist(), psd_bilinear=psd_a.tolist(),
                      acf=acf.tolist(), corr_len_px=float(ell),
                      corr_len_era5cells=float(ell / FACTOR), corr_len_over_tile60=float(ell / 60))
        print(f"[eda] {v:32s} corr_len ℓ = {ell:6.1f} px = {ell/FACTOR:5.1f} ERA5格 = {ell/60:4.2f}×tile", flush=True)

    # ---- 画图 ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    k_tile = bs / 60.0          # tile(60px hi-res)对应波数
    k_era5 = bs / 6.0           # ERA5 原生格(6px hi-res)
    for j, v in enumerate(VARS):
        pt = np.array(res[v]["psd_truth"]); pa = np.array(res[v]["psd_bilinear"])
        k = np.arange(1, len(pt))
        a0 = ax[0, j]
        a0.loglog(k, pt[1:], label="Daymet truth", lw=2)
        a0.loglog(k, pa[1:], label="bilinear(ERA5)", lw=2)
        a0.axvline(k_tile, color="green", ls="--", alpha=.7, label="60px tile scale")
        a0.axvline(k_era5, color="gray", ls=":", alpha=.7, label="6px ERA5-grid scale")
        a0.set_title(f"PSD  {LABEL[v]}"); a0.set_xlabel("radial wavenumber k (cyc/box)")
        a0.grid(True, which="both", alpha=.3); a0.legend(fontsize=8)
        # 残差谱占比: truth-bilinear 功率在各尺度
        gap = np.clip(pt - pa, 0, None)
        a1 = ax[1, j]
        acf = np.array(res[v]["acf"]); rr = np.arange(len(acf))
        a1.plot(rr, acf, lw=2)
        a1.axhline(1/np.e, color="red", ls="--", alpha=.6, label="1/e")
        a1.axvline(res[v]["corr_len_px"], color="purple", ls="-", alpha=.7,
                   label=f"corr-len $\\ell$={res[v]['corr_len_px']:.0f}px ({res[v]['corr_len_px']/6:.1f} ERA5-cells)")
        a1.axvline(60, color="green", ls="--", alpha=.7, label="60px tile")
        a1.set_xlim(0, min(200, len(acf))); a1.set_ylim(-0.1, 1.02)
        a1.set_title(f"residual ACF  {LABEL[v]}"); a1.set_xlabel("lag (px, hi-res)")
        a1.grid(True, alpha=.3); a1.legend(fontsize=8)
    fig.suptitle(f"EDA-1  task locality (ERA5->Daymet)   year={YEAR}   {len(days)} days   land-box={bs}px\n"
                 f"top: power spectrum (truth vs bilinear)   bottom: residual autocorrelation",
                 fontsize=12)
    fig.tight_layout()
    fp = os.path.join(OUTDIR, "psd_acf.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    json.dump({"year": YEAR, "days": days, "box": bs, "vars": res},
              open(os.path.join(OUTDIR, "summary.json"), "w"))
    print(f"[eda] saved -> {fp}", flush=True)


if __name__ == "__main__":
    main()
