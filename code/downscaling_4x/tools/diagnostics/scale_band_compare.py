#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
scale_band_compare.py — 两个确定性落场在各空间尺度带上的相关与方差比
============================================================================
把真值与两个方法的预测场用同一套 FFT 环形带通拆成若干波长带(像素), 逐日在每个带上算
预测与真值的相关系数、预测/真值的方差比、以及该带的误差(绝对值 MAE 与方差 MSE), 全年平均。
回答的是"哪个方法在哪个尺度上更接近真值": 相关看结构(相位), 方差比看振幅, MAE 与 MSE 看
该带在物理单位上贡献了多少误差。

注意两种误差量的可加性不同: 带内 MSE 近似可加(带内合计接近整场 MSE, 差在软边带通重叠、
逐带腐蚀掩膜不同与域均值先减掉), 所以只有 MSE 的柱高能读作"该带对总误差的贡献";
带内 MAE 不可加(三角不等式, 带内合计恒大于整场 MAE), 它给的是该带自身误差的物理量级。

口径:
  - 带通用软边环形滤波(边缘 15% 余弦过渡), 波长边界按像素给, 域均值先减掉;
  - 海洋用掩膜感知的高斯平滑外推填充, 避免海岸线阶跃泄漏到高波数; 统计只在按该带最长
    波长腐蚀过的陆地像素上做(腐蚀半径见 ERODE), 两方法与真值同一掩膜;
  - 相关给两种: 逐日相关的平均, 与全年合并(Σxy/√(Σxx·Σyy)); 方差比 = 预测带方差 / 真值带方差。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.scale_band_compare \\
      --a JiT-dense=runs/exp/<run_a> --b CorrDiff-A=runs/exp/<run_b> \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation.render.context import RenderContext

BANDS = [(2, 4), (4, 8), (8, 16), (16, 64), (64, None)]        # 波长带 [λ_lo, λ_hi) 像素; None = 无上限
ERODE = {2: 2, 4: 3, 8: 5, 16: 12, 64: 24}                      # 各带统计前的陆地腐蚀半径(px), 按 λ_lo 取
FILL_SIGMA = 6.0                                                 # 海洋填充的外推平滑尺度(px)
# 带内 MAE 不可加: L1 对低振幅带的相对权重按振幅(而非振幅平方)计, 带内合计恒大于整场 MAE,
# 且合计的方法排名可与整场 MAE 相反; 图上常驻此提示, 避免把逐带 MAE 读成对总误差的分解。
NOT_ADDITIVE = "band MAE is not additive: Σ bands ≠ full-field MAE"


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def band_label(lo, hi):
    return f"{lo}–{hi} px" if hi else f">{lo} px"


def ring_filters(H, W, bands, taper=0.15):
    """(nb, H, W//2+1) 的 rfft2 权重: 波数 k=1/λ 的软边环。"""
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    out = []
    for lo, hi in bands:
        k_hi = 1.0 / lo                       # 短波长 -> 高波数
        k_lo = 1.0 / hi if hi else 0.0
        w = np.ones_like(k)
        if k_lo > 0:
            a, b = k_lo * (1 - taper), k_lo * (1 + taper)
            w = np.where(k < a, 0.0, np.where(k < b, 0.5 * (1 - np.cos(np.pi * (k - a) / (b - a))), w))
        a, b = k_hi * (1 - taper), k_hi * (1 + taper)
        w = np.where(k > b, 0.0, np.where(k > a, 0.5 * (1 + np.cos(np.pi * (k - a) / (b - a))), w))
        if k_lo == 0:
            w[0, 0] = 0.0                     # 域均值已减, 保险起见去 DC
        out.append(w)
    return np.stack(out, 0)


def fill_ocean(f, land, m_smooth, sigma=FILL_SIGMA):
    """掩膜感知外推: 陆地值不动, 海洋取 G*(f·m)/G*m; 全场再减陆地均值。"""
    mean = float(f[land].mean())
    g = np.where(land, f - mean, 0.0)
    num = gaussian_filter(g, sigma)
    fill = num / np.maximum(m_smooth, 1e-6)
    return np.where(land, f - mean, fill)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="label=run_dir, 落场取 <run_dir>/fields<year>/<target>")
    ap.add_argument("--b", required=True, help="label=run_dir")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()

    la, da = parse_spec(a.a)
    lb, db = parse_spec(a.b)
    y = a.year
    fa = da / f"fields{y}" / a.target
    fb = db / f"fields{y}" / a.target
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(fa, a.regions, a.target, out, years=(y,), model_tag=f"{slug(la)}-vs-{slug(lb)}")
    land = ctx.land
    H, W = land.shape
    unit = ctx.unit
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    filt = ring_filters(H, W, BANDS)
    masks = [binary_erosion(land, iterations=ERODE[lo]) for lo, _ in BANDS]
    m_smooth = gaussian_filter(land.astype(np.float64), FILL_SIGMA)
    nb = len(BANDS)
    labels = [la, lb]
    # 累加: 逐日相关之和, 合并相关的分量, 方差比之和, 误差方差之和, 真值方差之和
    daily_corr = np.zeros((2, nb, nd)); daily_vr = np.zeros((2, nb, nd))
    daily_mse = np.zeros((2, nb, nd)); daily_mae = np.zeros((2, nb, nd))
    sxy = np.zeros((2, nb)); sxx = np.zeros((2, nb)); syy = np.zeros(nb); tvar = np.zeros((nb, nd))
    t0 = time.time()
    for t in range(nd):
        tr = ctx.truth(y, t)
        fields = {"truth": tr, la: np.load(fa / "ens_mean" / f"{y}_d{t}.npy"), lb: np.load(fb / "ens_mean" / f"{y}_d{t}.npy")}
        spec = {}
        for k, f in fields.items():
            f = np.nan_to_num(np.asarray(f, np.float64), nan=0.0)
            spec[k] = np.fft.rfft2(fill_ocean(f, land, m_smooth))
        for b in range(nb):
            m = masks[b]
            tb = np.fft.irfft2(spec["truth"] * filt[b], s=(H, W))[m]
            tb = tb - tb.mean()
            tvar[b, t] = tb.var()
            syy[b] += float((tb * tb).sum())
            for i, lab in enumerate(labels):
                pb = np.fft.irfft2(spec[lab] * filt[b], s=(H, W))[m]
                pb = pb - pb.mean()
                daily_corr[i, b, t] = float((pb * tb).mean() / max(np.sqrt(pb.var() * tb.var()), 1e-12))
                daily_vr[i, b, t] = float(pb.var() / max(tb.var(), 1e-12))
                daily_mse[i, b, t] = float(((pb - tb) ** 2).mean())
                daily_mae[i, b, t] = float(np.abs(pb - tb).mean())
                sxy[i, b] += float((pb * tb).sum()); sxx[i, b] += float((pb * pb).sum())
        if t % 30 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    res = {"a": la, "b": lb, "target": a.target, "year": y, "n_days": nd, "unit": unit,
           "bands_px": [[lo, hi] for lo, hi in BANDS], "erode_px": [ERODE[lo] for lo, _ in BANDS],
           "truth_band_variance_share": (tvar.mean(1) / tvar.mean(1).sum()).round(4).tolist(),
           "per_model": {}}
    for i, lab in enumerate(labels):
        res["per_model"][lab] = {
            "corr_daily_mean": daily_corr[i].mean(1).round(4).tolist(),
            "corr_daily_std": daily_corr[i].std(1).round(4).tolist(),
            "corr_pooled": (sxy[i] / np.sqrt(np.maximum(sxx[i] * syy, 1e-12))).round(4).tolist(),
            "var_ratio_daily_mean": daily_vr[i].mean(1).round(4).tolist(),
            "var_ratio_pooled": (sxx[i] / np.maximum(syy, 1e-12)).round(4).tolist(),
            "band_mse_daily_mean": daily_mse[i].mean(1).round(5).tolist(),
            "band_mae_daily_mean": daily_mae[i].mean(1).round(5).tolist(),
        }
    res["share_days_a_higher_corr"] = (daily_corr[0] > daily_corr[1]).mean(1).round(3).tolist()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(out / "daily.npz", daily_corr=daily_corr, daily_vr=daily_vr, daily_mse=daily_mse,
                        daily_mae=daily_mae, tvar=tvar, labels=np.array(labels))

    # ---- 图: 相关随波段; 方差比随波段 ----
    xl = [band_label(lo, hi) for lo, hi in BANDS]
    xs = np.arange(nb)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for i, lab in enumerate(labels):
        mu, sd = daily_corr[i].mean(1), daily_corr[i].std(1)
        ax.errorbar(xs, mu, yerr=sd, marker="o", lw=1.3, capsize=3, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band"); ax.set_ylabel("correlation with truth in band (daily mean ± std)")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(f"band-wise correlation · {a.target} · {y}", fontsize=10)
    ctx.savefig(fig, "band_corr")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for i, lab in enumerate(labels):
        mu, sd = daily_vr[i].mean(1), daily_vr[i].std(1)
        ax.errorbar(xs, mu, yerr=sd, marker="o", lw=1.3, capsize=3, label=lab)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band"); ax.set_ylabel("variance ratio pred / truth in band (daily mean ± std)")
    ax.legend(fontsize=8)
    ax.set_title(f"band-wise variance ratio · {a.target} · {y}", fontsize=10)
    ctx.savefig(fig, "band_var_ratio")
    # ---- 图: 逐日配对差(去掉日间共同波动), 95% 置信区间 ----
    dcorr = daily_corr[0] - daily_corr[1]                       # (nb, nd)  >0: A 更高
    mu, ci = dcorr.mean(1), 1.96 * dcorr.std(1, ddof=1) / np.sqrt(nd)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(xs, mu, yerr=ci, color=np.where(mu >= 0, "#4878a8", "#c0504d"), capsize=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for i in range(nb):
        ax.text(i, mu[i] + (ci[i] if mu[i] >= 0 else -ci[i]) * 1.15, f"{res['share_days_a_higher_corr'][i]:.0%} days",
                ha="center", va="bottom" if mu[i] >= 0 else "top", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band"); ax.set_ylabel(f"Δ correlation ({la} − {lb}), daily paired mean ± 95% CI")
    ax.set_title(f"band-wise correlation difference · {a.target} · {y} (label: share of days {la} higher)", fontsize=9)
    ctx.savefig(fig, "band_corr_paired_diff")
    mse_ratio = daily_mse[0] / np.maximum(daily_mse[1], 1e-12)
    lr_mu, lr_ci = np.log(mse_ratio).mean(1), 1.96 * np.log(mse_ratio).std(1, ddof=1) / np.sqrt(nd)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.errorbar(xs, np.exp(lr_mu), yerr=[np.exp(lr_mu) - np.exp(lr_mu - lr_ci), np.exp(lr_mu + lr_ci) - np.exp(lr_mu)],
                marker="o", lw=1.3, capsize=3, color="#4878a8")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band"); ax.set_ylabel(f"band error variance ratio {la} / {lb} (geometric daily mean ± 95% CI)")
    ax.set_title(f"band-wise error ratio · {a.target} · {y} (>1: {la} worse)", fontsize=9)
    ctx.savefig(fig, "band_mse_ratio")
    # ---- 图: 逐带绝对误差方差差, 与整场误差同量纲; 柱高即该带对总误差差额的实际贡献 ----
    dmse = daily_mse[0] - daily_mse[1]                          # (nb, nd)  <0: A 该带误差更小
    dm_mu, dm_ci = dmse.mean(1), 1.96 * dmse.std(1, ddof=1) / np.sqrt(nd)
    share = res["truth_band_variance_share"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(xs, dm_mu, yerr=dm_ci, color=np.where(dm_mu <= 0, "#4878a8", "#c0504d"), capsize=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for i in range(nb):
        ax.text(i, dm_mu[i] + (dm_ci[i] if dm_mu[i] >= 0 else -dm_ci[i]) * 1.15,
                f"{share[i]:.1%} of truth var", ha="center",
                va="bottom" if dm_mu[i] >= 0 else "top", fontsize=8)
    ax.margins(y=0.18)
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band")
    ax.set_ylabel(f"Δ band error variance ({la} − {lb}) [{unit}²]\ndaily paired mean ± 95% CI")
    ax.set_title(f"band-wise error contribution · {a.target} · {y} "
                 f"(<0: {la} better; Σ bands {dm_mu.sum():+.3f} {unit}²)", fontsize=9)
    ctx.savefig(fig, "band_mse_diff")
    res["band_mse_diff_mean"] = dm_mu.round(5).tolist()
    res["band_mse_diff_ci95"] = dm_ci.round(5).tolist()
    res["band_mse_sum"] = {lab: float(round(daily_mse[i].mean(1).sum(), 5)) for i, lab in enumerate(labels)}
    res["share_days_a_lower_band_mse"] = (daily_mse[0] < daily_mse[1]).mean(1).round(3).tolist()
    # ---- 图: 逐带 MAE 绝对量(该带误差的物理量级, 不可加) ----
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for i, lab in enumerate(labels):
        mu_, sd_ = daily_mae[i].mean(1), daily_mae[i].std(1)
        ax.errorbar(xs, mu_, yerr=sd_, marker="o", lw=1.3, capsize=3, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band"); ax.set_ylabel(f"band MAE [{unit}] (daily mean ± std)")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    ax.text(0.98, 0.04, NOT_ADDITIVE, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.35")
    ax.set_title(f"band-wise MAE · {a.target} · {y}", fontsize=10)
    ctx.savefig(fig, "band_mae")
    # ---- 图: 逐带 MAE 的逐日配对差, 与整场 MAE 同单位 ----
    dmae = daily_mae[0] - daily_mae[1]                          # (nb, nd)  <0: A 该带 MAE 更小
    da_mu, da_ci = dmae.mean(1), 1.96 * dmae.std(1, ddof=1) / np.sqrt(nd)
    win_mae = (daily_mae[0] < daily_mae[1]).mean(1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(xs, da_mu, yerr=da_ci, color=np.where(da_mu <= 0, "#4878a8", "#c0504d"), capsize=3)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for i in range(nb):
        ax.text(i, da_mu[i] + (da_ci[i] if da_mu[i] >= 0 else -da_ci[i]) * 1.15,
                f"{win_mae[i]:.0%} days", ha="center",
                va="bottom" if da_mu[i] >= 0 else "top", fontsize=8)
    ax.margins(y=0.18)
    ax.set_xticks(xs); ax.set_xticklabels(xl)
    ax.set_xlabel("wavelength band")
    ax.set_ylabel(f"Δ band MAE ({la} − {lb}) [{unit}]\ndaily paired mean ± 95% CI")
    ax.text(0.98, 0.96, NOT_ADDITIVE, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="0.35")
    ax.set_title(f"band-wise MAE difference · {a.target} · {y} "
                 f"(<0: {la} better; label: share of days {la} lower)", fontsize=9)
    ctx.savefig(fig, "band_mae_diff")
    mae_ratio = daily_mae[0] / np.maximum(daily_mae[1], 1e-12)
    res["band_mae_diff_mean"] = da_mu.round(5).tolist()
    res["band_mae_diff_ci95"] = da_ci.round(5).tolist()
    res["band_mae_ratio_geomean"] = np.exp(np.log(mae_ratio).mean(1)).round(4).tolist()
    res["band_mae_sum"] = {lab: float(round(daily_mae[i].mean(1).sum(), 5)) for i, lab in enumerate(labels)}
    res["share_days_a_lower_band_mae"] = win_mae.round(3).tolist()
    res["corr_paired_diff_mean"] = mu.round(4).tolist()
    res["corr_paired_diff_ci95"] = ci.round(4).tolist()
    res["band_mse_ratio_geomean"] = np.exp(lr_mu).round(4).tolist()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    ctx.write_scales()
    for lab in labels:
        r = res["per_model"][lab]
        print(f"{lab}: corr {r['corr_daily_mean']}  var_ratio {r['var_ratio_daily_mean']}")
    print("truth band variance share:", res["truth_band_variance_share"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
