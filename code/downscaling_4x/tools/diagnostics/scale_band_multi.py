#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
scale_band_multi.py — 任意多个模型的尺度分带对比(逐波长带 MAE / 相关 / 方差比)
============================================================================
scale_band_compare 的多模型版: 波长带、软边环形带通、海洋填充、逐带腐蚀掩膜等口径全部
从那份实现里引用, 不另起一套; 唯一的区别是模型数量不限, 且落场目录允许两种布局
(阶段A 的 <run>/fields<year>/<target>/ 与阶段B 落场的 <eval dir>/, 按有没有 ens_mean 自动判断)。

出图(全部单图, 多条线):
  band_mae    逐带 MAE(绝对量)
  band_mse    逐带误差方差
  band_corr   逐带预测-真值相关(逐日平均)
  band_var_ratio  逐带 预测方差 / 真值方差 —— 振幅是否够
给了 --ref 时另出相对该模型的配对差(逐日 A−ref 的均值与 95% CI)。

★带内 MAE 不可加★: Σ 各带 ≠ 整场 MAE, 且合计的排名可与整场相反; 只作各带的量级与相对优劣读数。
★CRPS 无法分带★: 分带要对每个集合成员做带通后再算 CRPS, 而落场只存 ens_mean/crps/spread/rank,
成员从未落盘; 把标量 crps 场做带通没有意义。要 band CRPS 必须先改采样入口落成员再重采。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.scale_band_multi \\
      --model CA=runs/exp/<corrdiffA run> --model JiT=runs/exp/<jda run> \\
      --model CB-CA=runs/exp/<cb eval2020> --model DEC=runs/exp/<dec eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag> [--ref CA]
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
from downscaling_4x.tools.diagnostics.scale_band_compare import (
    BANDS, ERODE, FILL_SIGMA, NOT_ADDITIVE, band_label, fill_ocean, ring_filters, slug)


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def resolve_fields(d, year, target):
    """两种落场布局: 目录里直接有 ens_mean/, 或者是 <run>/fields<year>/<target>/。"""
    if (d / "ens_mean").is_dir():
        return d
    alt = d / f"fields{year}" / target
    if (alt / "ens_mean").is_dir():
        return alt
    raise SystemExit(f"{d} 下找不到 ens_mean/(也不在 fields{year}/{target}/ 下)")


def lines(ctx, x, series, name, ylabel, title, note=None, hline=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for lab, (v, lo, hi) in series.items():
        ax.plot(x, v, marker="o", ms=4, lw=1.8, label=lab)
        if lo is not None:
            ax.fill_between(x, lo, hi, alpha=0.15, lw=0)
    if hline is not None:
        ax.axhline(hline, color="0.35", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([band_label(lo, hi) for lo, hi in BANDS], fontsize=9)
    ax.set_xlabel("wavelength band")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    if note:
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.35")
    return ctx.savefig(fig, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=dir, 可多次")
    ap.add_argument("--ref", default=None, help="出配对差时的参考模型标签")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--region", type=int, action="append", default=None,
                    help="额外按区域展示编号各出一套图(带通仍在整幅上做, 只把统计限制在该区域的格点上)")
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    if a.ref and a.ref not in labels:
        raise SystemExit(f"--ref {a.ref} 不在模型列表 {labels} 里")
    y = a.year
    fdirs = {l: resolve_fields(d, y, a.target) for l, d in specs}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(fdirs[labels[0]], a.regions, a.target, out, years=(y,),
                        model_tag="-".join(slug(l) for l in labels)[:60])
    land = ctx.land
    H, W = land.shape
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    filt = ring_filters(H, W, BANDS)
    masks = [binary_erosion(land, iterations=ERODE[lo]) for lo, _ in BANDS]
    m_smooth = gaussian_filter(land.astype(np.float64), FILL_SIGMA)
    nb, nm = len(BANDS), len(labels)

    # 统计域: 整幅有效域, 外加按需的逐区域; 带通一律在整幅上做, 区域只限制取哪些格点
    order = np.argsort([int(i) for i in ctx.region_ids])
    doms = [("", "whole domain", np.ones_like(land, bool))]
    for rd in (a.region or []):
        if str(rd) not in ctx.region_ids:
            raise SystemExit(f"区域展示编号 {rd} 不在分区里")
        i = ctx.region_ids.index(str(rd))
        doms.append((f"_region{rd}", f"region {rd} {ctx.region_names[i]}", ctx.region_id == (i + 1)))
    dmask = [[masks[b] & dm for b in range(nb)] for _, _, dm in doms]
    for di, (sfx, nmd, _) in enumerate(doms):
        npx = [int(dmask[di][b].sum()) for b in range(nb)]
        if min(npx) < 100:
            raise SystemExit(f"{nmd} 在最粗的带上只剩 {min(npx)} 个格点, 不足以统计")
    nD = len(doms)

    d_corr = np.zeros((nD, nm, nb, nd)); d_vr = np.zeros((nD, nm, nb, nd))
    d_mse = np.zeros((nD, nm, nb, nd)); d_mae = np.zeros((nD, nm, nb, nd))
    tvar = np.zeros((nD, nb, nd))
    t0 = time.time()
    for t in range(nd):
        tr = ctx.truth(y, t)
        spec = {}
        for k, f in [("truth", tr)] + [(l, np.load(fdirs[l] / "ens_mean" / f"{y}_d{t}.npy")) for l in labels]:
            f = np.nan_to_num(np.asarray(f, np.float64), nan=0.0)
            spec[k] = np.fft.rfft2(fill_ocean(f, land, m_smooth))
        for b in range(nb):
            tf = np.fft.irfft2(spec["truth"] * filt[b], s=(H, W))
            pf = [np.fft.irfft2(spec[lab] * filt[b], s=(H, W)) for lab in labels]
            for di in range(nD):
                m = dmask[di][b]
                tb = tf[m]; tb = tb - tb.mean()
                tvar[di, b, t] = tb.var()
                for i in range(nm):
                    pb = pf[i][m]; pb = pb - pb.mean()
                    d_corr[di, i, b, t] = float((pb * tb).mean() / max(np.sqrt(pb.var() * tb.var()), 1e-12))
                    d_vr[di, i, b, t] = float(pb.var() / max(tb.var(), 1e-12))
                    d_mse[di, i, b, t] = float(((pb - tb) ** 2).mean())
                    d_mae[di, i, b, t] = float(np.abs(pb - tb).mean())
        if t % 30 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    x = np.arange(nb)
    unit = ctx.unit
    res = {"models": labels, "target": a.target, "year": y, "n_days": nd, "unit": unit,
           "bands_px": [[lo, hi] for lo, hi in BANDS], "erode_px": [ERODE[lo] for lo, _ in BANDS],
           "note_not_additive": NOT_ADDITIVE,
           "note_no_band_crps": "CRPS 无法分带: 需要对每个集合成员做带通, 而落场未存成员",
           "note_region": "区域统计的带通仍在整幅上做, 只把统计限制在该区域与该带腐蚀掩膜的交集上",
           "domains": {}}
    if a.ref:
        res["ref"] = a.ref

    for di, (sfx, nmd, _) in enumerate(doms):
        share = (tvar[di].mean(1) / tvar[di].mean(1).sum()).round(4).tolist()
        res["domains"][nmd] = {
            "n_pixels_by_band": [int(dmask[di][b].sum()) for b in range(nb)],
            "truth_band_variance_share": share,
            "band_mae": {l: d_mae[di, i].mean(1).round(4).tolist() for i, l in enumerate(labels)},
            "band_mse": {l: d_mse[di, i].mean(1).round(5).tolist() for i, l in enumerate(labels)},
            "band_corr_daily_mean": {l: d_corr[di, i].mean(1).round(4).tolist() for i, l in enumerate(labels)},
            "band_var_ratio": {l: d_vr[di, i].mean(1).round(4).tolist() for i, l in enumerate(labels)}}
        ttl = f"{a.target} {y} · {nmd}"
        lines(ctx, x, {l: (d_mae[di, i].mean(1), None, None) for i, l in enumerate(labels)},
              f"band_mae{sfx}", f"MAE [{unit}]", f"{ttl} · MAE by wavelength band", note=NOT_ADDITIVE)
        lines(ctx, x, {l: (d_mse[di, i].mean(1), None, None) for i, l in enumerate(labels)},
              f"band_mse{sfx}", f"error variance [{unit}²]", f"{ttl} · error variance by band")
        lines(ctx, x, {l: (d_corr[di, i].mean(1), None, None) for i, l in enumerate(labels)},
              f"band_corr{sfx}", "correlation with truth", f"{ttl} · band correlation (daily mean)")
        lines(ctx, x, {l: (d_vr[di, i].mean(1), None, None) for i, l in enumerate(labels)},
              f"band_var_ratio{sfx}", "predicted / truth variance",
              f"{ttl} · amplitude ratio by band", hline=1.0)
        if a.ref:
            r = labels.index(a.ref)
            for nm_, arr, yl in (("mae", d_mae, f"MAE difference [{unit}]"),
                                 ("corr", d_corr, "correlation difference")):
                ser = {}
                for i, l in enumerate(labels):
                    if i == r:
                        continue
                    dd = arr[di, i] - arr[di, r]
                    mu = dd.mean(1); se = dd.std(1, ddof=1) / np.sqrt(nd)
                    ser[f"{l} − {a.ref}"] = (mu, mu - 1.96 * se, mu + 1.96 * se)
                lines(ctx, x, ser, f"band_{nm_}_paired_diff{sfx}", yl,
                      f"{ttl} · paired difference vs {a.ref} (95% CI)", hline=0.0)

    ctx.write_scales()
    np.savez_compressed(out / "daily.npz", labels=np.array(labels),
                        domains=np.array([d[1] for d in doms]), corr=d_corr, var_ratio=d_vr,
                        mse=d_mse, mae=d_mae, truth_var=tvar)
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for nmd, v in res["domains"].items():
        print(f"-- {nmd}")
        print(json.dumps({k: v[k] for k in ("band_mae", "band_corr_daily_mean", "band_var_ratio")},
                         ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
