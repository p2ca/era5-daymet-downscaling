#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_a_compare.py — 两个确定性落场(阶段 A 回归器)的配对诊断
============================================================================
把方法 A 与方法 B 在同一年份的逐日误差放到同一把尺子上拆开看, 回答"MAE 与 SSIM 排名相反"
这类现象来自误差的哪一层:

  daily      逐日 / 逐月 / 分区 MAE 与 bias 的配对差
  maps       年均 MAE 与 bias 地图(两方法共色标)、差值图、逐像素 "A 赢的天数占比"
  debias     去掉一层偏差后还剩多少误差: 全域常数偏差 / 逐像素年偏差 / 逐像素逐月偏差
  ssim       SSIM 的三分量(亮度 l / 对比 c / 结构 s)逐日均值与年均分量图
  terrain    按地形粗糙度(P×P 块内高程标准差)与高程各分五档的 MAE
  psd        径向功率谱比 pred/truth(全陆地方框, 全年平均)
  cases      A 领先最多与 B 领先最多的两天: 真值 / 预测 / 误差图

口径: 有效域 = 落场的非 NaN 区 = 分区文件的 land; 所有均值只在有效域内; SSIM 的窗口、
data_range 与掩膜腐蚀与 evaluation.metrics.ssim_masked 完全一致, 并用它的标量做对拍自检。
逐日 MAE 取平均与全池等价, RMSE 一律按全池算。全部单图, 同组共色标(scales.json), 纯 CPU。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.stage_a_compare \\
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation.metrics import ssim_components
from downscaling_4x.evaluation.render.context import RenderContext

# ---------------------------------------------------------------- 小工具
def parse_spec(s):
    """'label=dir' -> (label, Path)。"""
    if "=" not in s:
        raise SystemExit(f"--a/--b 需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def pool_sum(x, P):
    H, W = x.shape
    return x.reshape(H // P, P, W // P, P).sum((1, 3))


def unpool(tok, P):
    return np.repeat(np.repeat(tok, P, 0), P, 1)


def to_grid(vec, land, fill=np.nan):
    g = np.full(land.shape, fill, np.float64)
    g[land] = vec
    return g


# ---------------------------------------------------------------- 绘图助手(单图)
def lines(ctx, series, name, ylabel, xlabel="day of year", title=None, zero=False, logx=False):
    fig, ax = plt.subplots(figsize=(9.0, 3.6), constrained_layout=True)
    for lab, (x, y) in series.items():
        ax.plot(x, y, lw=1.1, label=lab)
    if zero:
        ax.axhline(0, color="k", lw=0.8, ls="--")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def grouped_bars(ctx, groups, labels, name, ylabel, title=None, zero=False, rotate=0):
    """groups: {系列名: 值列表}, labels: 类目; 同一类目的几个系列并排。"""
    k = len(groups)
    x = np.arange(len(labels))
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) + 2.0), 3.8), constrained_layout=True)
    for i, (lab, vals) in enumerate(groups.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=lab)
    if zero:
        ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="label=run_dir, 落场取 <run_dir>/fields<year>/<target>")
    ap.add_argument("--b", required=True, help="label=run_dir")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True, help="regions_v1.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=16, help="地形粗糙度的块边长(px)")
    ap.add_argument("--box", type=int, default=256, help="功率谱用的全陆地方框边长(px)")
    ap.add_argument("--limit-days", type=int, default=0, help="冒烟用: 只跑前 N 天")
    a = ap.parse_args()

    la, da = parse_spec(a.a)
    lb, db = parse_spec(a.b)
    if a.target == C.PRECIP:
        raise SystemExit("本诊断只定义在温度目标上: 降水的 SSIM 分量与去偏在物理空间没有同样的解释")
    y = a.year
    fa = da / f"fields{y}" / a.target
    fb = db / f"fields{y}" / a.target
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{slug(la)}-vs-{slug(lb)}"
    ctx = RenderContext(fa, a.regions, a.target, out, years=(y,), model_tag=tag)
    land = ctx.land
    land_er = MT.eroded_land_mask(land)
    H, W = land.shape
    nland = int(land.sum())
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    month = ctx.month_of_day[y][:nd]
    unit = ctx.unit

    # 与两个落场自己的标量对拍: 逐日 SSIM 均值须与 metrics.json 一致(只在整年时检查)
    def own_ssim(fd):
        p = fd / "metrics.json"
        if not p.exists():
            return None
        m = json.load(open(p))
        return float((m.get(unit) or m.get("K") or {}).get("ssim", np.nan))

    E = {la: np.zeros((nd, nland), np.float32), lb: np.zeros((nd, nland), np.float32)}
    comp_sum = {lab: {k: np.zeros((H, W)) for k in ("l", "c", "s", "ssim")} for lab in (la, lb)}
    comp_daily = {lab: {k: np.zeros(nd) for k in ("l", "c", "s", "ssim")} for lab in (la, lb)}
    box = MT.pick_land_box(land, a.box)
    y0, x0, sz = box
    psd_sum = {k: None for k in ("truth", la, lb)}
    t_start = time.time()
    for t in range(nd):
        tr = ctx.truth(y, t)
        if t == 0 and not np.array_equal(np.isfinite(tr), land):
            raise SystemExit("真值的有效域与分区文件的 land 不一致")
        preds = {}
        for lab, fd in ((la, fa), (lb, fb)):
            p = np.load(fd / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
            if not np.isfinite(p[land]).all():
                raise SystemExit(f"{lab} 第 {t} 天有效域内含非有限值")
            preds[lab] = p
            E[lab][t] = (p - tr)[land].astype(np.float32)
            l, c, s = ssim_components(p, tr, land)
            for k, f in (("l", l), ("c", c), ("s", s), ("ssim", l * c * s)):
                comp_sum[lab][k] += f
                comp_daily[lab][k][t] = float(f[land_er].mean())
        for k, f in (("truth", tr), (la, preds[la]), (lb, preds[lb])):
            r = MT.radial_psd(np.nan_to_num(f[y0:y0 + sz, x0:x0 + sz], nan=0.0), sz)
            psd_sum[k] = r if psd_sum[k] is None else psd_sum[k] + r
        if t % 30 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t_start:.0f}s", flush=True)

    res = {"a": {"label": la, "run": str(da)}, "b": {"label": lb, "run": str(db)},
           "target": a.target, "year": y, "n_days": nd, "n_land": nland, "unit": unit,
           "patch": a.patch, "psd_box": [int(y0), int(x0), int(sz)]}

    # ---- 逐日 / 逐月 / 分区 ----
    absE = {lab: np.abs(E[lab]) for lab in (la, lb)}
    daily = {lab: {"mae": absE[lab].mean(1), "bias": E[lab].mean(1),
                   "rmse": np.sqrt((E[lab].astype(np.float64) ** 2).mean(1))} for lab in (la, lb)}
    res["overall"] = {lab: {"mae": float(absE[lab].mean()), "bias": float(E[lab].mean()),
                            "rmse": float(np.sqrt((E[lab].astype(np.float64) ** 2).mean())),
                            "ssim": float(comp_daily[lab]["ssim"].mean()),
                            "ssim_l": float(comp_daily[lab]["l"].mean()),
                            "ssim_c": float(comp_daily[lab]["c"].mean()),
                            "ssim_s": float(comp_daily[lab]["s"].mean())} for lab in (la, lb)}
    for lab, fd in ((la, fa), (lb, fb)):
        ref = own_ssim(fd)
        if ref is not None and nd == C.DAYS_PER_YEAR:
            got = res["overall"][lab]["ssim"]
            res["overall"][lab]["ssim_metrics_json"] = ref
            if abs(got - ref) > 2e-3:
                raise SystemExit(f"{lab} 的 SSIM 分量乘积 {got:.4f} 与落场 metrics.json 的 {ref:.4f} 不符")
    dd = daily[lb]["mae"] - daily[la]["mae"]                       # >0: A 更好
    res["daily"] = {"mae_a": daily[la]["mae"].round(4).tolist(), "mae_b": daily[lb]["mae"].round(4).tolist(),
                    "bias_a": daily[la]["bias"].round(4).tolist(), "bias_b": daily[lb]["bias"].round(4).tolist(),
                    "share_days_a_better": float((dd > 0).mean()),
                    "a_best_day": int(np.argmax(dd)), "b_best_day": int(np.argmin(dd))}
    months = sorted(set(month.tolist()))
    res["monthly"] = {"months": months,
                      "mae_a": [float(daily[la]["mae"][month == m].mean()) for m in months],
                      "mae_b": [float(daily[lb]["mae"][month == m].mean()) for m in months],
                      "bias_a": [float(daily[la]["bias"][month == m].mean()) for m in months],
                      "bias_b": [float(daily[lb]["bias"][month == m].mean()) for m in months]}
    rid = ctx.region_id[land]
    reg = {"names": ctx.region_names, "display_ids": ctx.region_ids, "mae_a": [], "mae_b": [],
           "bias_a": [], "bias_b": [], "n_cells": []}
    for i, _ in enumerate(ctx.region_names):
        m = rid == i + 1
        reg["n_cells"].append(int(m.sum()))
        for lab, key in ((la, "a"), (lb, "b")):
            reg[f"mae_{key}"].append(float(absE[lab][:, m].mean()) if m.any() else float("nan"))
            reg[f"bias_{key}"].append(float(E[lab][:, m].mean()) if m.any() else float("nan"))
    res["regions"] = reg

    # ---- 去偏 ----
    def _stats(e):
        e = e.astype(np.float64)
        return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
    deb = {}
    for lab in (la, lb):
        e = E[lab].astype(np.float64)
        px = e - e.mean(0, keepdims=True)
        pm = e.copy()
        for m in months:
            sel = month == m
            pm[sel] -= e[sel].mean(0, keepdims=True)
        deb[lab] = {"raw": _stats(e), "minus_global_bias": _stats(e - e.mean()),
                    "minus_pixel_annual_bias": _stats(px), "minus_pixel_monthly_bias": _stats(pm)}
    res["debias"] = deb

    # ---- 地形分层 ----
    z = np.where(land, ctx.elevation(), 0.0)
    P = a.patch
    lf = land.astype(np.float64)
    n_k = pool_sum(lf, P)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_k = pool_sum(z * z, P) / np.maximum(n_k, 1) - (pool_sum(z, P) / np.maximum(n_k, 1)) ** 2
    rough = unpool(np.sqrt(np.maximum(var_k, 0.0)), P)[land]
    elev = z[land]
    # 每档: 误差在全部陆地像素上取均值; SSIM 分量用年均分量图, 只在腐蚀掩膜内的像素上取均值
    strata = {}
    er_v = land_er[land]
    for key, v in (("roughness_std_m", rough), ("elevation_m", elev)):
        qs = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
        b = np.digitize(v, qs)
        rows = []
        for i in range(5):
            m = b == i
            me = m & er_v
            row = {"bin": i, "v_lo": float(v[m].min()), "v_hi": float(v[m].max()),
                   "share_cells": float(m.mean())}
            for lab, k2 in ((la, "a"), (lb, "b")):
                row[f"mae_{k2}"] = float(absE[lab][:, m].mean())
                row[f"bias_{k2}"] = float(E[lab][:, m].mean())
                for comp in ("c", "s", "ssim"):
                    row[f"ssim_{comp}_{k2}"] = float((comp_sum[lab][comp][land] / nd)[me].mean())
            rows.append(row)
        strata[key] = rows
    res["strata"] = strata

    # ---- 功率谱比 ----
    kk = np.arange(len(psd_sum["truth"]))
    valid = (kk >= 1) & (kk < sz // 2)
    res["psd_ratio"] = {"k": kk[valid].tolist(),
                        "a_over_truth": (psd_sum[la] / psd_sum["truth"])[valid].round(4).tolist(),
                        "b_over_truth": (psd_sum[lb] / psd_sum["truth"])[valid].round(4).tolist()}
    hi = valid & (kk >= sz // 4)
    res["psd_ratio"]["hi_band_mean"] = {la: float((psd_sum[la] / psd_sum["truth"])[hi].mean()),
                                        lb: float((psd_sum[lb] / psd_sum["truth"])[hi].mean())}

    # ---- 图: 逐日 / 逐月 / 分区 ----
    doy = np.arange(1, nd + 1)
    lines(ctx, {la: (doy, daily[la]["mae"]), lb: (doy, daily[lb]["mae"])}, "daily_mae", f"MAE [{unit}]")
    lines(ctx, {f"{lb} − {la}": (doy, dd)}, "daily_mae_diff", f"ΔMAE [{unit}] (>0: {la} better)", zero=True)
    lines(ctx, {la: (doy, daily[la]["bias"]), lb: (doy, daily[lb]["bias"])}, "daily_bias", f"bias [{unit}]", zero=True)
    mlab = [ctx.MONTHS[m - 1] for m in months]
    grouped_bars(ctx, {la: res["monthly"]["mae_a"], lb: res["monthly"]["mae_b"]}, mlab, "monthly_mae", f"MAE [{unit}]")
    grouped_bars(ctx, {la: res["monthly"]["bias_a"], lb: res["monthly"]["bias_b"]}, mlab, "monthly_bias", f"bias [{unit}]", zero=True)
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    grouped_bars(ctx, {la: [reg["mae_a"][i] for i in order], lb: [reg["mae_b"][i] for i in order]},
                 rlab, "region_mae", f"MAE [{unit}]", title="regions by display id")
    ctx.bar([reg["mae_b"][i] - reg["mae_a"][i] for i in order], rlab, "region_mae_diff",
            ylabel=f"ΔMAE [{unit}] ({lb} − {la}; >0: {la} better)", ref=0.0)
    grouped_bars(ctx, {la: [reg["bias_a"][i] for i in order], lb: [reg["bias_b"][i] for i in order]},
                 rlab, "region_bias", f"bias [{unit}]", zero=True)

    # ---- 图: 地图 ----
    mae_map = {lab: to_grid(absE[lab].mean(0), land) for lab in (la, lb)}
    bias_map = {lab: to_grid(E[lab].mean(0), land) for lab in (la, lb)}
    both = np.concatenate([mae_map[la][land], mae_map[lb][land]])
    vmax = float(np.nanpercentile(both, 99))
    for lab in (la, lb):
        ctx.map(mae_map[lab], f"annual_mae_{slug(lab)}", cmap="magma_r", vmin=0, vmax=vmax,
                scale_group="annual_mae", cbar=f"MAE [{unit}]", title=f"annual MAE · {lab}")
    ctx.map(mae_map[lb] - mae_map[la], "annual_mae_diff", cmap="RdBu", scale_group="annual_mae_diff",
            cbar=f"ΔMAE [{unit}] ({lb} − {la}; blue: {la} better)", diverging=True)
    bvmax = float(np.nanpercentile(np.abs(np.concatenate([bias_map[la][land], bias_map[lb][land]])), 99))
    for lab in (la, lb):
        ctx.map(bias_map[lab], f"annual_bias_{slug(lab)}", cmap="RdBu_r", vmin=-bvmax, vmax=bvmax,
                scale_group="annual_bias", cbar=f"bias [{unit}]", title=f"annual bias · {lab}")
    win = to_grid((absE[la] < absE[lb]).mean(0), land)
    ctx.map(win, "win_share", cmap="RdBu", vmin=0.0, vmax=1.0, scale_group="win_share",
            cbar=f"share of days {la} closer to truth", title=f"{la} wins on this share of days")
    res["maps"] = {"win_share_mean": float(np.nanmean(win)),
                   "share_cells_a_better_annual_mae": float((mae_map[la][land] < mae_map[lb][land]).mean())}

    # ---- 图: SSIM 分量 ----
    grouped_bars(ctx, {la: [res["overall"][la][k] for k in ("ssim_l", "ssim_c", "ssim_s", "ssim")],
                       lb: [res["overall"][lb][k] for k in ("ssim_l", "ssim_c", "ssim_s", "ssim")]},
                 ["luminance l", "contrast c", "structure s", "SSIM"], "ssim_components", "mean over land")
    for comp, cmap in (("c", "viridis"), ("s", "viridis")):
        maps = {lab: np.where(land_er, comp_sum[lab][comp] / nd, np.nan) for lab in (la, lb)}
        lo = float(np.nanpercentile(np.concatenate([maps[la][land_er], maps[lb][land_er]]), 1))
        for lab in (la, lb):
            ctx.map(maps[lab], f"ssim_{comp}_{slug(lab)}", cmap=cmap, vmin=lo, vmax=1.0,
                    scale_group=f"ssim_{comp}", cbar=f"SSIM {comp} term (annual mean)",
                    title=f"SSIM {comp} · {lab}")
        ctx.map(maps[la] - maps[lb], f"ssim_{comp}_diff", cmap="RdBu", scale_group=f"ssim_{comp}_diff",
                cbar=f"Δ{comp} ({la} − {lb})", diverging=True)
    lines(ctx, {f"{la} c": (doy, comp_daily[la]["c"]), f"{lb} c": (doy, comp_daily[lb]["c"]),
                f"{la} s": (doy, comp_daily[la]["s"]), f"{lb} s": (doy, comp_daily[lb]["s"])},
          "daily_ssim_components", "term mean over land")

    # ---- 图: 去偏 / 地形 / 谱 ----
    keys = ["raw", "minus_global_bias", "minus_pixel_annual_bias", "minus_pixel_monthly_bias"]
    grouped_bars(ctx, {la: [deb[la][k]["mae"] for k in keys], lb: [deb[lb][k]["mae"] for k in keys]},
                 ["raw", "− global bias", "− pixel annual bias", "− pixel monthly bias"], "debias_mae", f"MAE [{unit}]")
    grouped_bars(ctx, {la: [deb[la][k]["rmse"] for k in keys], lb: [deb[lb][k]["rmse"] for k in keys]},
                 ["raw", "− global bias", "− pixel annual bias", "− pixel monthly bias"], "debias_rmse", f"RMSE [{unit}]")
    for key, short in (("roughness_std_m", "roughness"), ("elevation_m", "elevation")):
        rows = strata[key]
        lab5 = [f"Q{i + 1}\n{r['v_lo']:.0f}–{r['v_hi']:.0f} m" for i, r in enumerate(rows)]
        grouped_bars(ctx, {la: [r["mae_a"] for r in rows], lb: [r["mae_b"] for r in rows]}, lab5,
                     f"strata_mae_{short}", f"MAE [{unit}]", title=f"MAE by {short} quintile (P={P})")
        grouped_bars(ctx, {la: [r["ssim_s_a"] for r in rows], lb: [r["ssim_s_b"] for r in rows]}, lab5,
                     f"strata_ssim_s_{short}", "SSIM structure term", title=f"structure term by {short} quintile")
        grouped_bars(ctx, {la: [r["ssim_c_a"] for r in rows], lb: [r["ssim_c_b"] for r in rows]}, lab5,
                     f"strata_ssim_c_{short}", "SSIM contrast term", title=f"contrast term by {short} quintile")
    pr = res["psd_ratio"]
    lines(ctx, {la: (pr["k"], pr["a_over_truth"]), lb: (pr["k"], pr["b_over_truth"])}, "psd_ratio",
          "PSD pred / truth", xlabel=f"radial wavenumber (box {sz}px)", logx=True, title="annual-mean power ratio")

    # ---- 图: 案例日 ----
    cases = {"a_best": res["daily"]["a_best_day"], "b_best": res["daily"]["b_best_day"]}
    res["cases"] = {}
    for tagc, t in cases.items():
        tr = ctx.truth(y, t)
        pa = np.load(fa / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
        pb = np.load(fb / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
        fields = {"truth": tr, la: np.where(land, pa, np.nan), lb: np.where(land, pb, np.nan)}
        allv = np.concatenate([f[land] for f in fields.values()])
        lo, hi = float(np.nanpercentile(allv, 1)), float(np.nanpercentile(allv, 99))
        for k, f in fields.items():
            ctx.map(f, f"case_{tagc}_d{t}_{slug(k)}", cmap="turbo", vmin=lo, vmax=hi,
                    scale_group=f"case_{tagc}_field", cbar=f"[{unit}]", title=f"day {t} · {k}")
        errs = {la: np.where(land, pa - tr, np.nan), lb: np.where(land, pb - tr, np.nan)}
        ev = float(np.nanpercentile(np.abs(np.concatenate([e[land] for e in errs.values()])), 99))
        for k, e in errs.items():
            ctx.map(e, f"case_{tagc}_d{t}_err_{slug(k)}", cmap="RdBu_r", vmin=-ev, vmax=ev,
                    scale_group=f"case_{tagc}_err", cbar=f"pred − truth [{unit}]", title=f"day {t} · error · {k}")
        res["cases"][tagc] = {"day": t, "mae_a": float(daily[la]["mae"][t]), "mae_b": float(daily[lb]["mae"][t]),
                              "bias_a": float(daily[la]["bias"][t]), "bias_b": float(daily[lb]["bias"][t])}

    ctx.write_scales()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    o = res["overall"]
    print(json.dumps({lab: {k: round(v, 4) for k, v in o[lab].items()} for lab in (la, lb)}, ensure_ascii=False, indent=1))
    print("debias:", json.dumps({lab: {k: round(v["mae"], 4) for k, v in deb[lab].items()} for lab in (la, lb)}, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
