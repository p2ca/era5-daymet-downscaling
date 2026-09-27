#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
correction_drivers.py — 降尺度该做的修正由什么驱动, 模型读到了吗
============================================================================
不经过模型误差, 直接看数据: 逐 (天, 区域) 把"Daymet 相对 ERA5 该做的修正"拆成两件

  d_mean   区域平均的差      mean(Daymet) − mean(ERA5↑)   今天这个区整体该抬高/压低多少
  d_std    区域内空间起伏的差 std(Daymet) − std(ERA5↑)     今天该造出多少细尺度结构

再问三件事:

  1. 这两个修正量的逐日变化, 能被哪些输入通道解释? (秩相关 + 全因子调整 R²)
  2. 模型自己产生的修正(预测 − ERA5↑)与真值的修正相关多少? —— 它复现了多少
  3. 对每个通道, 真值修正与它的相关 ρ_truth, 模型修正与它的相关 ρ_pred, 两者的差
     ρ_truth − ρ_pred 就是"数据里这个通道在驱动修正, 而模型没跟着动"的量

★为什么看区域内跨天★: 静态地形在区域内不随天变, 解释不了逐日的修正; 而逐日的修正正是
降尺度在做的事。区域之间的静态对照另有工具。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.correction_drivers \\
      --model MU=runs/exp/<jda1>/fields2020/<target> --model TC=runs/exp/<tc eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.diagnostics.factor_screen_regions import FACTORS, era5_phys
from downscaling_4x.tools.plotting.plot_expert_region_month import parse_spec, slug


def ranks(v):
    o = np.argsort(np.argsort(v)).astype(np.float64)
    return (o - o.mean()) / max(o.std(), 1e-12)


def adj_r2(X, y):
    A = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(((y - A @ beta) ** 2).sum()); ss_tot = float((y ** 2).sum())
    n, k = X.shape[0], X.shape[1]
    return 1 - (ss_res / max(ss_tot, 1e-12)) * (n - 1) / max(n - k - 1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=场目录, 可多次")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y = a.year
    ctx = RenderContext(specs[0][1], a.regions, a.target, out, years=(y,), model_tag="corrdrv")
    land = ctx.land
    nreg = len(ctx.region_names)
    rid = ctx.region_id[land]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]
    npx = np.bincount(rid, minlength=nreg + 1)[1:].astype(np.float64)
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    era5_var = {C.TARGETS[0]: "2m_temperature_max", C.TARGETS[1]: "2m_temperature_min",
                C.TARGETS[2]: C.PRECIP}[a.target]

    stats = Stats(a.era5_dir, a.daymet_dir)
    dd = DownscaleData(a.era5_dir, a.daymet_dir, [y], stats, mode=C.DEFAULT_MODE, era5_cache_years=1)
    elev = np.where(land, ctx.elevation(), np.nan)
    gy, gx = np.gradient(np.nan_to_num(elev, nan=0.0))
    rmean = lambda v: np.bincount(rid, weights=v, minlength=nreg + 1)[1:] / npx
    sx = rmean(gx[land]); sy = rmean(gy[land])
    sn = np.hypot(sx, sy); sn[sn == 0] = 1.0; sx, sy = sx / sn, sy / sn

    def rstd(v):
        m = rmean(v)
        return np.sqrt(np.maximum(rmean(v * v) - m * m, 0.0))

    F = {k: np.zeros((nd, nreg)) for k in FACTORS}
    D = {"truth": {"mean": np.zeros((nd, nreg)), "std": np.zeros((nd, nreg))}}
    for l in labels:
        D[l] = {"mean": np.zeros((nd, nreg)), "std": np.zeros((nd, nreg))}
    pname = {l: ("ens_mean" if (d / "ens_mean").is_dir() else "prediction") for l, d in specs}

    t0 = time.time()
    for t in range(nd):
        g = {v: era5_phys(dd, stats, y, t, v)[land] for v in
             ("2m_temperature", "2m_temperature_max", "2m_temperature_min", C.PRECIP,
              "volumetric_soil_water_layer_1", "geopotential_500", "geopotential_850",
              "specific_humidity_500", "specific_humidity_850", "temperature_850",
              "u_component_of_wind_500", "u_component_of_wind_850",
              "v_component_of_wind_500", "v_component_of_wind_850")}
        u8, v8 = g["u_component_of_wind_850"], g["v_component_of_wind_850"]
        u5, v5 = g["u_component_of_wind_500"], g["v_component_of_wind_500"]
        for k, v in {"t2m": g["2m_temperature"], "era5_tmax": g["2m_temperature_max"],
                     "era5_tmin": g["2m_temperature_min"], "precip": g[C.PRECIP],
                     "soil_w": g["volumetric_soil_water_layer_1"],
                     "stability": g["temperature_850"] - g["2m_temperature"],
                     "wspd850": np.hypot(u8, v8), "shear": np.hypot(u5 - u8, v5 - v8),
                     "thickness": g["geopotential_500"] - g["geopotential_850"],
                     "q850": g["specific_humidity_850"], "q500": g["specific_humidity_500"]}.items():
            F[k][t] = rmean(v)
        F["upslope"][t] = rmean(u8) * sx + rmean(v8) * sy
        F["doy_cos"][t] = C.doy_sincos(t)[1]

        e = g[era5_var]                                   # ERA5 上采样到目标网格
        tr = ctx.truth(y, t)[land]
        D["truth"]["mean"][t] = rmean(tr) - rmean(e)
        D["truth"]["std"][t] = rstd(tr) - rstd(e)
        for l, d in specs:
            p = np.load(d / pname[l] / f"{y}_d{t}.npy")[land].astype(np.float64)
            D[l]["mean"][t] = rmean(p) - rmean(e)
            D[l]["std"][t] = rstd(p) - rstd(e)
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    res = {"year": y, "target": a.target, "n_days": nd, "models": {l: str(d) for l, d in specs},
           "note": "修正量 = 该区域内 Daymet 与 ERA5(上采样) 的区域平均之差 / 区域内空间标准差之差; "
                   "全部在区域内跨天算, 静态地形不参与",
           "regions": {"display_ids": rlab, "names": rname}, "which": {}, "reproduce": {}, "ceiling": {}}
    mats = {}
    for which in ("mean", "std"):
        tv = D["truth"][which]
        r2 = np.full(nreg, np.nan)
        for j, ri in enumerate(order):
            X = np.stack([ranks(F[f][:, ri]) for f in FACTORS if np.ptp(F[f][:, ri]) > 0], 1)
            r2[j] = adj_r2(X, ranks(tv[:, ri]))
        res["ceiling"][which] = {"adj_r2_by_region": [round(float(v), 3) for v in r2],
                                 "mean_adj_r2": round(float(np.nanmean(r2)), 3),
                                 "truth_correction_mean": [round(float(tv[:, ri].mean()), 3) for ri in order],
                                 "truth_correction_daily_std": [round(float(tv[:, ri].std()), 3) for ri in order]}
        RT = np.full((len(FACTORS), nreg), np.nan)
        for i, f in enumerate(FACTORS):
            for j, ri in enumerate(order):
                if np.ptp(F[f][:, ri]) > 0 and np.ptp(tv[:, ri]) > 0:
                    RT[i, j] = spearmanr(F[f][:, ri], tv[:, ri]).correlation
        mats[f"truth_{which}"] = RT
        ctx.heatmap(RT, FACTORS, rlab, f"driver_of_truth_correction_{which}",
                    cbar=f"Spearman ρ (factor, truth {which} correction)", scale_group="drv",
                    cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                    title=f"what drives the {which} correction in the DATA · region × factor")
        for l in labels:
            pv = D[l][which]
            RP = np.full((len(FACTORS), nreg), np.nan)
            for i, f in enumerate(FACTORS):
                for j, ri in enumerate(order):
                    if np.ptp(F[f][:, ri]) > 0 and np.ptp(pv[:, ri]) > 0:
                        RP[i, j] = spearmanr(F[f][:, ri], pv[:, ri]).correlation
            mats[f"{l}_{which}"] = RP
            gap = RT - RP
            mats[f"gap_{l}_{which}"] = gap
            ctx.heatmap(gap, FACTORS, rlab, f"unread_driver_{slug(l)}_{which}",
                        cbar="ρ(truth) − ρ(model)", scale_group="gap",
                        cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                        title=f"{l} · driver present in data but not in the model's correction ({which})")
            rep = [float(spearmanr(tv[:, ri], pv[:, ri]).correlation) for ri in order]
            res["reproduce"].setdefault(which, {})[l] = [round(v, 3) for v in rep]
            res["which"].setdefault(which, {})[l] = {
                f: {"rho_truth": round(float(np.nanmean(RT[i])), 3),
                    "rho_model": round(float(np.nanmean(RP[i])), 3),
                    "gap": round(float(np.nanmean(gap[i])), 3)} for i, f in enumerate(FACTORS)}
    for which in ("mean", "std"):
        fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
        w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            ax.bar(np.arange(nreg) + (i - (len(labels) - 1) / 2) * w,
                   res["reproduce"][which][l], w, label=l)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(nreg)); ax.set_xticklabels(rlab, fontsize=8)
        ax.set_ylabel("Spearman ρ (truth correction, model correction)")
        ax.set_title(f"does the model reproduce the daily {which} correction · by region", fontsize=10)
        ax.legend(fontsize=8)
        ctx.savefig(fig, f"reproduce_correction_{which}")
    ctx.write_scales()
    np.savez_compressed(out / "drivers.npz", region_display_ids=np.array(rlab),
                        region_names=np.array(rname), factors=np.array(FACTORS),
                        **{f"F__{k}": v[:, order] for k, v in F.items()},
                        **{f"D__{k}__{w}": D[k][w][:, order] for k in D for w in ("mean", "std")},
                        **{f"rho__{k}": v for k, v in mats.items()})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for which in ("mean", "std"):
        print(f"\n=== {which} 修正: 数据里由什么驱动, 模型跟到多少 (域平均) ===")
        print(f"{'因子':<12}{'ρ(真值修正)':>12}" + "".join(f"{'ρ('+l+')':>11}" for l in labels)
              + "".join(f"{'差'+l:>9}" for l in labels))
        w = res["which"][which]
        idx = np.argsort(-np.abs([w[labels[0]][f]["rho_truth"] for f in FACTORS]))
        for i in idx:
            f = FACTORS[i]
            print(f"{f:<12}{w[labels[0]][f]['rho_truth']:12.3f}"
                  + "".join(f"{w[l][f]['rho_model']:11.3f}" for l in labels)
                  + "".join(f"{w[l][f]['gap']:9.3f}" for l in labels))
        print(f"  输入对真值{which}修正的调整 R² 域平均 = {res['ceiling'][which]['mean_adj_r2']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
