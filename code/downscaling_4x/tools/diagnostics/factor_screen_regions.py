#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
factor_screen_regions.py — 专家帮上忙的程度随哪些当日因子变化
============================================================================
逐 (天, 区域) 同时算两类量, 再在★每个区域内部跨天★求秩相关:

  目标量   expert    路由专家对 token 表示的贡献(需要带 norm_sum 的路由落盘)
           moe_gain  MoE 相对 dense 的 CRPS 相对差 (crps_dense − crps_moe) / crps_dense

  候选因子 从 15 个 ERA5 动态通道反归一化后现算, 全部取区域内有效格点的面积平均:
           t2m / era5_tmax / era5_tmin / precip / soil_w  直接来自通道
           stability   T850 − T2m           低层稳定度, 逆温与谷地冷池的代理
           wspd850     √(u850²+v850²)       低层风速, 冷池形成与混合强度
           upslope     (u850,v850)·∇z 的单位化投影   迎风抬升(正)/背风下沉(负)
           shear       |V500 − V850|        垂直切变
           thickness   z500 − z850          气团厚度(冷暖平流)
           q850 / q500 水汽; 无云量与辐射通道, 以 q850 与降水代理
           doy_cos     年内相位

★为什么在区域内部跨天算★: 静态因子(地形、高程)在一个区域内跨天不变, 它们解释不了逐日变化。
路由现在跟随的恰恰是静态结构, 所以"还剩多少逐日变化可被解释"只能在区域内部问。
区域之间的静态对照另列一张表。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.factor_screen_regions \\
      --moe TC=runs/exp/<tc eval2020> --dense JDB=runs/exp/<jdb eval2020> \\
      --routing runs/exp/<tc>-eval2020-normt \\
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
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, parse_spec, slug

FACTORS = ["t2m", "era5_tmax", "era5_tmin", "precip", "soil_w", "stability",
           "wspd850", "upslope", "shear", "thickness", "q850", "q500", "doy_cos"]


def era5_phys(dd, stats, y, day, var):
    """某天某 ERA5 变量的物理值(上采样到目标网格); 反掉 z-score 与降水正变换。"""
    i = dd.in_vars.index(var)
    x = dd._era5_norm_day(y, day, var) * stats.e_std[i] + stats.e_mean[i]
    if var == C.PRECIP and stats.precip_log:
        x = C.precip_inv(x, stats.precip_scale)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moe", required=True, help="label=eval_dir, 稀疏模型的落场")
    ap.add_argument("--dense", required=True, help="label=eval_dir, dense 对照的落场")
    ap.add_argument("--mu", default=None, help="阶段A μ 的逐日场目录; 给了就另算信息上限")
    ap.add_argument("--routing", default=None, help="带 norm_sum 的路由落盘目录(与 --moe 同一模型)")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ml, md = parse_spec(a.moe)
    dl, dd_ = parse_spec(a.dense)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y = a.year
    ctx = RenderContext(md, a.regions, C.TARGETS[0], out, years=(y,), model_tag=f"{slug(ml)}-vs-{slug(dl)}")
    land = ctx.land
    nreg = len(ctx.region_names)
    rid = ctx.region_id[land]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]
    npx = np.bincount(rid, minlength=nreg + 1)[1:].astype(np.float64)
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)

    stats = Stats(a.era5_dir, a.daymet_dir)
    dd = DownscaleData(a.era5_dir, a.daymet_dir, [y], stats, mode=C.DEFAULT_MODE, era5_cache_years=1)

    # 区域主坡向: 高程梯度的区域平均, 单位化; 行索引朝北, 与 u(东)/v(北) 同坐标系
    elev = np.where(land, ctx.elevation(), np.nan)
    gy, gx = np.gradient(np.nan_to_num(elev, nan=0.0))
    sx = np.bincount(rid, weights=gx[land], minlength=nreg + 1)[1:] / npx
    sy = np.bincount(rid, weights=gy[land], minlength=nreg + 1)[1:] / npx
    snorm = np.hypot(sx, sy); snorm[snorm == 0] = 1.0
    sx, sy = sx / snorm, sy / snorm

    F = {k: np.zeros((nd, nreg)) for k in FACTORS}
    T = {"expert": np.full((nd, nreg), np.nan), "moe_gain": np.zeros((nd, nreg))}
    RAW = {"crps_moe": np.zeros((nd, nreg)), "crps_dense": np.zeros((nd, nreg))}
    if a.mu:
        RAW["crps_mu"] = np.zeros((nd, nreg))
    rmean = lambda v: np.bincount(rid, weights=v, minlength=nreg + 1)[1:] / npx

    rinfo = RD.load_routing_meta(a.routing) if a.routing else None
    if rinfo:
        gh, gw = rinfo["token_grid"]
        wmap = TokenRegionWeights(land, rid, gh, gw, rinfo["patch"], nreg)
        Lr = rinfo["n_moe_layers"]

    t0 = time.time()
    for t in range(nd):
        g = {v: era5_phys(dd, stats, y, t, v)[land] for v in
             ("2m_temperature", "2m_temperature_max", "2m_temperature_min",
              C.PRECIP, "volumetric_soil_water_layer_1", "geopotential_500", "geopotential_850",
              "specific_humidity_500", "specific_humidity_850", "temperature_850",
              "u_component_of_wind_500", "u_component_of_wind_850",
              "v_component_of_wind_500", "v_component_of_wind_850")}
        u8, v8 = g["u_component_of_wind_850"], g["v_component_of_wind_850"]
        u5, v5 = g["u_component_of_wind_500"], g["v_component_of_wind_500"]
        px = {"t2m": g["2m_temperature"], "era5_tmax": g["2m_temperature_max"],
              "era5_tmin": g["2m_temperature_min"], "precip": g[C.PRECIP],
              "soil_w": g["volumetric_soil_water_layer_1"],
              "stability": g["temperature_850"] - g["2m_temperature"],
              "wspd850": np.hypot(u8, v8), "shear": np.hypot(u5 - u8, v5 - v8),
              "thickness": g["geopotential_500"] - g["geopotential_850"],
              "q850": g["specific_humidity_850"], "q500": g["specific_humidity_500"]}
        for k, v in px.items():
            F[k][t] = rmean(v)
        # 迎风分量: 区域平均风在该区主坡向上的投影
        ur, vr = rmean(u8), rmean(v8)
        F["upslope"][t] = ur * sx + vr * sy
        F["doy_cos"][t] = C.doy_sincos(t)[1]

        cm = np.load(md / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
        cd = np.load(dd_ / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
        sm, sd = rmean(cm), rmean(cd)
        RAW["crps_moe"][t], RAW["crps_dense"][t] = sm, sd
        if a.mu:
            RAW["crps_mu"][t] = rmean(np.load(Path(a.mu) / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64))
        T["moe_gain"][t] = (sd - sm) / np.maximum(sd, 1e-12)

        if rinfo:
            p = RD.routing_path(a.routing, y, t, 0)
            if p.exists():
                r = RD.load_routing(a.routing, y, t, 0)
                w = wmap(r["offset"])
                nrm = r["norm_sum"].sum((0, 2)) / (float(r["nfwd"]) * Lr)
                ws = w.sum(0)
                T["expert"][t] = np.where(ws > 0, (w.T @ nrm) / np.maximum(ws, 1), np.nan)
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    res = {"year": y, "n_days": nd, "moe": str(md), "dense": str(dd_), "routing": str(a.routing),
           "note_within": "在每个区域内部跨天算秩相关; 静态因子在区域内不随天变, 因此不参与",
           "note_missing": "15 个 ERA5 通道里没有云量与辐射, 以 q850 与降水代理",
           "factors": FACTORS, "regions": {"display_ids": rlab, "names": rname},
           "within_region_spearman": {}}
    mats = {}
    for tn, tv in T.items():
        if not np.isfinite(tv).any():
            continue
        Mt = np.full((len(FACTORS), nreg), np.nan)
        for i, f in enumerate(FACTORS):
            for j, ri in enumerate(order):
                x, yv = F[f][:, ri], tv[:, ri]
                ok = np.isfinite(x) & np.isfinite(yv)
                if ok.sum() >= 30 and np.ptp(x[ok]) > 0 and np.ptp(yv[ok]) > 0:
                    Mt[i, j] = spearmanr(x[ok], yv[ok]).correlation
        mats[tn] = Mt
        res["within_region_spearman"][tn] = {
            f: {"by_region": [None if not np.isfinite(v) else round(float(v), 3) for v in Mt[i]],
                "mean_abs": round(float(np.nanmean(np.abs(Mt[i]))), 3),
                "mean_signed": round(float(np.nanmean(Mt[i])), 3)} for i, f in enumerate(FACTORS)}
        ctx.heatmap(Mt, FACTORS, rlab, f"factor_vs_{tn}_within_region",
                    cbar=f"Spearman ρ (factor, {tn}) within region across days",
                    scale_group="factor_rho", cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                    title=f"which daily factors move {tn} · within region, across days")
        o = np.argsort(-np.nanmean(np.abs(Mt), 1))
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        ax.bar(range(len(FACTORS)), np.nanmean(np.abs(Mt), 1)[o], color="#4878a8")
        ax.set_xticks(range(len(FACTORS)))
        ax.set_xticklabels([FACTORS[i] for i in o], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean |Spearman ρ| across regions")
        ax.set_title(f"daily factors ranked by how much they move {tn}", fontsize=10)
        ctx.savefig(fig, f"factor_rank_{tn}")

    # ---- 分档池化: 逐 (区域, 因子) 把 365 天按因子分成 5 档, 档内把 CRPS 池化后再比 ----
    # 逐日相关的功效被采样噪声压住; 档内池化几十天能把噪声平掉, 留下真的依赖关系。
    # 区域内每天的有效格点数相同, 所以档内池化等于对该档各天的区域均值取平均。
    NQ = 5
    binned = np.full((len(FACTORS), nreg, NQ), np.nan)
    for i, f in enumerate(FACTORS):
        for j, ri in enumerate(order):
            x = F[f][:, ri]
            if np.ptp(x) == 0:
                continue
            q = np.quantile(x, np.linspace(0, 1, NQ + 1)); q[-1] += 1e-9
            idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, NQ - 1)
            for b in range(NQ):
                s_ = idx == b
                if s_.sum() < 10:
                    continue
                mo, de = RAW["crps_moe"][s_, ri].mean(), RAW["crps_dense"][s_, ri].mean()
                binned[i, j, b] = (de - mo) / max(de, 1e-12) * 100
    spread = np.nanmax(binned, 2) - np.nanmin(binned, 2)          # (因子, 区域) 档间跨度(百分点)
    res["binned_moe_gain_percent"] = {
        f: {"by_region_spread": [None if not np.isfinite(v) else round(float(v), 3) for v in spread[i]],
            "mean_spread": round(float(np.nanmean(spread[i])), 3),
            "domain_quintile_gain": [round(float(np.nanmean(binned[i, :, b])), 3) for b in range(NQ)]}
        for i, f in enumerate(FACTORS)}
    o = np.argsort(-np.nanmean(spread, 1))
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.bar(range(len(FACTORS)), np.nanmean(spread, 1)[o], color="#a85f4a")
    ax.set_xticks(range(len(FACTORS)))
    ax.set_xticklabels([FACTORS[i] for i in o], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MoE gain spread across factor quintiles [pp]")
    ax.set_title("how much does the MoE-over-dense gain change across a factor's quintiles", fontsize=10)
    ctx.savefig(fig, "factor_rank_binned_gain")
    ctx.heatmap(spread, FACTORS, rlab, "factor_binned_gain_spread",
                cbar="gain spread across quintiles [pp]", scale_group="spread_rm",
                cmap="magma", title="MoE gain spread across factor quintiles · factor × region")

    # ---- 信息上限: 逐区域, 这一天的难度与"值不值得多算"能被输入通道解释多少 ----
    # 全部转成秩再做最小二乘, 报调整 R²; 低 R² 表示输入里根本没有这一天的信息,
    # 那么任何按输入学出来的路由信号在该区域都拿不到东西。
    def ranks(v):
        o = np.argsort(np.argsort(v)).astype(np.float64)
        return (o - o.mean()) / max(o.std(), 1e-12)
    ceiling = {}
    if a.mu:
        mu_err = RAW["crps_mu"]
        tgt = {"mu_err": mu_err,
               "stageb_gain": (mu_err - RAW["crps_moe"]) / np.maximum(mu_err, 1e-12)}
        for tn, tv in tgt.items():
            r2 = np.full(nreg, np.nan); best = np.full(nreg, np.nan)
            bestf = [None] * nreg
            for j, ri in enumerate(order):
                Xc = np.stack([ranks(F[f][:, ri]) for f in FACTORS
                               if np.ptp(F[f][:, ri]) > 0], 1)
                yv = ranks(tv[:, ri])
                X = np.concatenate([Xc, np.ones((nd, 1))], 1)
                beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
                ss_res = float(((yv - X @ beta) ** 2).sum()); ss_tot = float((yv ** 2).sum())
                k = Xc.shape[1]
                r2[j] = 1 - (ss_res / max(ss_tot, 1e-12)) * (nd - 1) / max(nd - k - 1, 1)
                rr = [abs(spearmanr(F[f][:, ri], tv[:, ri]).correlation) for f in FACTORS]
                best[j] = float(np.nanmax(rr)); bestf[j] = FACTORS[int(np.nanargmax(rr))]
            ceiling[tn] = {"adj_r2_by_region": [round(float(v), 3) for v in r2],
                           "best_single_abs_rho": [round(float(v), 3) for v in best],
                           "best_single_factor": bestf,
                           "mean_adj_r2": round(float(np.nanmean(r2)), 3)}
            fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
            ax.bar(np.arange(nreg) - 0.2, r2, 0.4, label="adjusted R² (all factors)")
            ax.bar(np.arange(nreg) + 0.2, best, 0.4, label="best single |ρ|")
            ax.set_xticks(np.arange(nreg)); ax.set_xticklabels(rlab, fontsize=8)
            ax.set_ylabel("explained"); ax.legend(fontsize=8)
            ax.set_title(f"how much of daily {tn} the input channels can explain · by region", fontsize=10)
            ctx.savefig(fig, f"information_ceiling_{tn}")
        res["information_ceiling"] = ceiling

    ctx.write_scales()
    np.savez_compressed(out / "factors.npz", binned=binned, spread=spread,
                        **{f"R__{k}": v[:, order] for k, v in RAW.items()}, region_display_ids=np.array(rlab),
                        region_names=np.array(rname), factors=np.array(FACTORS),
                        slope_x=sx[order], slope_y=sy[order],
                        **{f"F__{k}": v[:, order] for k, v in F.items()},
                        **{f"T__{k}": v[:, order] for k, v in T.items()},
                        **{f"rho__{k}": v for k, v in mats.items()})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for tn, c in ceiling.items():
        print(f"\n=== 信息上限 {tn}: 逐区域 调整R²(全部因子) / 最强单因子 ===")
        for j, rd in enumerate(rlab):
            print(f"  {rd:>2} {rname[j]:<13} R²={c['adj_r2_by_region'][j]:6.3f}   "
                  f"最强单因子 {c['best_single_factor'][j]:<11}|ρ|={c['best_single_abs_rho'][j]:.3f}")
        print(f"  域平均 调整R² = {c['mean_adj_r2']}")
    o2 = np.argsort(-np.nanmean(spread, 1))
    print("\n=== moe_gain 分档池化: 档间跨度(百分点), 按平均跨度排名 ===")
    for i in o2:
        q = res["binned_moe_gain_percent"][FACTORS[i]]["domain_quintile_gain"]
        print(f"  {FACTORS[i]:<12}跨度 {np.nanmean(spread[i]):5.2f}pp   五档增益 {q}")
    for tn, Mt in mats.items():
        o = np.argsort(-np.nanmean(np.abs(Mt), 1))
        print(f"\n=== {tn}: 区域内跨天的平均 |ρ| 排名 ===")
        for i in o:
            print(f"  {FACTORS[i]:<12}{np.nanmean(np.abs(Mt[i])):6.3f}   (带号均值 {np.nanmean(Mt[i]):+.3f})")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
