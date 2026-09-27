#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
routing_share_and_stability.py — 路由专家出了多少力, 以及选中集合有多稳
============================================================================
两件事, 共用一趟对路由落盘的扫描:

1. ★路由专家对 token 表示的贡献★。落盘的 norm_sum 是每个路由专家的加权输出范数与该 token
   进 FFN 时的隐状态范数之比(共享专家不在其中, 它不经过路由)。据此分两个量:
     总贡献    每 token 每层的 Σ_e ‖w_e·f_e(x)‖ / ‖x‖ —— 路由这一路整体把表示推动了多少
     单位贡献  Σ norm / Σ k —— 平均每被选中一次贡献多少, 用来区分"没给"与"给了也没推动"
2. ★选中集合的逐日稳定性★。逐像素统计一年里有多少天拿到过路由专家, 以及先验与 k 的方差里
   有多少是跨像素(空间)的、多少是同一像素的逐日变化。先验若几乎全是空间成分, 那么全帧
   top-C 每天挑中的基本是同一批地方。

口径: token 的域内判定与模型一致(块内任一像素在域内即域内); token 量按其覆盖的有效格点摊到
像素再按区域求和, 与 plot_expert_region_month 共用同一份 offset 映射。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.routing_share_and_stability \\
      --model DEC=runs/exp/<dec eval2020> --model TC=... --model EC=... --members 0 \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, parse_spec, slug
from downscaling_4x.tools.diagnostics.routing_capacity_regions import region_roughness, grouped_bars


def scan(label, dump, a, out, scales):
    info = RD.load_routing_meta(dump)
    if info is None:
        raise SystemExit(f"{dump} 下没有 routing/meta.json")
    gh, gw = info["token_grid"]
    patch, L, E = info["patch"], info["n_moe_layers"], info["n_experts"]
    items = [(y, d, m) for (y, d, m) in RD.available_routing(dump)
             if y == a.year and m in a.members and (a.days is None or d in a.days)]
    if not items:
        raise SystemExit(f"{label}: 没有符合条件的路由文件")
    ctx = RenderContext(dump, a.regions, C.TARGETS[0], out, years=(a.year,),
                        model_tag=slug(label), scales_path=scales)
    land = ctx.land
    nreg = len(ctx.region_names)
    wmap = TokenRegionWeights(land, ctx.region_id[land], gh, gw, patch, nreg)
    Y, X = np.nonzero(land)
    npx_all = int(land.sum())
    T = gh * gw

    reg_norm = np.zeros(nreg); reg_k = np.zeros(nreg); reg_w = np.zeros(nreg)
    px_k = np.zeros(npx_all); px_k2 = np.zeros(npx_all); px_sel = np.zeros(npx_all)
    px_norm = np.zeros(npx_all)
    px_pr = np.zeros(npx_all); px_pr2 = np.zeros(npx_all); has_prior = False
    nday = 0
    t0 = time.time()
    for n, (y, d, m) in enumerate(items):
        r = RD.load_routing(dump, y, d, m)
        if "norm_sum" not in r:
            raise SystemExit(f"{label}: 路由落盘里没有 norm_sum")
        w = wmap(r["offset"])
        dy, dx = int(r["offset"][0]), int(r["offset"][1])
        tok = ((Y + dy) // patch) * gw + ((X + dx) // patch)
        nf = float(r["nfwd"])
        k_t = r["k_sum"].sum(0) / (nf * L)                 # (T,) 每层每次前向的专家数
        nrm_t = r["norm_sum"].sum((0, 2)) / (nf * L)       # (T,) 每层每次前向的总贡献
        reg_k += w.T @ k_t; reg_norm += w.T @ nrm_t; reg_w += w.sum(0)
        kk = k_t[tok]
        px_k += kk; px_k2 += kk * kk; px_sel += (kk > 0.5); px_norm += nrm_t[tok]
        if "prior_sum_t" in r:
            p = (r["prior_sum_t"].sum(0) / nf)[tok]
            px_pr += p; px_pr2 += p * p; has_prior = True
        nday += 1
        if (n + 1) % 120 == 0 or n + 1 == len(items):
            print(f"  [{label}] {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    res = {"n_files": len(items), "n_days": nday, "layers": int(L), "experts": int(E),
           "k_by_region": reg_k / np.maximum(reg_w, 1),
           "norm_by_region": reg_norm / np.maximum(reg_w, 1),
           "norm_per_selection_by_region": reg_norm / np.maximum(reg_k, 1e-12),
           "px_k": px_k / nday, "px_sel": px_sel / nday, "px_norm": px_norm / nday,
           "px_k_var_within": px_k2 / nday - (px_k / nday) ** 2,
           "seconds": round(time.time() - t0, 1), "ctx": ctx}
    if has_prior:
        res["px_prior"] = px_pr / nday
        res["px_prior_var_within"] = px_pr2 / nday - (px_pr / nday) ** 2
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = out / "scales.json"
    M = {l: scan(l, d, a, out, scales) for l, d in specs}
    ctx = M[labels[0]]["ctx"]
    land = ctx.land
    nreg = len(ctx.region_names)
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [f"{ctx.region_ids[i]} {ctx.region_names[i]}" for i in order]
    rid_short = [ctx.region_ids[i] for i in order]
    rough = region_roughness(ctx, land)[order]

    def to_map(v):
        f = np.full(land.shape, np.nan)
        f[land] = v
        return f

    res = {"year": a.year, "members": a.members,
           "note_norm": "norm = 路由专家加权输出范数 / token 进 FFN 时的隐状态范数; 共享专家不在其中",
           "regions": {"display_ids": rid_short, "names": [ctx.region_names[i] for i in order],
                       "roughness_std_m": rough.round(1).tolist()},
           "models": {}}
    for l in labels:
        m = M[l]
        sel = m["px_sel"]
        bins = {"almost_never_lt2pct": float((sel < 0.02).mean()),
                "sometimes": float(((sel >= 0.02) & (sel < 0.5)).mean()),
                "often": float(((sel >= 0.5) & (sel < 0.98)).mean()),
                "almost_always_gt98pct": float((sel >= 0.98).mean()),
                "never_selected": float((sel == 0).mean())}
        sp_k = float(m["px_k"].var()); wi_k = float(m["px_k_var_within"].mean())
        entry = {"n_files": m["n_files"], "seconds": m["seconds"],
                 "k_by_region": m["k_by_region"][order].round(4).tolist(),
                 "routed_norm_by_region": m["norm_by_region"][order].round(5).tolist(),
                 "routed_norm_per_selection_by_region": m["norm_per_selection_by_region"][order].round(5).tolist(),
                 "selection_frequency_bins": bins,
                 "k_variance_split": {"between_pixels": round(sp_k, 4), "within_pixel_across_days": round(wi_k, 4),
                                      "spatial_share": round(sp_k / max(sp_k + wi_k, 1e-12), 3)}}
        if "px_prior" in m:
            sp_p = float(m["px_prior"].var()); wi_p = float(m["px_prior_var_within"].mean())
            entry["prior_variance_split"] = {"between_pixels": round(sp_p, 4),
                                             "within_pixel_across_days": round(wi_p, 4),
                                             "spatial_share": round(sp_p / max(sp_p + wi_p, 1e-12), 3)}
            ctx_l = m["ctx"]
            ctx_l.map(to_map(m["px_prior"]), f"prior_annual_mean_map_{slug(l)}", cmap="viridis",
                      scale_group="prior_map", cbar="annual mean difficulty prior",
                      title=f"{l} · annual mean difficulty prior")
        res["models"][l] = entry
        ctx_l = m["ctx"]
        ctx_l.map(to_map(sel), f"selection_frequency_map_{slug(l)}", cmap="magma",
                  scale_group="selfreq_map", vmin=0, vmax=1,
                  cbar="share of days with a routed expert",
                  title=f"{l} · share of days the pixel's token got a routed expert")
        ctx_l.map(to_map(m["px_k"]), f"mean_k_map_{slug(l)}", cmap="magma", scale_group="k_map",
                  cbar="routed experts per token per layer", title=f"{l} · mean routed experts")
        ctx_l.map(to_map(m["px_norm"]), f"routed_norm_map_{slug(l)}", cmap="viridis",
                  scale_group="norm_map", cbar="Σ‖w·f_e(x)‖ / ‖x‖ per layer per forward",
                  title=f"{l} · routed expert output relative to token hidden state")

    grouped_bars(ctx, {l: M[l]["norm_by_region"][order] for l in labels}, rlab,
                 "routed_norm_by_region", "Σ‖w·f_e(x)‖ / ‖x‖ per layer per forward",
                 "routed expert output relative to token hidden state · by region")
    grouped_bars(ctx, {l: M[l]["norm_per_selection_by_region"][order] for l in labels}, rlab,
                 "routed_norm_per_selection_by_region", "‖w·f_e(x)‖ / ‖x‖ per selection",
                 "routed expert output per single selection · by region")
    ctx.write_scales()
    np.savez_compressed(out / "share_stability.npz", region_display_ids=np.array(rid_short),
                        region_names=np.array([ctx.region_names[i] for i in order]), roughness=rough,
                        **{f"{q}__{slug(l)}": (M[l][q][order] if q.endswith("region") else M[l][q])
                           for l in labels for q in ("k_by_region", "norm_by_region",
                                                     "norm_per_selection_by_region", "px_k", "px_sel", "px_norm")})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for l in labels:
        e = res["models"][l]
        b = e["selection_frequency_bins"]
        print(f"\n[{l}] k 方差里空间占 {e['k_variance_split']['spatial_share']:.3f}"
              + (f"; 先验方差里空间占 {e['prior_variance_split']['spatial_share']:.3f}" if "prior_variance_split" in e else ""))
        print(f"   像素选中频率: 几乎从不 {b['almost_never_lt2pct']*100:.1f}% | 偶尔 {b['sometimes']*100:.1f}%"
              f" | 经常 {b['often']*100:.1f}% | 几乎每天 {b['almost_always_gt98pct']*100:.1f}%")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
