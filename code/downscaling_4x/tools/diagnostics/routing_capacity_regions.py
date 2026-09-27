#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
routing_capacity_regions.py — 路由容量花在了哪里: 域外 token、逐区域专家数
============================================================================
回答两件事:

1. ★容量有多少花在有效域外的 token 上★。EC 不排除域外 token, 每专家在整帧内挑 top-C,
   于是一部分容量落在没有真值、不计入任何指标的 token 上; D-EC 的 drop 成分把域外 token
   的选择分置 -inf 并按域内 token 数算容量, 这部分容量就回到域内。本工具直接从落盘的
   tok_in 与 sel_cnt 数出这个比例, 逐层给出。

2. ★收回来的容量落到哪个区域★。逐区域给每 token 每层平均路由到的专家数 k 与 k=0 的占比,
   再与该区域的 CRPS 相对差(--gain)并排, 看"多给专家"与"分数更好"是否落在同一批区域。
   注意这只是并排看, 不是因果: 路由与分数都由同一个模型产生。

口径: token 的域内判定与模型一致(块内任一像素在有效域内即域内); 区域权重 = 该 token 覆盖的
块在各区域内的有效格点数, 按落盘记录的切块起点 offset 逐条算, 与 plot_expert_region_month
共用同一份实现。k 对层取平均(每层一次路由决策), 与 plot_expert_region_month 的 k 同口径。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.routing_capacity_regions \\
      --model DEC=runs/exp/<dec eval2020> --model TC=runs/exp/<tc eval2020> \\
      --model EC=runs/exp/<ec eval2020> --members 0 \\
      --regions runs/exp/<regions>/regions_v1.npz \\
      --gain runs/exp/<fullset gain>/summary.json --ref DEC --out runs/exp/<diag>
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
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, parse_spec, slug


def pool_sum(x, P):
    H, W = x.shape
    return x.reshape(H // P, P, W // P, P).sum((1, 3))


def region_roughness(ctx, land, patch=16):
    """区域内地形粗糙度: patch×patch 块内高程标准差, 落回像素后取区域均值。"""
    z = np.where(land, ctx.elevation(), 0.0)
    n = pool_sum(land.astype(np.float64), patch)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = pool_sum(z * z, patch) / np.maximum(n, 1) - (pool_sum(z, patch) / np.maximum(n, 1)) ** 2
    px = np.repeat(np.repeat(np.sqrt(np.maximum(var, 0.0)), patch, 0), patch, 1)[land]
    rid = ctx.region_id[land]
    nreg = len(ctx.region_names)
    s = np.bincount(rid, weights=px, minlength=nreg + 1)[1:]
    c = np.bincount(rid, minlength=nreg + 1)[1:]
    return s / np.maximum(c, 1)


def grouped_bars(ctx, groups, labels, name, ylabel, title, rotate=45, hline=None):
    k = len(groups)
    x = np.arange(len(labels))
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(labels) + 2.2), 3.9), constrained_layout=True)
    for i, (lab, vals) in enumerate(groups.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=lab)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def scatter_regions(ctx, x, y, names, name, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.scatter(x, y, s=26, color="#4878a8")
    for xi, yi, nm in zip(x, y, names):
        ax.annotate(nm, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).correlation)
    ax.set_title(f"{title}  ·  Spearman ρ = {rho:.2f}", fontsize=10)
    return ctx.savefig(fig, name), rho


def scan_model(label, dump, a, out, scales):
    info = RD.load_routing_meta(dump)
    if info is None:
        raise SystemExit(f"{dump} 下没有 routing/meta.json")
    gh, gw = info["token_grid"]
    patch, L, E = info["patch"], info["n_moe_layers"], info["n_experts"]
    items = [(y, d, m) for (y, d, m) in RD.available_routing(dump)
             if y == a.year and (a.members is None or m in a.members)
             and (a.days is None or d in a.days)]
    if not items:
        raise SystemExit(f"{label}: 没有符合条件的路由文件")
    ctx = RenderContext(dump, a.regions, C.TARGETS[0], out, years=(a.year,),
                        model_tag=slug(a.ref), scales_path=scales)
    land = ctx.land
    nreg = len(ctx.region_names)
    wmap = TokenRegionWeights(land, ctx.region_id[land], gh, gw, patch, nreg)

    sel_in = np.zeros(L); sel_out = np.zeros(L)          # 选中次数: 域内 / 域外 token
    n_in_fwd = 0.0; n_out_fwd = 0.0                      # token-前向数, 作分母
    reg_k = np.zeros(nreg); reg_k0 = np.zeros(nreg); reg_w = np.zeros(nreg)
    t0 = time.time()
    for n, (y, d, m) in enumerate(items):
        r = RD.load_routing(dump, y, d, m)
        w = wmap(r["offset"])                            # (T, nreg) 有效格点计数
        tin = np.asarray(r["tok_in"], bool)
        if not np.array_equal(tin, w.sum(1) > 0):
            raise SystemExit(f"{label} {y}-d{d}-m{m}: 落盘的 tok_in 与按 offset 算出的"
                             "域内 token 不一致, 说明 offset 或 token 网格对不上")
        sc = r["sel_cnt"]                                # (L, T, E)
        if not np.array_equal(sc.sum(-1), r["k_sum"]):
            raise SystemExit(f"{label} {y}-d{d}-m{m}: sel_cnt 逐专家求和与 k_sum 不符")
        nf = float(r["nfwd"])
        sel_in += sc[:, tin].sum((1, 2)); sel_out += sc[:, ~tin].sum((1, 2))
        n_in_fwd += tin.sum() * nf; n_out_fwd += (~tin).sum() * nf
        k_t = r["k_sum"].sum(0) / (nf * L)               # (T,) 每层每次前向的专家数
        k0_t = r["k0_cnt"].sum(0) / (nf * L)
        reg_k += w.T @ k_t; reg_k0 += w.T @ k0_t; reg_w += w.sum(0)
        if (n + 1) % 120 == 0 or n + 1 == len(items):
            print(f"  [{label}] {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    tot = sel_in + sel_out
    return {"ctx": ctx, "layers": L, "experts": E, "n_files": len(items),
            "out_share_by_layer": sel_out / np.maximum(tot, 1),
            "k_in_by_layer": sel_in / max(n_in_fwd, 1),
            "k_out_by_layer": sel_out / max(n_out_fwd, 1),
            "mean_k_in": float(sel_in.sum() / max(n_in_fwd * L, 1)),
            "mean_k_out": float(sel_out.sum() / max(n_out_fwd * L, 1)),
            "reg_k": reg_k / np.maximum(reg_w, 1), "reg_k0": reg_k0 / np.maximum(reg_w, 1),
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--ref", required=True, help="做差时的参考模型标签(通常是 DEC)")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--gain", default=None, help="完整评测集 CRPS 相对差的 summary.json(plot_cross_set_gain --fields)")
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    if a.ref not in labels:
        raise SystemExit(f"--ref {a.ref} 不在模型列表 {labels} 里")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = out / "scales.json"

    M = {}
    for lab, d in specs:
        M[lab] = scan_model(lab, d, a, out, scales)
    ctx = M[a.ref]["ctx"]
    nreg = len(ctx.region_names)
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [f"{ctx.region_ids[i]} {ctx.region_names[i]}" for i in order]
    rid_short = [ctx.region_ids[i] for i in order]
    rough = region_roughness(ctx, ctx.land)[order]

    res = {"year": a.year, "members": a.members, "ref": a.ref,
           "note_k": "k = 每 token 每层平均路由到的专家数(不含共享专家); 区域值按 token 覆盖的有效格点数加权",
           "regions": {"display_ids": rid_short, "names": [ctx.region_names[i] for i in order],
                       "roughness_std_m": rough.round(1).tolist()},
           "models": {}}
    for lab in labels:
        m = M[lab]
        res["models"][lab] = {"n_files": m["n_files"], "layers": int(m["layers"]),
                              "out_of_domain_share_by_layer": m["out_share_by_layer"].round(4).tolist(),
                              "out_of_domain_share_overall": float(m["out_share_by_layer"].mean().round(4)),
                              "mean_k_in_domain": round(m["mean_k_in"], 4),
                              "mean_k_out_of_domain": round(m["mean_k_out"], 4),
                              "k_in_domain_by_layer": m["k_in_by_layer"].round(4).tolist(),
                              "k_out_of_domain_by_layer": m["k_out_by_layer"].round(4).tolist(),
                              "k_by_region": m["reg_k"][order].round(4).tolist(),
                              "share_k0_by_region": m["reg_k0"][order].round(4).tolist(),
                              "seconds": m["seconds"]}

    # ---- 图 ----
    nl = int(M[a.ref]["layers"])
    grouped_bars(ctx, {l: M[l]["out_share_by_layer"] for l in labels}, [f"L{i}" for i in range(nl)],
                 "capacity_out_of_domain_by_layer", "share of selections",
                 "share of routing capacity spent on out-of-domain tokens", rotate=0)
    grouped_bars(ctx, {l: M[l]["reg_k"][order] for l in labels}, rlab,
                 "mean_k_by_region", "routed experts per token per layer",
                 "mean routed experts per in-domain token · by region")
    grouped_bars(ctx, {l: M[l]["reg_k0"][order] for l in labels}, rlab,
                 "share_k0_by_region", "share of forwards with k=0",
                 "share of token-forwards with no routed expert · by region")

    gain = None
    if a.gain:
        g = json.load(open(a.gain))
        if g.get("ref") != a.ref:
            raise SystemExit(f"--gain 的参考模型是 {g.get('ref')}, 与 --ref {a.ref} 不符")
        gain = {r["cmp"]: np.array(g["by_region"][r["cmp"]]["pooled_overall"]) for r in g["by_cmp"]}
        res["gain_source"] = a.gain
    rhos = {}
    for lab in labels:
        if lab == a.ref:
            continue
        dk = M[a.ref]["reg_k"][order] - M[lab]["reg_k"][order]
        _, rho = scatter_regions(ctx, dk, rough, rid_short, f"dk_vs_roughness_{slug(lab)}",
                                 f"k({a.ref}) − k({lab})  [experts per token per layer]",
                                 "region terrain roughness [m]",
                                 f"extra routed experts vs terrain · {a.ref} − {lab}")
        rhos[f"dk_vs_roughness_{lab}"] = rho
        if gain is not None and lab in gain:
            _, rho2 = scatter_regions(ctx, dk, gain[lab], rid_short, f"dk_vs_crps_gain_{slug(lab)}",
                                      f"k({a.ref}) − k({lab})  [experts per token per layer]",
                                      f"CRPS relative difference ({lab} − {a.ref}) / {lab}  [%]",
                                      f"extra routed experts vs CRPS difference · {a.ref} − {lab}")
            rhos[f"dk_vs_crps_gain_{lab}"] = rho2
    if gain is not None:
        for lab in gain:
            _, rho = scatter_regions(ctx, rough, gain[lab], rid_short, f"crps_gain_vs_roughness_{slug(lab)}",
                                     "region terrain roughness [m]",
                                     f"CRPS relative difference ({lab} − {a.ref}) / {lab}  [%]",
                                     f"CRPS difference vs terrain · {a.ref} − {lab}")
            rhos[f"crps_gain_vs_roughness_{lab}"] = rho
    res["spearman"] = {k: round(v, 3) for k, v in rhos.items()}

    ctx.write_scales()
    np.savez_compressed(out / "capacity.npz", region_display_ids=np.array(rid_short),
                        region_names=np.array([ctx.region_names[i] for i in order]),
                        roughness=rough,
                        **{f"k__{slug(l)}": M[l]["reg_k"][order] for l in labels},
                        **{f"k0__{slug(l)}": M[l]["reg_k0"][order] for l in labels},
                        **{f"outshare__{slug(l)}": M[l]["out_share_by_layer"] for l in labels})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n{'模型':<6}{'域外容量占比':>14}{'域内平均k':>12}{'域外平均k':>12}")
    for lab in labels:
        m = res["models"][lab]
        print(f"{lab:<6}{m['out_of_domain_share_overall']*100:13.1f}%{m['mean_k_in_domain']:12.3f}{m['mean_k_out_of_domain']:12.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
