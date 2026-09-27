#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_routing.py — 从 jit_dump 落盘的路由直接出图, 不重放采样
============================================================================
读 <dump>/routing/ 下的逐成员逐日路由, 按切块起点落回像素, 在成员与日期上平均后出:

  capacity   逐像素平均专家数 k̄、k=0 占比的地图; (区域 × 月) 的 k̄、k=0 占比
  territory  各 MoE 层 (专家 × 区域) 份额热图; 首层与末层的主导专家地图
  diversity  (区域 × 月) 的参与专家数 1/Σp²; 选择过少的格子置空(见 --min-k)
  strength   (区域 × 月) 与地图: 每次前向施加的门控权重总量、专家输出占 token 隐状态的比例
  prior      [D-EC] 难度先验的地图与 (区域 × 月)
  by_t       各 t 档专家份额的两两余弦距离, D-EC 的 k̄ 随 t; token 级 t 专业化指数

参与专家数是 1/Σp² 型的多样性计数, 只说明有多少专家在分摊流量, 不说明专家是否有用;
某格总选择次数太少时份额向量退化成近似均匀, 会假性地逼近 E, 因此按平均专家数设门槛。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_routing --dump runs/exp/<eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz [--members 0] [--out <dir>]
============================================================================
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def cos_dist_mean(V):
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    cs = Vn @ Vn.T
    iu = np.triu_indices(V.shape[0], 1)
    return float((1 - cs[iu]).mean()) if iu[0].size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="jit_dump 输出目录(含 routing/)")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", default=None, help="缺省 <dump>/routing_figs")
    ap.add_argument("--members", type=int, nargs="+", default=None, help="只用这些成员(缺省全部已落盘成员)")
    ap.add_argument("--days", type=int, nargs="+", default=None, help="只用这些日(缺省全部)")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--label", default=None, help="图题用的模型名(缺省取 run 目录名)")
    ap.add_argument("--min-k", type=float, default=0.05, help="参与专家数的门槛: 格内平均专家数低于此值置空")
    ap.add_argument("--scales", default=None,
                    help="共享色标文件; 多个模型指向同一个才使各自的热图与地图颜色可比")
    a = ap.parse_args()

    dump = Path(a.dump)
    info = RD.load_routing_meta(dump)
    if info is None:
        raise SystemExit(f"{dump} 下没有 routing/meta.json")
    label = a.label or Path(info.get("run", dump.name)).name
    out = Path(a.out) if a.out else dump / "routing_figs"
    items = [(y, d, m) for (y, d, m) in RD.available_routing(dump) if y == a.year
             and (a.members is None or m in a.members) and (a.days is None or d in a.days)]
    if not items:
        raise SystemExit("没有符合条件的路由文件")
    gh, gw = info["token_grid"]
    patch, L, E, K = info["patch"], info["n_moe_layers"], info["n_experts"], info["top_k"]
    nb = len(info["t_bins"]) - 1
    is_dec = info.get("dec") is not None
    ctx = RenderContext(dump, a.regions, C.TARGETS[0], out, years=(a.year,), model_tag=slug(label),
                        scales_path=a.scales)
    land = ctx.land
    H, W = land.shape
    nreg = len(ctx.region_names)
    rid_land = ctx.region_id[land]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    month_of = ctx.month_of_day[a.year]
    mlab = ctx.MONTHS
    days_used = sorted({d for _, d, _ in items})

    # ---- 逐文件累加(像素级只在有效域上) ----
    P = int(land.sum())
    pix_k = np.zeros((L, P)); pix_k0 = np.zeros((L, P)); pix_freq = np.zeros((L, E, P), np.float32)
    pix_gate = np.zeros((L, P)); pix_norm = np.zeros((L, P)) if info["norms"] else None
    pix_prior = np.zeros(P); n_prior = 0; n_files = 0
    tb_exp = np.zeros((L, nb, E)); tb_k = np.zeros((L, nb)); tb_n = np.zeros((L, nb))
    tspec = np.zeros(L); n_tspec = 0
    # (日, 区域) 单元: 逐文件累加后按天平均
    cell_sum = {k: np.zeros((C.DAYS_PER_YEAR, nreg)) for k in ("k", "k0", "gate", "norm", "prior")}
    cell_cnt = np.zeros((C.DAYS_PER_YEAR, nreg))
    cell_sel = np.zeros((C.DAYS_PER_YEAR, nreg, E))          # 各区域各日的专家选择计数(层与像素求和)
    reg_exp = np.zeros((L, nreg, E))
    for (y, d, m) in items:
        r = RD.load_routing(dump, y, d, m)
        off = tuple(int(v) for v in r["offset"]); nf = r["nfwd"]
        freq = r["sel_cnt"] / nf                                          # (L, T, E)
        k = r["k_sum"] / nf; k0 = r["k0_cnt"] / nf                        # (L, T)
        gate = r["gate_sum"].sum(-1) / nf                                 # (L, T) 每次前向施加的门控总量
        to_px = lambda x: RD.tok_to_pixel(x, gh, gw, patch, off, H, W)[..., land]
        k_px, k0_px, g_px = to_px(k), to_px(k0), to_px(gate)
        f_px = to_px(freq.transpose(0, 2, 1))                             # (L, E, P)
        pix_k += k_px; pix_k0 += k0_px; pix_gate += g_px; pix_freq += f_px.astype(np.float32); n_files += 1
        if pix_norm is not None and "norm_sum" in r:
            n_px = to_px(r["norm_sum"].sum(-1) / nf)
            pix_norm += n_px
        if "prior_sum_t" in r:
            pr_px = to_px(r["prior_sum_t"].sum(0) / nf)
            pix_prior += pr_px; n_prior += 1
        # by_t: 域内 token
        tin = r["tok_in"]
        sct = r["sel_cnt_t"]                                              # (L, nb, T, E)
        tb_exp += sct[:, :, tin].sum(2)
        tb_k += sct[:, :, tin].sum((2, 3))
        tb_n += (r["nfwd_t"][None, :] * tin.sum())
        ft = sct / np.maximum(r["nfwd_t"], 1)[None, :, None, None]
        fn = ft / np.maximum(np.linalg.norm(ft, axis=-1, keepdims=True), 1e-12)
        cs = np.einsum("lbte,lcte->lbct", fn, fn)
        iu = np.triu_indices(nb, 1)
        dist = 1 - cs[:, iu[0], iu[1], :]                                 # (L, npair, T)
        has = (ft.sum(-1) > 0).all(1) & tin[None, :]
        for l in range(L):
            if has[l].any():
                tspec[l] += dist[l][:, has[l]].mean()
        n_tspec += 1
        # (日, 区域)
        for rr in range(nreg):
            mm = rid_land == rr + 1
            if not mm.any():
                continue
            cell_sum["k"][d, rr] += k_px[:, mm].mean(); cell_sum["k0"][d, rr] += k0_px[:, mm].mean()
            cell_sum["gate"][d, rr] += g_px[:, mm].mean()
            if pix_norm is not None and "norm_sum" in r:
                cell_sum["norm"][d, rr] += n_px[:, mm].mean()
            if "prior_sum_t" in r:
                cell_sum["prior"][d, rr] += pr_px[mm].mean()
            cell_cnt[d, rr] += 1
            cell_sel[d, rr] += f_px[:, :, mm].sum((0, 2))
            reg_exp[:, rr, :] += f_px[:, :, mm].sum(2)
    cells = {k: np.where(cell_cnt > 0, v / np.maximum(cell_cnt, 1), np.nan) for k, v in cell_sum.items()}
    share_cells = cell_sel / np.maximum(cell_sel.sum(-1, keepdims=True), 1e-12)
    eff_cells = np.where(cells["k"] >= a.min_k, 1.0 / np.maximum((share_cells ** 2).sum(-1), 1e-12), np.nan)

    def region_month(cell):
        return np.array([[np.nanmean(cell[np.array([d for d in days_used if month_of[d] == mo]), i])
                          if any(month_of[d] == mo for d in days_used) else np.nan for mo in range(1, 13)] for i in order])
    def _fill(land_, v):
        g = np.full(land_.shape, np.nan); g[land_] = v; return g

    res = {"dump": str(dump), "label": label, "router": info["router"], "n_files": n_files, "members": sorted({m for _, _, m in items}),
           "days": days_used, "min_k": a.min_k}
    sl = slug(label)
    # ---- capacity ----
    kmean = pix_k.mean(0) / n_files
    res["mean_k_land"] = float(kmean.mean())
    if is_dec:
        ctx.map(_fill(land, kmean), f"capacity_k_map_{sl}", cmap="magma", scale_group="k_map", cbar="mean routed experts per token (layer mean)", title=f"{label} · capacity")
        ctx.map(_fill(land, pix_k0.mean(0) / n_files), f"capacity_k0_share_map_{sl}", cmap="magma_r", vmin=0, vmax=1, scale_group="k0_map", cbar="share of forwards with 0 routed experts", title=f"{label} · share of k=0")
        ctx.heatmap(region_month(cells["k"]), rlab, mlab, f"capacity_k_region_month_{sl}", cbar="mean k", scale_group="k_rm", cmap="magma", title=f"{label} · mean routed experts · region × month")
        ctx.heatmap(region_month(cells["k0"]), rlab, mlab, f"capacity_k0_region_month_{sl}", cbar="share k=0", scale_group="k0_rm", cmap="magma_r", vmin=0, vmax=1, title=f"{label} · share of k=0 · region × month")
        res["k_by_region"] = {rlab[j]: float(np.nanmean(cells["k"][days_used][:, order[j]])) for j in range(nreg)}
    # ---- territory ----
    share = reg_exp / np.maximum(reg_exp.sum(2, keepdims=True), 1e-12)
    for l in range(L):
        ctx.heatmap(share[l][order].T, [f"E{e}" for e in range(E)], rlab, f"expert_region_L{l}_{sl}", cbar="share of routed selections",
                    scale_group="expert_region_share", cmap="viridis", vmin=0, vmax=float(np.quantile(share, 0.99)), title=f"{label} · MoE layer {l} · expert × region share")
    for l in (0, L - 1):
        dom = pix_freq[l].argmax(0).astype(float)
        dom[pix_freq[l].sum(0) <= 0] = np.nan
        ctx.map(_fill(land, dom), f"dominant_expert_L{l}_{sl}", cmap="tab20", vmin=-0.5, vmax=E - 0.5, scale_group="dominant_expert", cbar="dominant expert id", title=f"{label} · MoE layer {l} · dominant expert")
    res["region_routing_dissimilarity_by_layer"] = [cos_dist_mean(share[l][order]) for l in range(L)]
    # ---- diversity ----
    ctx.heatmap(region_month(eff_cells), rlab, mlab, f"participating_experts_region_month_{sl}", cbar="participating experts (1/Σp²)",
                scale_group="eff_rm", cmap="cividis", vmin=1, vmax=float(E), title=f"{label} · participating experts · region × month (blank: mean k < {a.min_k})")
    # ---- strength ----
    ctx.heatmap(region_month(cells["gate"]), rlab, mlab, f"gate_total_region_month_{sl}", cbar="gate weight per forward", scale_group="gate_rm", cmap="viridis", title=f"{label} · total gate weight per forward · region × month")
    ctx.map(_fill(land, pix_gate.mean(0) / n_files), f"gate_total_map_{sl}", cmap="viridis", scale_group="gate_map", cbar="gate weight per forward (layer mean)", title=f"{label} · total gate weight")
    if pix_norm is not None:
        ctx.heatmap(region_month(cells["norm"]), rlab, mlab, f"expert_output_ratio_region_month_{sl}", cbar="Σ_e ‖w·f_e(x)‖/‖x‖ per forward", scale_group="norm_rm", cmap="viridis", title=f"{label} · expert output / token norm · region × month")
        ctx.map(_fill(land, pix_norm.mean(0) / n_files), f"expert_output_ratio_map_{sl}", cmap="viridis", scale_group="norm_map", cbar="Σ_e ‖w·f_e(x)‖/‖x‖ (layer mean)", title=f"{label} · expert output / token norm")
    # ---- prior ----
    if n_prior:
        ctx.map(_fill(land, pix_prior / n_prior), f"prior_map_{sl}", cmap="viridis", scale_group="prior_map", cbar="difficulty-head prior", title=f"{label} · difficulty prior")
        ctx.heatmap(region_month(cells["prior"]), rlab, mlab, f"prior_region_month_{sl}", cbar="prior", scale_group="prior_rm", cmap="viridis", title=f"{label} · difficulty prior · region × month")
    # ---- by_t ----
    tb_share = tb_exp / np.maximum(tb_exp.sum(2, keepdims=True), 1e-12)
    res["expert_share_t_distance_by_layer"] = [cos_dist_mean(tb_share[l]) for l in range(L)]
    res["t_specialization_token_level_by_layer"] = (tspec / max(n_tspec, 1)).tolist()
    res["k_by_t_by_layer"] = (tb_k / np.maximum(tb_n, 1)).round(4).tolist()
    tl = [f"[{info['t_bins'][i]:.1f},{min(info['t_bins'][i + 1], 1):.1f})" for i in range(nb)]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.bar(range(L), res["t_specialization_token_level_by_layer"], color="#4878a8")
    ax.set_xticks(range(L)); ax.set_xticklabels([f"L{l}" for l in range(L)])
    ax.set_ylabel("token-level mean (1 − cos) across t-bins"); ax.set_title(f"{label} · same token, different experts at different t?", fontsize=10)
    ctx.savefig(fig, f"t_specialization_by_layer_{sl}")
    if is_dec:
        fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
        for l in range(L):
            ax.plot(range(nb), res["k_by_t_by_layer"][l], marker="o", lw=1, label=f"L{l}")
        ax.set_xticks(range(nb)); ax.set_xticklabels(tl); ax.set_xlabel("noise level t"); ax.set_ylabel("mean routed experts per in-domain token")
        ax.legend(fontsize=7, ncol=3); ax.set_title(f"{label} · capacity by noise level", fontsize=10)
        ctx.savefig(fig, f"k_by_t_{sl}")
    else:
        ctx.heatmap(tb_share[L - 1].T, [f"E{e}" for e in range(E)], tl, f"expert_by_t_lastlayer_{sl}", cbar="share", scale_group="expert_t_share",
                    cmap="viridis", vmin=0, vmax=float(np.quantile(tb_share, 0.99)), title=f"{label} · last MoE layer · expert share by noise level t")
    ctx.write_scales()
    np.savez_compressed(out / "routing_maps.npz", pix_k=pix_k / n_files, pix_k0=pix_k0 / n_files, pix_freq=pix_freq / n_files,
                        pix_gate=pix_gate / n_files, **({"pix_norm": pix_norm / n_files} if pix_norm is not None else {}),
                        **({"pix_prior": pix_prior / n_prior} if n_prior else {}), cells_k=cells["k"], cells_k0=cells["k0"],
                        cells_eff=eff_cells, cells_gate=cells["gate"], cells_norm=cells["norm"], cells_prior=cells["prior"],
                        reg_exp=reg_exp, land=land, days=np.array(days_used))
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("k_by_region",)}, ensure_ascii=False)[:800])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
