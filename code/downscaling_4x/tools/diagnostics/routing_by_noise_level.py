#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
routing_by_noise_level.py — 容量沿噪声轴的去向, 与该噪声档上的误差分布对照
============================================================================
路由落盘按 5 个噪声档分开记了选中次数、亲和分与难度先验(sel_cnt_t / score_sum_t /
prior_sum_t), 因此不重采样就能回答:

  每档给出多少容量   域内 token 每层平均路由到的专家数, 逐档
  容量在空间上怎么分  逐 (区域, 档) 的选中次数占比 ÷ 该区域的域内 token 占比,
                      >1 表示该区域在该档上分到的容量高于它的 token 份额
  先验在空间上怎么分  逐 (区域, 档) 的先验(档内标准化后, 与进选择分的形式一致)
  误差在哪            逐 (区域, 档) 的逐 token 训练损失, 取自 difficulty_head_probe 落的表

最后给每档一个秩相关: 区域的"超额服务比"与区域的"相对误差"之间排得一致吗。

★两个 t 的来源不同★: 路由来自采样轨迹的确定性时间表, 误差来自训练侧那条 t 分布下的独立抽样。
两者都按同一组档边界归档, 因此"在噪声水平 t 上路由做了什么 / 误差有多大"可以并排, 但它们
不是同一批前向。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.routing_by_noise_level \\
      --model DEC=runs/exp/<dec eval2020> --model TC=... --model EC=... --members 0 \\
      --probe runs/exp/<head probe>/head_probe_table.npz \\
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
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, parse_spec, slug


def lines(ctx, x, series, name, xlabel, ylabel, title, hline=None):
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for lab, v in series.items():
        ax.plot(x, v, marker="o", ms=4, label=lab)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def scan(label, dump, a, out, scales):
    info = RD.load_routing_meta(dump)
    gh, gw = info["token_grid"]
    patch, L, E = info["patch"], info["n_moe_layers"], info["n_experts"]
    items = [(y, d, m) for (y, d, m) in RD.available_routing(dump)
             if y == a.year and m in a.members and (a.days is None or d in a.days)]
    if not items:
        raise SystemExit(f"{label}: 没有符合条件的路由文件")
    ctx = RenderContext(dump, a.regions, C.TARGETS[0], out, years=(a.year,),
                        model_tag=slug(label), scales_path=scales)
    nreg = len(ctx.region_names)
    wmap = TokenRegionWeights(ctx.land, ctx.region_id[ctx.land], gh, gw, patch, nreg)
    nb = len(RD.T_BINS) - 1

    nrm_rt = np.zeros((nreg, nb))        # 逐 (区域, 档) 的路由专家输出贡献
    nrm_tot = np.zeros(nb)               # 逐档 域内 token 的贡献总量
    has_norm_t = False
    sel_rt = np.zeros((nreg, nb))        # 逐 (区域, 档) 的选中次数(按有效格点摊)
    pri_rt = np.zeros((nreg, nb))        # 档内标准化后的先验, 按有效格点加权求和
    fwd_tok = np.zeros(nb)               # 逐档 域内 token-前向数
    sel_tot = np.zeros(nb)               # 逐档 域内选中次数
    wsum = np.zeros(nreg)
    has_prior = False
    t0 = time.time()
    for n, (y, d, m) in enumerate(items):
        r = RD.load_routing(dump, y, d, m)
        w = wmap(r["offset"])
        tin = np.asarray(r["tok_in"], bool)
        nft = r["nfwd_t"].astype(np.float64)
        sc = r["sel_cnt_t"]                                    # (L, nb, T, E)
        per_tok = sc.sum((0, 3))                               # (nb, T)
        for b in range(nb):
            sel_rt[:, b] += w.T @ per_tok[b]
            sel_tot[b] += per_tok[b][tin].sum()
            fwd_tok[b] += nft[b] * L * tin.sum()
        if "norm_sum_t" in r:
            has_norm_t = True
            nm = r["norm_sum_t"].sum((0, 3))                    # (nb, T) 对层与专家求和
            for b in range(nb):
                nrm_rt[:, b] += w.T @ nm[b]
                nrm_tot[b] += nm[b][tin].sum()
        if "prior_sum_t" in r:
            has_prior = True
            for b in range(nb):
                p = r["prior_sum_t"][b] / max(nft[b], 1)
                pin = p[tin]
                z = (p - pin.mean()) / max(pin.std(), 1e-9)    # 档内域内标准化
                pri_rt[:, b] += w.T @ z
        wsum += w.sum(0)
        if (n + 1) % 120 == 0 or n + 1 == len(items):
            print(f"  [{label}] {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    k_by_bin = sel_tot / np.maximum(fwd_tok, 1)                # 域内每 token 每层每前向的专家数
    share_tok = wsum / wsum.sum()                              # 各区域的有效格点份额
    share_sel = sel_rt / np.maximum(sel_rt.sum(0, keepdims=True), 1e-12)
    over = share_sel / np.maximum(share_tok[:, None], 1e-12)   # 超额服务比
    norm_by_bin = (nrm_tot / np.maximum(fwd_tok, 1)) if has_norm_t else None
    norm_per_sel = (nrm_tot / np.maximum(sel_tot, 1e-12)) if has_norm_t else None
    return {"ctx": ctx, "k_by_bin": k_by_bin, "over": over, "share_tok": share_tok,
            "norm_by_bin": norm_by_bin, "norm_per_sel": norm_per_sel,
            "norm_rt": (nrm_rt / np.maximum(wsum[:, None], 1)) if has_norm_t else None,
            "prior": (pri_rt / np.maximum(wsum[:, None], 1)) if has_prior else None,
            "n_files": len(items), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--probe", default=None, help="difficulty_head_probe 落的 head_probe_table.npz")
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
    nreg = len(ctx.region_names)
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]
    nb = len(RD.T_BINS) - 1
    blab = [f"{RD.T_BINS[b]:.1f}-{min(RD.T_BINS[b+1],1.0):.1f}" for b in range(nb)]

    res = {"year": a.year, "members": a.members, "t_bins": RD.T_BINS.tolist(),
           "note_t": "t=1 是数据端, t=0 是噪声端; 路由的 t 来自采样时间表, 误差的 t 来自训练侧分布下的独立抽样",
           "note_over": "超额服务比 = 该区域在该档拿到的选中次数占比 ÷ 该区域的有效格点占比; 1 表示按面积等分",
           "regions": {"display_ids": rlab, "names": rname},
           "note_norm": "贡献 = 路由专家加权输出范数 / token 进 FFN 时的隐状态范数; 共享专家不在其中",
           "models": {l: {"n_files": M[l]["n_files"],
                          "mean_k_in_domain_by_t_bin": M[l]["k_by_bin"].round(5).tolist(),
                          "expert_contribution_by_t_bin": (M[l]["norm_by_bin"].round(5).tolist()
                                                           if M[l]["norm_by_bin"] is not None else None),
                          "contribution_per_selection_by_t_bin": (M[l]["norm_per_sel"].round(5).tolist()
                                                                  if M[l]["norm_per_sel"] is not None else None),
                          "over_service_by_region_and_t": M[l]["over"][order].round(3).tolist()}
                      for l in labels}}
    lines(ctx, np.arange(nb), {l: M[l]["k_by_bin"] for l in labels}, "k_by_t_bin",
          "noise level bin (t; 1 = data end)", "routed experts per in-domain token per layer",
          "routing capacity by noise level")
    if any(M[l]["norm_by_bin"] is not None for l in labels):
        av = {l: M[l]["norm_by_bin"] for l in labels if M[l]["norm_by_bin"] is not None}
        lines(ctx, np.arange(nb), av, "expert_contribution_by_t_bin",
              "noise level bin (t; 1 = data end)", "Σ‖w·f_e(x)‖ / ‖x‖ per layer per forward",
              "routed expert output relative to token hidden state · by noise level")
        ps = {l: M[l]["norm_per_sel"] for l in labels if M[l]["norm_per_sel"] is not None}
        lines(ctx, np.arange(nb), ps, "contribution_per_selection_by_t_bin",
              "noise level bin (t; 1 = data end)", "‖w·f_e(x)‖ / ‖x‖ per selection",
              "what one routed-expert selection buys · by noise level")
        for l in av:
            M[l]["ctx"].heatmap(M[l]["norm_rt"][order], rlab, blab,
                                f"expert_contribution_region_by_t_{slug(l)}",
                                cbar="Σ‖w·f_e(x)‖ / ‖x‖ per layer per forward",
                                scale_group="contrib_rt", cmap="viridis",
                                title=f"{l} · routed expert contribution · region × noise level")
    for l in labels:
        ctx_l = M[l]["ctx"]
        ctx_l.heatmap(M[l]["over"][order], rlab, blab, f"over_service_region_by_t_{slug(l)}",
                      cbar="selections share / token share", scale_group="over_rt",
                      cmap="magma", title=f"{l} · capacity share relative to area · region × noise level")
        if M[l]["prior"] is not None:
            ctx_l.heatmap(M[l]["prior"][order], rlab, blab, f"prior_region_by_t_{slug(l)}",
                          cbar="within-bin standardized prior", scale_group="prior_rt",
                          cmap="RdBu_r", title=f"{l} · standardized difficulty prior · region × noise level")

    if a.probe:
        z = np.load(a.probe)
        tb = np.clip(np.searchsorted(RD.T_BINS, z["t"], side="right") - 1, 0, nb - 1)
        dmap = {i + 1: int(ctx.region_ids[i]) for i in range(nreg)}
        disp = np.array([dmap[int(r)] for r in z["region"]])
        err = np.exp(z["logerr"])                              # 逐 token 的训练损失
        loss_rt = np.full((nreg, nb), np.nan)
        cnt_rt = np.zeros((nreg, nb))
        for i, rd in enumerate(rlab):
            for b in range(nb):
                s = (disp == int(rd)) & (tb == b)
                cnt_rt[i, b] = s.sum()
                if s.sum() >= 30:
                    loss_rt[i, b] = err[s].mean()
        bin_mean = np.array([err[tb == b].mean() if (tb == b).any() else np.nan for b in range(nb)])
        rel_loss = loss_rt / bin_mean[None, :]                 # 档内相对误差
        ctx.heatmap(rel_loss, rlab, blab, "relative_loss_region_by_t",
                    cbar="per-token loss / bin mean", scale_group="loss_rt", cmap="magma",
                    title="per-token training loss relative to bin mean · region × noise level")
        lines(ctx, np.arange(nb), {"mean per-token loss": bin_mean},
              "loss_by_t_bin", "noise level bin (t; 1 = data end)", "per-token loss",
              "per-token training loss by noise level")
        rho = {}
        for l in labels:
            ov = M[l]["over"][order]
            rho[l] = [float(spearmanr(ov[:, b], rel_loss[:, b], nan_policy="omit").correlation)
                      for b in range(nb)]
        lines(ctx, np.arange(nb), rho, "service_vs_loss_rho_by_t",
              "noise level bin (t; 1 = data end)", "Spearman ρ across 19 regions",
              "does capacity go where the loss is? · by noise level", hline=0.0)
        res["loss"] = {"bin_mean_per_token_loss": bin_mean.round(6).tolist(),
                       "relative_loss_by_region_and_t": np.round(rel_loss, 3).tolist(),
                       "n_rows_by_region_and_t": cnt_rt.astype(int).tolist(),
                       "spearman_over_service_vs_relative_loss": {l: [round(v, 3) for v in rho[l]] for l in labels},
                       "probe": str(a.probe)}
    ctx.write_scales()
    np.savez_compressed(out / "by_noise_level.npz", region_display_ids=np.array(rlab),
                        region_names=np.array(rname), t_bins=RD.T_BINS,
                        **{f"over__{slug(l)}": M[l]["over"][order] for l in labels},
                        **{f"k__{slug(l)}": M[l]["k_by_bin"] for l in labels},
                        **({"rel_loss": rel_loss} if a.probe else {}))
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("\n逐档 域内平均 k:")
    for l in labels:
        print(f"  {l:<5}", [round(v, 4) for v in M[l]["k_by_bin"]])
    if any(M[l]["norm_by_bin"] is not None for l in labels):
        print("逐档 路由专家对 token 表示的贡献:")
        for l in labels:
            if M[l]["norm_by_bin"] is not None:
                v = M[l]["norm_by_bin"]
                print(f"  {l:<5}", [round(x, 4) for x in v], f"  跨度 {v.max()/max(v.min(),1e-9):.2f}x")
        print("逐档 每次选中买到多少:")
        for l in labels:
            if M[l]["norm_per_sel"] is not None:
                v = M[l]["norm_per_sel"]
                print(f"  {l:<5}", [round(x, 4) for x in v], f"  跨度 {v.max()/max(v.min(),1e-9):.2f}x")
    if a.probe:
        print("逐档 每 token 平均损失:", [round(v, 5) for v in res["loss"]["bin_mean_per_token_loss"]])
        print("逐档 ρ(超额服务比, 相对误差):")
        for l in labels:
            print(f"  {l:<5}", res["loss"]["spearman_over_service_vs_relative_loss"][l])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
