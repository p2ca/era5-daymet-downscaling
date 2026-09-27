#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
expert_effect_regions.py — 阶段 B JiT-MoE 的专家路由在哪里生效、效果如何 (TC 对 D-EC)
============================================================================
沿着与落场完全相同的采样轨迹(同种子、同切块起点、Heun 同步数)重放每一次网络前向, 在运行时
截获每个 MoE 层的路由结果, 不改动模型代码:

  TC   每 token 的 top-K 专家索引 (DSMoE.route)
  D-EC 每 token 被哪些专家选中 (DSMoE.route_dec 的选择掩膜), 以及难度头的逐 token 先验

把 token 级结果按该轨迹的切块起点落回像素, 再按分区与月份聚合, 得到:

  capacity   [D-EC] 逐像素平均专家数 k̄ 地图、(区域 × 月) 的 k̄ 与 k=0 占比
  territory  各 MoE 层的 (专家 × 区域) 份额热图、逐像素主导专家地图; 区域内有效专家数
  by_t       路由随噪声水平 t 的变化: 各 t 档专家份额的两两余弦距离(t 专业化指数), D-EC 的 k̄ 随 t
  prior      [D-EC] 难度头先验的地图与 (区域 × 月) 表, 与该单元实际 CRPS 的秩相关
  effect     与 stage_b_pairwise_regions 的单元矩阵联结: D-EC 的 k̄ 与 "D-EC 相对 TC 的 CRPS 增益"
             在 (区域, 日) 单元上的关系

口径: 每天只重放 1 个成员(成员 0), 统计覆盖该轨迹的全部网络前向; 区域统计只计有效域内像素;
"域内 token" 对两个模型用同一定义(块内任一像素在有效域内)。全部单图, 同组共色标。需要 GPU。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.expert_effect_regions \\
      --model TC=runs/exp/<jmb1 tc run> --model DEC=runs/exp/<jmb1 dec run> \\
      --regions runs/exp/<regions>/regions_v1.npz \\
      --cells runs/exp/<pairwise diag>/cells.npz --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.data.mu_cache import MuCache
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.models.jit_backbone import draw_patch_offset, token_domain_mask
from downscaling_4x.models.jit_sampler import generate
from downscaling_4x.training.train_jit import build_model

from downscaling_4x.evaluation.routing_dump import RoutingCapture as Capture, T_BINS, tok_to_pixel  # noqa: E402


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"--a/--b 需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def cosine_distance_matrix_mean(V):
    """V: (n, E) 份额向量 -> 两两 (1 − 余弦) 的均值。"""
    n = V.shape[0]
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    cs = Vn @ Vn.T
    iu = np.triu_indices(n, 1)
    return float((1 - cs[iu]).mean()) if iu[0].size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=run_dir, 可多次(TC / DEC)")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--cells", default="", help="stage_b_pairwise_regions 的 cells.npz, 给了就做 effect 联结")
    ap.add_argument("--cells-ref", default="jmb-dec-jda", help="cells.npz 里参考模型的标签")
    ap.add_argument("--cells-cmp", default="jmb-tc-jda", help="cells.npz 里对照模型的标签")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=1, help="每隔几天重放一天")
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = [parse_spec(s) for s in a.model]
    y = a.year
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本诊断要重放整条采样轨迹, 需要 GPU")
    days = list(range(0, C.DAYS_PER_YEAR, a.stride))
    if a.limit_days:
        days = days[:a.limit_days]

    # 分区 / 日历 / 绘图上下文(取第一个模型的目录只为 model_tag, 不读它的落场)
    ctx = RenderContext(models[0][1], a.regions, C.TARGETS[0], out, years=(y,),
                        model_tag="-vs-".join(slug(l) for l, _ in models))
    land = ctx.land
    H, W = land.shape
    nreg = len(ctx.region_names)
    rid_px = ctx.region_id
    month_of = ctx.month_of_day[y]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    months = list(range(1, 13))
    mlab = ctx.MONTHS

    results = {"year": y, "days": days, "steps": a.steps, "member": 0, "models": {}}
    per_model = {}
    for label, run in models:
        ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        args = ck["args"]
        II.check(args.get("era5_dir"), a.era5_dir, f"checkpoint {run}")
        target = args["target"]
        mode = args.get("mode", C.DEFAULT_MODE)
        if not args.get("mu_cache"):
            raise SystemExit(f"{label}: 本诊断只针对阶段 B(残差模式)的 checkpoint")
        mu_cache = MuCache(args["mu_cache"], [target])
        if args.get("stage_a_ckpt"):
            mu_cache.verify({target: args["stage_a_ckpt"]})
        II.check(mu_cache.manifest.get("era5_dir"), a.era5_dir, f"μ 缓存 {args['mu_cache']}")
        stats = Stats(a.era5_dir, a.daymet_dir)
        need, lags = C.pairing_history_days(mode), C.history_lags(mode)
        avail = sorted({y - 1, y}) if need else [y]
        fi = FrameIndex([y], avail, need, lags, split="dump")
        frame_of = {f: k for k, f in enumerate(fi.frames)}
        ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
        dd = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode,
                           era5_cache_years=len(ds_years))
        if not np.array_equal(dd.mask, land):
            raise SystemExit("数据层有效域与分区文件 land 不一致")
        net = build_model(args, (H, W))
        net.load_state_dict(dict(ck["model"]))
        net = net.to(device).eval()
        is_dec = getattr(net, "dec", None) is not None
        if is_dec:
            frac = float(ck.get("samples") or 0) / max(1.0, float(args.get("duration", 1)))
            net.set_dec_progress(frac)
            net.set_dec_eval("frame", 1.0)
        layers = net.moe_layers()
        L, E = len(layers), layers[0].n_experts
        lid = {id(m): k for k, m in enumerate(layers)}
        patch, gh, gw = net.patch, net.x_embedder.gh, net.x_embedder.gw
        Tn = gh * gw
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
        print(f"[{label}] router={'dec' if is_dec else args.get('router', 'tc')} MoE层={L} 专家={E} "
              f"token {gh}x{gw} patch {patch} 天数={len(days)}", flush=True)

        # 累加器
        pix_k = np.zeros((L, H, W))                 # 逐像素 k 之和(按天)
        pix_k0 = np.zeros((L, H, W))                # 逐像素 k=0 的前向占比之和(按天)
        pix_exp = np.zeros((L, E, H, W), np.float32)  # 逐像素各专家被选的前向占比之和(按天)
        pix_prior = np.zeros((H, W))
        pix_n = np.zeros((H, W))
        tb_exp = np.zeros((L, len(T_BINS) - 1, E))  # 域内 token 的专家选择计数, 按 t 档
        tb_k = np.zeros((L, len(T_BINS) - 1))
        tb_n = np.zeros((L, len(T_BINS) - 1))
        tspec_sum = np.zeros(L)                      # token 级 t 专业化指数之和(按天)
        tspec_n = 0
        cell_k = np.full((C.DAYS_PER_YEAR, nreg), np.nan)       # (日, 区域) 的 k̄(层平均)
        cell_k0 = np.full((C.DAYS_PER_YEAR, nreg), np.nan)
        cell_prior = np.full((C.DAYS_PER_YEAR, nreg), np.nan)
        cell_eff = np.full((C.DAYS_PER_YEAR, nreg), np.nan)     # 区域内有效专家数(层平均)
        reg_exp = np.zeros((L, nreg, E))                         # (层, 区域, 专家) 像素-天计数

        cap = Capture(net).install()
        t_state = {"t": None}
        hook = net.register_forward_pre_hook(lambda m, args_: t_state.__setitem__("t", float(args_[1][0])))
        t0 = time.time()
        try:
            for di, day in enumerate(days):
                cond, _tgt, _mask, _hr = dd.full(y, day, fi.history_of(frame_of[(y, day)]))
                cond_t = torch.from_numpy(cond[None]).float().to(device)
                mu = mu_cache.get(target, y, day)
                cond_t = torch.cat([cond_t, torch.from_numpy(mu[None, None]).float().to(device) * land_t], 1)
                g = torch.Generator(device=device)
                g.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + 0)
                offset = draw_patch_offset(patch, device, g)      # 与 generate 内部同一顺序抽取
                dy, dx = int(offset[0]) % patch, int(offset[1]) % patch
                tok_in = token_domain_mask(land_t, patch, (dy, dx), net.grid_hw).reshape(-1).cpu().numpy()

                # 本轨迹的 token 级累加(GPU)
                tk_sum = torch.zeros(L, Tn, device=device)
                tk0_sum = torch.zeros(L, Tn, device=device)
                texp = torch.zeros(L, Tn, E, device=device)
                nb = len(T_BINS) - 1
                texp_t = torch.zeros(L, nb, Tn, E, device=device)   # 按 t 档的 token×专家选择计数
                nfwd_t = torch.zeros(nb, device=device)
                tprior = torch.zeros(Tn, device=device)
                n_fwd = 0
                tok_in_t = torch.from_numpy(tok_in).to(device)

                def after_forward(_m, _args, _out):
                    nonlocal n_fwd
                    n_fwd += 1
                    tbin = int(np.searchsorted(T_BINS, t_state["t"], side="right") - 1)
                    nfwd_t[tbin] += 1
                    for lidx, m in enumerate(layers):
                        sel = cap.sel[lidx].float()                  # (T, E)
                        k = sel.sum(1)
                        tk_sum[lidx] += k
                        tk0_sum[lidx] += (k == 0).float()
                        texp[lidx] += sel
                        texp_t[lidx, tbin] += sel
                        tb_exp[lidx, tbin] += sel[tok_in_t].sum(0).cpu().numpy()
                        tb_k[lidx, tbin] += float(k[tok_in_t].sum())
                        tb_n[lidx, tbin] += float(tok_in_t.sum())
                    if cap.prior is not None:
                        tprior.add_(cap.prior)
                h2 = net.register_forward_hook(after_forward)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    generate(net, cond_t, steps=a.steps, noise_scale=args["noise_scale"],
                             t_eps=args["t_eps"], land=land_t, generator=g, offset=(dy, dx))
                h2.remove()
                if n_fwd != 2 * (a.steps - 1) + 1:
                    raise SystemExit(f"前向次数 {n_fwd} 与 Heun 预期 {2 * (a.steps - 1) + 1} 不符")

                # token 级 t 专业化: 同一域内 token 在各 t 档的专家选择频率向量, 两两 (1 − 余弦) 的均值
                freq = texp_t / nfwd_t.clamp_min(1)[None, :, None, None]           # (L, nb, T, E)
                fn = freq / freq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cs = torch.einsum("lbte,lcte->lbct", fn, fn)                        # (L, nb, nb, T)
                iu = torch.triu_indices(nb, nb, 1)
                dist = (1 - cs[:, iu[0], iu[1], :])                                 # (L, npair, T)
                has = (freq.sum(-1) > 0).all(1)                                     # (L, T) 各档都被选过的 token
                keep = has & tok_in_t[None, :]
                for lidx in range(L):
                    kk = keep[lidx]
                    if int(kk.sum()) > 0:
                        tspec_sum[lidx] += float(dist[lidx][:, kk].mean())
                tspec_n += 1
                k_tok = (tk_sum / n_fwd).cpu().numpy()             # (L, T)
                k0_tok = (tk0_sum / n_fwd).cpu().numpy()
                exp_tok = (texp / n_fwd).cpu().numpy()             # (L, T, E)
                pr_tok = (tprior / n_fwd).cpu().numpy() if cap.prior is not None else None
                k_px = tok_to_pixel(k_tok, gh, gw, patch, (dy, dx), H, W)        # (L, H, W)
                k0_px = tok_to_pixel(k0_tok, gh, gw, patch, (dy, dx), H, W)
                exp_px = tok_to_pixel(exp_tok.transpose(0, 2, 1), gh, gw, patch, (dy, dx), H, W)  # (L, E, H, W)
                pix_k += k_px; pix_k0 += k0_px; pix_exp += exp_px.astype(np.float32); pix_n += 1
                if pr_tok is not None:
                    pr_px = tok_to_pixel(pr_tok, gh, gw, patch, (dy, dx), H, W)
                    pix_prior += pr_px
                # (日, 区域) 单元
                for r in range(nreg):
                    m = land & (rid_px == r + 1)
                    if not m.any():
                        continue
                    cell_k[day, r] = k_px[:, m].mean()
                    cell_k0[day, r] = k0_px[:, m].mean()
                    if pr_tok is not None:
                        cell_prior[day, r] = pr_px[m].mean()
                    sh = exp_px[:, :, m].mean(2)                   # (L, E)
                    sh = sh / np.maximum(sh.sum(1, keepdims=True), 1e-12)
                    cell_eff[day, r] = float((1.0 / np.maximum((sh ** 2).sum(1), 1e-12)).mean())
                    reg_exp[:, r, :] += exp_px[:, :, m].sum(2)
                if di % 20 == 0 or di == len(days) - 1:
                    print(f"  [{label}] day {day} ({di + 1}/{len(days)}) offset=({dy},{dx}) "
                          f"{time.time() - t0:.0f}s", flush=True)
        finally:
            hook.remove()
            cap.uninstall()

        pm = {"is_dec": is_dec, "L": L, "E": E, "pix_k": pix_k / np.maximum(pix_n, 1), "pix_k0": pix_k0 / np.maximum(pix_n, 1),
              "pix_exp": pix_exp / np.maximum(pix_n, 1).astype(np.float32), "pix_prior": pix_prior / np.maximum(pix_n, 1),
              "tb_exp": tb_exp, "tb_k": tb_k, "tb_n": tb_n, "tspec": tspec_sum / max(tspec_n, 1),
              "cell_k": cell_k, "cell_k0": cell_k0,
              "cell_prior": cell_prior, "cell_eff": cell_eff, "reg_exp": reg_exp}
        per_model[label] = pm
        del net
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 汇总与出图
    valid_days = np.array(days)
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    def region_month(cell):
        return np.array([[np.nanmean(cell[valid_days[month_of[valid_days] == mo], i]) for mo in months] for i in order])

    for label, pm in per_model.items():
        L, E = pm["L"], pm["E"]
        R = {"is_dec": pm["is_dec"], "n_days": len(days)}
        sl = slug(label)
        # territory: (专家 × 区域) 份额, 逐层
        share = pm["reg_exp"] / np.maximum(pm["reg_exp"].sum(2, keepdims=True), 1e-12)   # (L, R, E)
        eff = 1.0 / np.maximum((share ** 2).sum(2), 1e-12)                                # (L, R)
        R["effective_experts_by_region_layer_mean"] = {rlab[j]: float(eff[:, order[j]].mean()) for j in range(nreg)}
        R["region_similarity_by_layer"] = [cosine_distance_matrix_mean(share[l][order]) for l in range(L)]
        for l in range(L):
            ctx.heatmap(share[l][order].T, [f"E{e}" for e in range(E)], rlab, f"expert_region_L{l}_{sl}",
                        cbar="share of routed selections", scale_group="expert_region_share", cmap="viridis",
                        vmin=0, vmax=float(np.quantile(share, 0.99)), title=f"{label} · MoE layer {l} · expert × region share")
        # 主导专家地图(首层与末层 MoE)
        for l in (0, L - 1):
            dom = np.where(land, pm["pix_exp"][l].argmax(0), np.nan)
            ctx.map(dom, f"dominant_expert_L{l}_{sl}", cmap="tab20", vmin=-0.5, vmax=E - 0.5,
                    scale_group="dominant_expert", cbar="dominant expert id", title=f"{label} · MoE layer {l} · dominant expert")
        # by_t
        tb_share = pm["tb_exp"] / np.maximum(pm["tb_exp"].sum(2, keepdims=True), 1e-12)      # (L, nb, E)
        R["t_specialization_by_layer"] = [float(x) for x in pm["tspec"]]          # token 级: 同一 token 跨 t 档的专家变化
        R["expert_share_t_distance_by_layer"] = [cosine_distance_matrix_mean(tb_share[l]) for l in range(L)]  # 专家用量随 t(EC 类按构造为 0)
        R["k_by_t_by_layer"] = (pm["tb_k"] / np.maximum(pm["tb_n"], 1)).round(4).tolist()
        if not pm["is_dec"]:
            ctx.heatmap(tb_share[L - 1].T, [f"E{e}" for e in range(E)],
                        [f"t∈[{T_BINS[i]:.1f},{min(T_BINS[i + 1], 1.0):.1f})" for i in range(len(T_BINS) - 1)],
                        f"expert_by_t_lastlayer_{sl}", cbar="share", scale_group="expert_t_share", cmap="viridis",
                        vmin=0, vmax=float(np.quantile(tb_share, 0.99)), title=f"{label} · last MoE layer · expert share by noise level t")
        # capacity(仅 D-EC 有信息量; TC 恒为 K)
        R["mean_k_overall"] = float(np.nanmean(pm["cell_k"][valid_days]))
        if pm["is_dec"]:
            kmap = np.where(land, pm["pix_k"].mean(0), np.nan)
            ctx.map(kmap, f"capacity_k_map_{sl}", cmap="magma", scale_group="k_map", cbar="mean routed experts per token (layer mean)",
                    title=f"{label} · mean routed experts per token")
            ctx.map(np.where(land, pm["pix_k0"].mean(0), np.nan), f"capacity_k0_share_map_{sl}", cmap="magma_r",
                    vmin=0, vmax=1, scale_group="k0_map", cbar="share of forwards with 0 routed experts", title=f"{label} · share of k=0")
            ctx.heatmap(region_month(pm["cell_k"]), rlab, mlab, f"capacity_k_region_month_{sl}", cbar="mean k",
                        scale_group="k_rm", cmap="magma", title=f"{label} · mean routed experts · region × month")
            ctx.heatmap(region_month(pm["cell_k0"]), rlab, mlab, f"capacity_k0_region_month_{sl}", cbar="share k=0",
                        scale_group="k0_rm", cmap="magma_r", vmin=0, vmax=1, title=f"{label} · share of k=0 · region × month")
            R["k_by_region"] = {rlab[j]: float(np.nanmean(pm["cell_k"][valid_days][:, order[j]])) for j in range(nreg)}
            R["k0_by_region"] = {rlab[j]: float(np.nanmean(pm["cell_k0"][valid_days][:, order[j]])) for j in range(nreg)}
            R["k_by_month"] = [float(np.nanmean(pm["cell_k"][valid_days[month_of[valid_days] == mo]])) for mo in months]
            # prior
            pmap = np.where(land, pm["pix_prior"], np.nan)
            ctx.map(pmap, f"prior_map_{sl}", cmap="viridis", scale_group="prior_map", cbar="difficulty-head prior (mean over forwards)",
                    title=f"{label} · difficulty prior")
            ctx.heatmap(region_month(pm["cell_prior"]), rlab, mlab, f"prior_region_month_{sl}", cbar="prior",
                        scale_group="prior_rm", cmap="viridis", title=f"{label} · difficulty prior · region × month")
        ctx.heatmap(region_month(pm["cell_eff"]), rlab, mlab, f"effective_experts_region_month_{sl}", cbar="effective experts",
                    scale_group="eff_rm", cmap="cividis", vmin=1, vmax=float(E), title=f"{label} · effective experts (1/Σp²) · region × month")
        results["models"][label] = R

    # by_t 对比图: t 专业化指数 逐层, 两模型并排
    labels = list(per_model)
    L = per_model[labels[0]]["L"]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    xg = np.arange(L); w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        ax.bar(xg + (i - (len(labels) - 1) / 2) * w, results["models"][lab]["t_specialization_by_layer"], w, label=lab)
    ax.set_xticks(xg); ax.set_xticklabels([f"L{l}" for l in range(L)])
    ax.set_ylabel("token-level mean (1 − cos) across t-bins"); ax.legend(fontsize=8)
    ax.set_title("same token, different experts at different noise levels? (in-domain tokens)", fontsize=10)
    ctx.savefig(fig, "t_specialization_by_layer")
    for lab in labels:
        if per_model[lab]["is_dec"]:
            fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
            kb = np.array(results["models"][lab]["k_by_t_by_layer"])
            for l in range(L):
                ax.plot(range(len(T_BINS) - 1), kb[l], marker="o", lw=1, label=f"L{l}")
            ax.set_xticks(range(len(T_BINS) - 1)); ax.set_xticklabels([f"[{T_BINS[i]:.1f},{min(T_BINS[i+1],1):.1f})" for i in range(len(T_BINS) - 1)])
            ax.set_xlabel("noise level t (0 = noise, 1 = data)"); ax.set_ylabel("mean routed experts per in-domain token")
            ax.legend(fontsize=7, ncol=3); ax.set_title(f"{lab} · capacity by noise level", fontsize=10)
            ctx.savefig(fig, f"k_by_t_{slug(lab)}")

    # effect: 与单元矩阵联结
    if a.cells:
        z = np.load(a.cells, allow_pickle=False)
        cr_ref, cr_cmp = z[f"crps_{slug(a.cells_ref)}"], z[f"crps_{slug(a.cells_cmp)}"]
        gain = (cr_cmp - cr_ref) / cr_cmp                                   # >0: 参考(DEC) 更好
        dec_lab = next((l for l in labels if per_model[l]["is_dec"]), None)
        if dec_lab is not None:
            ck_ = per_model[dec_lab]["cell_k"][valid_days]
            cp_ = per_model[dec_lab]["cell_prior"][valid_days]
            g_ = gain[valid_days]
            ok = np.isfinite(ck_) & np.isfinite(g_)
            eff_ = {"spearman_k_vs_gain_cells": float(spearmanr(ck_[ok], g_[ok]).correlation),
                    "spearman_prior_vs_crps_cells": float(spearmanr(cp_[ok], cr_ref[valid_days][ok]).correlation),
                    "spearman_k_vs_crps_cells": float(spearmanr(ck_[ok], cr_ref[valid_days][ok]).correlation)}
            qs = np.quantile(ck_[ok], np.linspace(0.1, 0.9, 9))
            dec_bin = np.digitize(ck_[ok], qs)
            eff_["gain_by_k_decile"] = [float(g_[ok][dec_bin == i].mean()) for i in range(10)]
            eff_["k_decile_edges"] = [float(ck_[ok].min())] + qs.tolist() + [float(ck_[ok].max())]
            win_reg = [float((g_[:, i] > 0).mean()) for i in order]
            k_reg = [float(np.nanmean(ck_[:, i])) for i in order]
            eff_["spearman_region_k_vs_region_winshare"] = float(spearmanr(k_reg, win_reg).correlation)
            results["effect"] = eff_
            fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
            ax.bar(range(10), np.array(eff_["gain_by_k_decile"]) * 100, color="#4878a8")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xticks(range(10)); ax.set_xticklabels([f"D{i + 1}" for i in range(10)])
            ax.set_xlabel(f"{dec_lab} mean routed experts per token, decile over (region, day) cells")
            ax.set_ylabel(f"CRPS gain vs {a.cells_cmp} [%]")
            ax.set_title("does more capacity go where D-EC gains?", fontsize=10)
            ctx.savefig(fig, "effect_gain_by_k_decile")
            fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
            ax.scatter(k_reg, win_reg, color="#4878a8")
            for j in range(nreg):
                ax.annotate(rlab[j], (k_reg[j], win_reg[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("region mean routed experts per token (D-EC)"); ax.set_ylabel(f"share of days D-EC beats {a.cells_cmp}")
            ax.set_title("region capacity vs win share", fontsize=10)
            ctx.savefig(fig, "effect_region_k_vs_winshare")

    ctx.write_scales()
    np.savez_compressed(out / "routing_maps.npz",
                        **{f"{k}_{slug(l)}": v for l, pm in per_model.items() for k, v in pm.items()
                           if isinstance(v, np.ndarray)},
                        days=np.array(days), region_display_ids=np.array(ctx.region_ids))
    json.dump(results, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for lab in labels:
        R = results["models"][lab]
        print(f"{lab}: mean_k {R['mean_k_overall']:.3f}  t-spec by layer {[round(x, 3) for x in R['t_specialization_by_layer']]}  "
              f"region-sim {[round(x, 3) for x in R['region_similarity_by_layer']]}")
    if "effect" in results:
        print("effect:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results["effect"].items() if k != "k_decile_edges"})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
