#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
expert_member_stability.py — MoE 路由在集合成员之间稳不稳定 (成员 0 能否代表全集合)
============================================================================
在少数几天上重放全部集合成员的采样轨迹(同种子方案, 每个成员各自的噪声与切块起点),
截获每层路由, 落回像素后回答:

  variance   逐像素路由量(D-EC 的 k̄; 两模型各专家的选中频率)的方差里, "日"与"成员"各占多少
  agreement  同一天不同成员之间的路由一致性: 逐像素专家频率向量的成员两两余弦, 分 t 档;
             对照量是同一成员不同天之间的余弦(日效应)
  region     区域级 k̄ 的成员间离散度(std/mean)对比区域间差异; 成员 0 与成员均值的区域排序一致性
  dominant   逐像素主导专家在成员间的一致率
  prior      [D-EC] 难度先验的成员间离散度

口径与 expert_effect_regions 一致: 区域统计只计有效域内像素, t 分 5 档, 域内 token 用同一定义。
需要 GPU; 6 天 × 32 成员 × 2 模型约 1 小时(login node 单 GPU)。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.expert_member_stability \\
      --model TC=runs/exp/<tc run> --model DEC=runs/exp/<dec run> \\
      --regions runs/exp/<regions>/regions_v1.npz --days 15 76 137 198 251 320 --members 32 --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import time
from itertools import combinations
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
from downscaling_4x.evaluation.routing_dump import RoutingCapture as Capture, T_BINS, tok_to_pixel
from downscaling_4x.tools.diagnostics.expert_effect_regions import parse_spec, slug
from downscaling_4x.training.train_jit import build_model


def pair_cosine(A, B, eps=1e-12):
    """A, B: (..., E, P) 频率向量场 -> 逐像素余弦的均值(忽略两边都为零的像素)。"""
    num = (A * B).sum(-2)
    den = np.sqrt((A * A).sum(-2) * (B * B).sum(-2))
    ok = den > eps
    return float((num[ok] / den[ok]).mean()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=run_dir")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--days", type=int, nargs="+", default=[15, 76, 137, 198, 251, 320])
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--year", type=int, default=2020)
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
        raise SystemExit("需要 GPU")
    ctx = RenderContext(models[0][1], a.regions, C.TARGETS[0], out, years=(y,),
                        model_tag="-vs-".join(slug(l) for l, _ in models) + "-members")
    land = ctx.land
    H, W = land.shape
    nreg = len(ctx.region_names)
    rid_land = ctx.region_id[land]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    P = int(land.sum())
    days, Mn, nb = list(a.days), a.members, len(T_BINS) - 1
    res = {"year": y, "days": days, "members": Mn, "steps": a.steps, "models": {}}

    for label, run in models:
        ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        args = ck["args"]
        II.check(args.get("era5_dir"), a.era5_dir, f"checkpoint {run}")
        target, mode = args["target"], args.get("mode", C.DEFAULT_MODE)
        mu_cache = MuCache(args["mu_cache"], [target])
        if args.get("stage_a_ckpt"):
            mu_cache.verify({target: args["stage_a_ckpt"]})
        II.check(mu_cache.manifest.get("era5_dir"), a.era5_dir, f"μ 缓存 {args['mu_cache']}")
        stats = Stats(a.era5_dir, a.daymet_dir)
        need, lags = C.pairing_history_days(mode), C.history_lags(mode)
        fi = FrameIndex([y], sorted({y - 1, y}) if need else [y], need, lags, split="dump")
        frame_of = {f: k for k, f in enumerate(fi.frames)}
        ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
        dd = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode, era5_cache_years=len(ds_years))
        if not np.array_equal(dd.mask, land):
            raise SystemExit("数据层有效域与分区文件 land 不一致")
        net = build_model(args, (H, W)).to(device)
        net.load_state_dict(dict(ck["model"]))
        net.eval()
        is_dec = getattr(net, "dec", None) is not None
        if is_dec:
            net.set_dec_progress(float(ck.get("samples") or 0) / max(1.0, float(args.get("duration", 1))))
            net.set_dec_eval("frame", 1.0)
        layers = net.moe_layers()
        L, E = len(layers), layers[0].n_experts
        patch, gh, gw = net.patch, net.x_embedder.gh, net.x_embedder.gw
        Tn = gh * gw
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
        print(f"[{label}] router={'dec' if is_dec else 'tc'} L={L} E={E} days={days} members={Mn}", flush=True)

        # 逐 (日, 成员) 的像素级量(只存有效域像素)
        k_px = np.zeros((len(days), Mn, L, P), np.float32)              # 逐层 k̄
        f_first = np.zeros((len(days), Mn, E, P), np.float16)           # 首层 MoE 专家频率
        f_last = np.zeros((len(days), Mn, E, P), np.float16)            # 末层
        Mt = min(Mn, 8)                                                 # 按 t 档只存前 8 个成员(内存)
        f_last_t = np.zeros((len(days), Mt, nb, E, P), np.float16)      # 末层按 t 档
        prior_px = np.zeros((len(days), Mn, P), np.float32) if is_dec else None
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
                for m in range(Mn):
                    g = torch.Generator(device=device)
                    g.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
                    offset = draw_patch_offset(patch, device, g)
                    dy, dx = int(offset[0]) % patch, int(offset[1]) % patch
                    tk = torch.zeros(L, Tn, device=device)
                    tf = torch.zeros(2, Tn, E, device=device)             # 首层 / 末层
                    tft = torch.zeros(nb, Tn, E, device=device)
                    nft = torch.zeros(nb, device=device)
                    tpr = torch.zeros(Tn, device=device)
                    n_fwd = 0

                    def after_forward(_m, _args, _out):
                        nonlocal n_fwd
                        n_fwd += 1
                        tbin = int(np.searchsorted(T_BINS, t_state["t"], side="right") - 1)
                        nft[tbin] += 1
                        for lidx, lay in enumerate(layers):
                            sel = cap.sel[lidx].float()
                            tk[lidx] += sel.sum(1)
                            if lidx == 0:
                                tf[0] += sel
                            if lidx == L - 1:
                                tf[1] += sel
                                tft[tbin] += sel
                        if cap.prior is not None:
                            tpr.add_(cap.prior)
                    h2 = net.register_forward_hook(after_forward)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        generate(net, cond_t, steps=a.steps, noise_scale=args["noise_scale"],
                                 t_eps=args["t_eps"], land=land_t, generator=g, offset=(dy, dx))
                    h2.remove()
                    k_px[di, m] = tok_to_pixel((tk / n_fwd).cpu().numpy(), gh, gw, patch, (dy, dx), H, W)[:, land]
                    fr = (tf / n_fwd).cpu().numpy().transpose(0, 2, 1)                         # (2, E, T)
                    f_first[di, m] = tok_to_pixel(fr[0], gh, gw, patch, (dy, dx), H, W)[:, land]
                    f_last[di, m] = tok_to_pixel(fr[1], gh, gw, patch, (dy, dx), H, W)[:, land]
                    if m < Mt:
                        frt = (tft / nft.clamp_min(1)[:, None, None]).cpu().numpy().transpose(0, 2, 1)   # (nb, E, T)
                        f_last_t[di, m] = tok_to_pixel(frt, gh, gw, patch, (dy, dx), H, W)[..., land]
                    if is_dec:
                        prior_px[di, m] = tok_to_pixel((tpr / n_fwd).cpu().numpy(), gh, gw, patch, (dy, dx), H, W)[land]
                print(f"  [{label}] day {day} ({di + 1}/{len(days)}) × {Mn} members  {time.time() - t0:.0f}s", flush=True)
        finally:
            hook.remove()
            cap.uninstall()
        del net
        torch.cuda.empty_cache()

        R = {"is_dec": is_dec, "L": L, "E": E}
        sl = slug(label)
        # ---- variance: 逐像素量在 (日, 成员) 上的方差分解 ----
        def var_sums(x):                                     # x: (D, M, P) -> (日, 成员, 残差, 总) 方差之和
            x = x.astype(np.float32)
            tot = x.var(axis=(0, 1))
            vd = x.mean(1).var(0)
            vm = x.mean(0).var(0)
            resid = np.maximum(tot - vd - vm, 0)
            ok = tot > 1e-12
            return np.array([vd[ok].sum(), vm[ok].sum(), resid[ok].sum(), tot[ok].sum()])

        def var_split(*xs):                                  # 若干 (D, M, P) 场合并后的 日 / 成员 / 残差 份额
            acc = sum(var_sums(x) for x in xs)
            s = acc[:3] / max(acc[3], 1e-12)
            return {"day": float(s[0]), "member": float(s[1]), "residual": float(s[2])}
        R["variance_share_expert_freq_last_layer"] = var_split(*[f_last[:, :, e, :] for e in range(E)])
        if is_dec:
            R["variance_share_k_layer_mean"] = var_split(k_px.mean(2))
            R["variance_share_prior"] = var_split(prior_px)
        # ---- agreement: 成员两两余弦 vs 同成员跨天余弦 ----
        pairs = list(combinations(range(Mn), 2))
        rng = np.random.default_rng(0)
        if len(pairs) > 96:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 96, replace=False)]
        def agree(F):                                        # F: (D, M, E, P)
            same_day = [pair_cosine(F[d, i].astype(np.float32), F[d, j].astype(np.float32)) for d in range(len(days)) for i, j in pairs[:32]]
            cross_day = [pair_cosine(F[d1, m].astype(np.float32), F[d2, m].astype(np.float32))
                         for m in range(min(Mn, 8)) for d1, d2 in combinations(range(len(days)), 2)]
            return float(np.nanmean(same_day)), float(np.nanmean(cross_day))
        R["agreement_first_layer"] = dict(zip(("members_same_day", "days_same_member"), agree(f_first)))
        R["agreement_last_layer"] = dict(zip(("members_same_day", "days_same_member"), agree(f_last)))
        pairs_all = pairs
        pairs = list(combinations(range(Mt), 2))
        R["agreement_last_layer_by_t"] = [dict(zip(("members_same_day", "days_same_member"), agree(f_last_t[:, :, b]))) for b in range(nb)]
        pairs = pairs_all
        # ---- dominant: 主导专家的成员一致率(末层) ----
        mean_f = f_last.astype(np.float32).mean(1)                       # (D, E, P)
        dom_mean = mean_f.argmax(1)                                      # (D, P)
        dom_m = f_last.astype(np.float32).argmax(2)                      # (D, M, P)
        agree_px = (dom_m == dom_mean[:, None, :]).mean(1)              # (D, P)
        R["dominant_expert_member_agreement_mean"] = float(agree_px.mean())
        amap = np.full((H, W), np.nan); amap[land] = agree_px.mean(0)
        ctx.map(amap, f"dominant_agreement_L{L - 1}_{sl}", cmap="viridis", vmin=0, vmax=1, scale_group="dom_agree",
                cbar="share of members agreeing with member-mean dominant expert", title=f"{label} · last MoE layer · dominant-expert agreement across members")
        # ---- region: k̄ 的成员间离散度 ----
        if is_dec:
            kreg = np.zeros((len(days), Mn, nreg))
            for r in range(nreg):
                sel = rid_land == r + 1
                kreg[:, :, r] = k_px.mean(2)[:, :, sel].mean(2)
            cv_member = kreg.std(1) / np.maximum(kreg.mean(1), 1e-9)                 # (D, R)
            between = kreg.mean(1).std(1) / np.maximum(kreg.mean(1).mean(1), 1e-9)   # (D,)
            R["region_k_member_cv_by_region"] = {rlab[j]: float(np.median(cv_member[:, order[j]])) for j in range(nreg)}
            R["region_k_member_cv_median"] = float(np.median(cv_member))
            R["region_k_between_region_cv_median"] = float(np.median(between))
            R["region_k_rank_spearman_member0_vs_mean"] = [float(spearmanr(kreg[d, 0], kreg[d].mean(0)).correlation) for d in range(len(days))]
            R["region_k_rank_spearman_members_pairwise_mean"] = float(np.mean([spearmanr(kreg[d, i], kreg[d, j]).correlation for d in range(len(days)) for i, j in pairs[:32]]))
            fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * nreg + 2.0), 3.8), constrained_layout=True)
            mk = kreg.mean((0, 1))[order]; sd = kreg.std(1).mean(0)[order]
            ax.bar(range(nreg), mk, yerr=sd, color="#4878a8", capsize=2)
            ax.set_xticks(range(nreg)); ax.set_xticklabels(rlab, fontsize=8)
            ax.set_ylabel("mean routed experts per token"); ax.set_title(f"{label} · region k̄ (bar = mean over days & members, error = std across members)", fontsize=9)
            ctx.savefig(fig, f"region_k_member_spread_{sl}")
            kstd = np.full((H, W), np.nan); kstd[land] = k_px.mean(2).std(1).mean(0)
            ctx.map(kstd, f"k_member_std_map_{sl}", cmap="magma", scale_group="k_member_std", cbar="std of k̄ across members (mean over days)", title=f"{label} · member spread of capacity")
            pstd = np.full((H, W), np.nan); pstd[land] = prior_px.std(1).mean(0)
            R["prior_member_std_over_day_std"] = float(prior_px.std(1).mean() / max(prior_px.mean(1).std(0).mean(), 1e-9))
            ctx.map(pstd, f"prior_member_std_map_{sl}", cmap="viridis", scale_group="prior_member_std", cbar="std of difficulty prior across members", title=f"{label} · member spread of difficulty prior")
        res["models"][label] = R
        np.savez_compressed(out / f"member_maps_{sl}.npz", k_px_layer_mean=k_px.mean(2), f_last_mean=mean_f,
                            dominant_agreement=agree_px, days=np.array(days),
                            **({"prior_px": prior_px} if is_dec else {}))

    # agreement 对比图: 成员间 vs 跨天, 分 t 档
    labels = list(res["models"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    xs = range(nb)
    for lab in labels:
        A = res["models"][lab]["agreement_last_layer_by_t"]
        ax.plot(xs, [d["members_same_day"] for d in A], marker="o", lw=1.2, label=f"{lab} · members, same day")
        ax.plot(xs, [d["days_same_member"] for d in A], marker="s", lw=1.0, ls="--", label=f"{lab} · days, same member")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"[{T_BINS[i]:.1f},{min(T_BINS[i + 1], 1):.1f})" for i in range(nb)])
    ax.set_xlabel("noise level t"); ax.set_ylabel("mean per-pixel cosine of expert-frequency vectors")
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("routing agreement · last MoE layer", fontsize=10)
    ctx.savefig(fig, "agreement_by_t")
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    keys = ["variance_share_expert_freq_last_layer"] + (["variance_share_k_layer_mean", "variance_share_prior"])
    xt, vals = [], {"day": [], "member": [], "residual": []}
    for lab in labels:
        for k in keys:
            if k in res["models"][lab]:
                xt.append(f"{lab}\n{k.replace('variance_share_', '')}")
                for c in vals:
                    vals[c].append(res["models"][lab][k][c])
    bottom = np.zeros(len(xt))
    for c in ("day", "member", "residual"):
        ax.bar(range(len(xt)), vals[c], bottom=bottom, label=c); bottom += np.array(vals[c])
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, fontsize=7)
    ax.set_ylabel("share of per-pixel variance"); ax.legend(fontsize=8); ax.set_title("routing variance: day vs member", fontsize=10)
    ctx.savefig(fig, "variance_split")
    ctx.write_scales()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for lab in labels:
        R = res["models"][lab]
        print(lab, json.dumps({k: v for k, v in R.items() if k not in ("region_k_member_cv_by_region",)}, ensure_ascii=False)[:1500])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
