#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
difficulty_head_probe.py — 难度头预测的准不准: 在测试年上直接对拍它自己的监督目标
============================================================================
难度头的训练监督是★本次前向的逐 token 损失★(diff²·weight 按路由同一套 token 网格池化后取
log 再标准化)。那个量只在训练时存在, 落场里没有, 因此"头预测得准不准"和"头预测的东西对不对"
一直分不开。本工具用同一个 checkpoint 在指定年份重跑纯前向, 同时取出头的输出与它的监督目标,
逐 token 落表, 于是可以分别回答:

  ρ(先验, 逐 token 损失)   头把自己的目标学到了多少
  ρ(先验, μ 误差)          头的输出与"阶段B 要去噪的残差幅度"的关系
  ρ(逐 token 损失, μ 误差) 目标本身与残差幅度的关系 —— 头即使完美, 上限也在这里

复用训练侧的 jit_vloss 口径: 同样的 t 分布、同样的 noise_scale、同样的残差构造与 token 池化,
只是不反传、不更新。每天抽 --draws 次噪声, 每次一个独立的切块起点, 与训练一致。

需要 GPU。login node 上跑要设 MIOPEN_DISABLE_CACHE=1 与可写的 MIOPEN_USER_DB_PATH /
MIOPEN_CUSTOM_CACHE_DIR, 否则 conv 报 miopenStatusInternalError。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.difficulty_head_probe \\
      --run runs/exp/<dec 训练 run> --regions runs/exp/<regions>/regions_v1.npz \\
      --year 2020 --draws 4 --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
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
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.models.jit_backbone import draw_patch_offset, token_domain_mask, token_pool
from downscaling_4x.training.train_jit import build_model, make_residual
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, slug


def safe_spearman(x, y, nmin=30):
    g = np.isfinite(x) & np.isfinite(y)
    if g.sum() < nmin or np.ptp(x[g]) == 0 or np.ptp(y[g]) == 0:
        return np.nan
    return float(spearmanr(x[g], y[g]).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="D-EC 训练 run 目录(取 ckpt.pt)")
    ap.add_argument("--which", choices=["ckpt", "last"], default="ckpt")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--draws", type=int, default=4, help="每天抽几次 (t, 噪声, 切块起点)")
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(a.run) / f"{a.which}.pt", map_location="cpu", weights_only=False)
    args = ck["args"]
    target = args["target"]
    ti = C.TARGETS.index(target)
    mode = args.get("mode", C.DEFAULT_MODE)

    mu_cache, sigma_r = None, 1.0
    if args.get("mu_cache"):
        mu_cache = MuCache(args["mu_cache"], [target])
        rs = args.get("residual_scale", "")
        sigma_r = float(json.loads(Path(str(rs)).read_text())["residual_std"]) \
            if Path(str(rs)).is_file() else float(rs)
        assert sigma_r > 0

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    days_all = a.days if a.days is not None else list(range(C.DAYS_PER_YEAR))
    if a.limit_days:
        days_all = days_all[:a.limit_days]
    days = [(a.year, d) for d in days_all]
    avail = sorted({a.year - 1, a.year}) if need else [a.year]
    fi = FrameIndex([a.year], avail, need, lags, split="dump")
    frame_of = {f: k for k, f in enumerate(fi.frames)}
    days = [d for d in days if d in frame_of]
    ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
    dd = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode,
                       era5_cache_years=len(ds_years))
    H, W = dd.H, dd.W
    land = dd.mask

    net = build_model(args, (H, W))
    net.load_state_dict(dict(ck["model"]))
    net = net.to(device).eval()
    if getattr(net, "dec", None) is None or net.dec_head is None:
        raise SystemExit("该 checkpoint 没有难度头(不是 D-EC 或 prior 成分关着)")
    frac = float(ck.get("samples") or 0) / max(1.0, float(args.get("duration", 1)))
    lam = net.set_dec_progress(frac)
    net.set_dec_eval("frame", 1.0)
    patch, gh, gw = int(net.patch), net.x_embedder.gh, net.x_embedder.gw
    T = gh * gw

    ctx = RenderContext(Path(a.run), a.regions, target, out, years=(a.year,),
                        model_tag="headprobe", scales_path=out / "scales.json")
    if not np.array_equal(ctx.land, land):
        raise SystemExit("分区文件的 land 与数据管线的有效域不一致")
    nreg = len(ctx.region_names)
    wmap = TokenRegionWeights(land, ctx.region_id[land], gh, gw, patch, nreg)
    Y, X = np.nonzero(land)
    land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)

    rows = {k: [] for k in ("day", "draw", "t", "region", "npix", "pred", "logerr", "mu_err")}
    t0 = time.time()
    for n, (y, dayi) in enumerate(days):
        cond, tgt_all, _m, _raw = dd.full(y, dayi, fi.history_of(frame_of[(y, dayi)]))
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        tgt_t = torch.from_numpy(tgt_all[ti:ti + 1][None]).float().to(device) * land_t
        mu_err_px = None
        if mu_cache is not None:
            mu_np = mu_cache.get(target, y, dayi)
            mu_t = torch.from_numpy(mu_np[None, None]).float().to(device)
            mu_err_px = (tgt_t - mu_t * land_t).abs()          # 归一化空间的 |y − μ|
            tgt_t, cond_t = make_residual(tgt_t, mu_t, land_t, cond_t, sigma_r)
        w = land_t
        g = torch.Generator(device=device)
        g.manual_seed(a.seed * 100003 + (y * 1000 + dayi) * 131)
        for dr in range(a.draws):
            t = torch.sigmoid(torch.randn(1, device=device, generator=g)
                              * args["p_std"] + args["p_mean"])
            tb = t.view(1, 1, 1, 1)
            e = torch.randn(tgt_t.shape, device=device, generator=g) * args["noise_scale"]
            z = tb * tgt_t + (1.0 - tb) * e
            off = draw_patch_offset(patch, device, g)
            dy, dx = int(off[0]) % patch, int(off[1]) % patch
            with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                                 enabled=(device != "cpu")):
                x_hat, info = net(z, t, cond_t, offset=off, domain_mask=land_t, return_dec=True)
            diff = (tgt_t - x_hat.float()) / (1.0 - tb).clamp_min(args["t_eps"])
            num = token_pool(diff.square() * w, patch, off, net.grid_hw).reshape(-1)
            den = token_pool(w, patch, off, net.grid_hw).reshape(-1)
            pred = info["pred"].reshape(-1)
            tok_in = token_domain_mask(land_t, patch, (dy, dx), net.grid_hw).reshape(-1)
            valid = (den > 0) & tok_in
            if mu_err_px is not None:
                mnum = token_pool(mu_err_px * w, patch, off, net.grid_hw).reshape(-1)
                mu_tok = (mnum / den.clamp_min(1e-12)).detach().cpu().numpy()
            else:
                mu_tok = np.full(T, np.nan)
            wr = wmap((dy, dx))
            v = valid.detach().cpu().numpy()
            if not np.array_equal(v, wr.sum(1) > 0):
                raise SystemExit(f"{y}-d{dayi} draw{dr}: 域内 token 与按 offset 算出的不一致")
            sel = np.nonzero(v)[0]
            err = (num / den.clamp_min(1e-12)).detach().cpu().numpy()
            rows["day"].append(np.full(sel.size, dayi, np.int16))
            rows["draw"].append(np.full(sel.size, dr, np.int8))
            rows["t"].append(np.full(sel.size, float(t), np.float32))
            rows["region"].append((np.argmax(wr[sel], axis=1) + 1).astype(np.int16))
            rows["npix"].append(wr[sel].sum(1).astype(np.float32))
            rows["pred"].append(pred.detach().cpu().numpy()[sel].astype(np.float32))
            rows["logerr"].append(np.log(err[sel] + 1e-6).astype(np.float32))
            rows["mu_err"].append(mu_tok[sel].astype(np.float32))
        if (n + 1) % 20 == 0 or n + 1 == len(days):
            print(f"  {n + 1}/{len(days)} 天  {time.time() - t0:.0f}s", flush=True)
    tab = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez_compressed(out / "head_probe_table.npz", **tab)

    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    dmap = {i + 1: int(ctx.region_ids[i]) for i in range(nreg)}
    disp = np.array([dmap[int(r)] for r in tab["region"]])
    month = ctx.month_of_day[a.year][tab["day"]]
    P, Lg, Mu = tab["pred"], tab["logerr"], tab["mu_err"]
    logmu = np.log(np.maximum(Mu, 1e-9))

    res = {"run": str(a.run), "year": a.year, "draws": a.draws, "n_days": len(days),
           "n_rows": int(P.size), "lambda": round(lam, 4), "sigma_r": sigma_r,
           "note_target": "监督目标 = 逐 token 的 diff²·weight 均值, 表里存 log; 相关用秩, 与是否取 log 无关",
           "note_mu": "mu_err 是归一化空间的 |y − μ| 的 token 均值, 不是 K",
           "overall": {
               "rho_pred_vs_logerr": safe_spearman(P, Lg),
               "rho_pred_vs_mu_err": safe_spearman(P, logmu),
               "rho_logerr_vs_mu_err": safe_spearman(Lg, logmu)},
           "by_region": {}, "by_t_bin": {}}
    qs = np.quantile(tab["t"], np.linspace(0, 1, 6)); qs[-1] += 1e-9
    tb_idx = np.clip(np.searchsorted(qs, tab["t"], side="right") - 1, 0, 4)
    for b in range(5):
        s = tb_idx == b
        res["by_t_bin"][f"t{qs[b]:.3f}-{qs[b+1]:.3f}"] = {
            "n": int(s.sum()), "rho_pred_vs_logerr": safe_spearman(P[s], Lg[s]),
            "rho_pred_vs_mu_err": safe_spearman(P[s], logmu[s])}
    for i, rd in enumerate(rlab):
        s = disp == int(rd)
        res["by_region"][rd] = {"name": ctx.region_names[order[i]], "n": int(s.sum()),
                                "mean_pred": float(P[s].mean()), "mean_logerr": float(Lg[s].mean()),
                                "rho_pred_vs_logerr": safe_spearman(P[s], Lg[s]),
                                "rho_pred_vs_mu_err": safe_spearman(P[s], logmu[s]),
                                "rho_logerr_vs_mu_err": safe_spearman(Lg[s], logmu[s])}

    months = list(range(1, 13))
    mlab = [ctx.MONTHS[m - 1] for m in months]
    for qn, qv, cb in (("logerr", Lg, "Spearman ρ (prior, per-token loss)"),
                       ("mu_err", logmu, "Spearman ρ (prior, |y − μ|)")):
        Mrm = np.full((nreg, 12), np.nan)
        for i, rd in enumerate(rlab):
            for j, mm in enumerate(months):
                s = (disp == int(rd)) & (month == mm)
                Mrm[i, j] = safe_spearman(P[s], qv[s])
        ctx.heatmap(Mrm, rlab, mlab, f"head_pred_vs_{qn}_region_month", cbar=cb,
                    scale_group="head_rho", cmap="RdBu_r", vmin=-1, vmax=1,
                    title=f"difficulty prior vs {qn} · region × month")
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.bar(np.arange(nreg) - 0.2, [res["by_region"][r]["rho_pred_vs_logerr"] for r in rlab], 0.4,
           label="prior vs per-token loss")
    ax.bar(np.arange(nreg) + 0.2, [res["by_region"][r]["rho_pred_vs_mu_err"] for r in rlab], 0.4,
           label="prior vs |y − μ|")
    ax.set_xticks(np.arange(nreg)); ax.set_xticklabels(rlab, fontsize=8)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Spearman ρ"); ax.legend(fontsize=8)
    ax.set_title("how well the difficulty head predicts its own target · by region", fontsize=10)
    ctx.savefig(fig, "head_rho_by_region")
    ctx.write_scales()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("\n全域:", json.dumps(res["overall"], ensure_ascii=False))
    print(f"{'区':<15}{'ρ(先验,逐token损失)':>22}{'ρ(先验,|y−μ|)':>18}{'ρ(损失,|y−μ|)':>18}")
    for rd in rlab:
        b = res["by_region"][rd]
        print(f"{rd:>2} {b['name']:<12}{b['rho_pred_vs_logerr']:22.3f}{b['rho_pred_vs_mu_err']:18.3f}"
              f"{b['rho_logerr_vs_mu_err']:18.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
