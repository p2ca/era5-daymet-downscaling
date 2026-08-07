#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_big_check.py — 阶段 B 大检验(72 验证日的统计诊断)
============================================================================
默认取每月 5/15/25 日 x 12 月 x 2018/2019 = 72 天(均匀覆盖季节, 无挑拣空间),
对锁定的融合配置采样 N member, 在作业内直接归约, 不落 member 全场:

  * rank histogram —— 逐陆地像素的真值名次计数, 跨全部天累加;
  * 功率谱 —— 固定全陆地方框上的径向谱, truth / mu / ens-mean / 单 member 四线
    (降水在 log1p(mm) 空间, 温度在物理空间);
  * 逐月 CRPS 与 CRPSS = 1 - CRPS/MAE_mu (以阶段 A 均值为参考);
  * spread-skill —— 逐像素按 spread 分箱的 RMSE 曲线(对角线 = 完美校准)
    与逐月 spread/RMSE_ens 比;
  * 降水另记 expm1 钳制比例(逐 member 超上界像素占比, 训练良好应≈0)。

多卡: 按天切分给 SLURM rank, 各写 parts/ 分片; --merge-only 汇总并出全部单图。
member 种子与集合检验同式, 重合日子的成员逐位一致。
============================================================================
"""
import argparse
import datetime
import glob
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.mu_cache import MuCache
from era5_daymet.models.patching import GridPatching2D
from era5_daymet.models.preconditioning import EDMPrecondSuperResolution
from era5_daymet.models.stochastic_sampler import stochastic_sampler
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.stage_b_mean import pin_ocean


def default_days(years, doms=(5, 15, 25)):
    """每月固定日 x 12 月 x 各年 -> [(year, 0基日序), ...]; 365 日历下均匀覆盖季节。"""
    out = []
    for y in years:
        for m in range(1, 13):
            for dom in doms:
                doy = (datetime.date(y, m, dom) - datetime.date(y, 1, 1)).days
                out.append((y, doy))
    return out


def month_of(y, doy):
    return (datetime.date(y, 1, 1) + datetime.timedelta(days=doy)).month


def merge(out_dir, target, unit, members):
    parts_m = {}
    for fp in sorted(glob.glob(str(Path(out_dir) / "parts" / "big_part_rank*.json"))):
        parts_m.update(json.load(open(fp)))
    rank_counts = np.zeros(members + 1)
    psd = None
    ss = None
    env_above = env_below = err2 = spread2 = land = None
    for fp in sorted(glob.glob(str(Path(out_dir) / "parts" / "big_part_rank*.npz"))):
        z = np.load(fp)
        rank_counts += z["rank_counts"]
        psd = z["psd_sum"] if psd is None else psd + z["psd_sum"]
        ss = z["ss"] if ss is None else ss + z["ss"]
        if "env_above" in z.files:
            env_above = z["env_above"] if env_above is None else env_above + z["env_above"]
            env_below = z["env_below"] if env_below is None else env_below + z["env_below"]
            err2 = z["err2_map"] if err2 is None else err2 + z["err2_map"]
            spread2 = z["spread2_map"] if spread2 is None else spread2 + z["spread2_map"]
            if land is None:
                land = z["land"].astype(bool)
    n_days = len(parts_m)
    psd = psd / max(n_days, 1)

    # ---- 逐月聚合 ----
    months = sorted({v["month"] for v in parts_m.values()})
    bym = {mo: [v for v in parts_m.values() if v["month"] == mo] for mo in months}
    monthly = {mo: {
        "crps": float(np.mean([e["crps_ens"] for e in es])),
        "mae_mu": float(np.mean([e["mae_mu"] for e in es])),
        "rmse_ens": float(np.mean([e["rmse_ens_mean"] for e in es])),
        "spread": float(np.mean([e["spread_land_mean"] for e in es])),
        "clip_frac": float(np.mean([e.get("clip_frac", 0.0) for e in es])),
        "n_days": len(es)} for mo, es in bym.items()}
    for mo in months:
        m = monthly[mo]
        m["crpss"] = 1.0 - m["crps"] / max(m["mae_mu"], 1e-9)
        m["spread_ratio"] = m["spread"] / max(m["rmse_ens"], 1e-9)

    agg = {k: float(np.mean([v[k] for v in parts_m.values()]))
           for k in ("crps_ens", "mae_mu", "rmse_mu", "rmse_ens_mean",
                     "rmse_member_avg", "spread_land_mean")}
    agg["crpss"] = 1.0 - agg["crps_ens"] / agg["mae_mu"]

    payload = {"target": target, "unit": unit, "members": members, "n_days": n_days,
               "per_day": parts_m, "monthly": monthly, "overall": agg,
               "rank_counts": rank_counts.tolist()}
    json.dump(payload, open(Path(out_dir) / f"bigcheck_{target}.json", "w"),
              indent=1, ensure_ascii=False)

    figd = Path(out_dir) / "figs"
    figd.mkdir(exist_ok=True)

    # rank histogram
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    freq = rank_counts / rank_counts.sum()
    ax.bar(range(members + 1), freq, color="#4878a8", width=0.85)
    ax.axhline(1.0 / (members + 1), color="k", lw=1, ls="--", label="flat (calibrated)")
    ax.set_xlabel("rank of truth among members"); ax.set_ylabel("frequency")
    ax.set_title(f"rank histogram  {target}  ({n_days} days x land pixels)")
    ax.legend(frameon=False)
    fig.savefig(figd / "rank_hist.png", dpi=140); plt.close(fig)

    # 功率谱四线
    fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
    labels = ["truth", "mu (stage-A)", "ens_mean", "member"]
    styles = [("k", "-"), ("0.55", "--"), ("#4878a8", "-"), ("#d9822b", "-")]
    for i, (lab, (c, ls)) in enumerate(zip(labels, styles)):
        cur = psd[i][1:]
        ax.loglog(np.arange(1, len(cur) + 1), cur, color=c, ls=ls, lw=1.6, label=lab)
    space = "log1p(mm) space" if target == TD.PRECIP else "physical (K)"
    ax.set_xlabel("radial wavenumber (cycles / box)"); ax.set_ylabel("power")
    ax.set_title(f"radial power spectrum  {target}  ({space}, {n_days}-day mean)")
    ax.legend(frameon=False)
    fig.savefig(figd / "psd.png", dpi=140); plt.close(fig)

    # 逐月 CRPS vs muMAE / CRPSS / spread ratio / clip_frac
    mos = months
    x = np.arange(len(mos))
    fig, ax = plt.subplots(figsize=(8.6, 4), constrained_layout=True)
    ax.bar(x - 0.2, [monthly[m]["crps"] for m in mos], 0.4, label="CRPS (ensemble)", color="#4878a8")
    ax.bar(x + 0.2, [monthly[m]["mae_mu"] for m in mos], 0.4, label="MAE (mu, reference)", color="#9aa0a6")
    ax.set_xticks(x); ax.set_xticklabels(mos); ax.set_xlabel("month")
    ax.set_ylabel(unit); ax.set_title(f"monthly CRPS vs mu-MAE  {target}")
    ax.legend(frameon=False)
    fig.savefig(figd / "crps_monthly.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4), constrained_layout=True)
    vals = [monthly[m]["crpss"] for m in mos]
    ax.bar(x, vals, 0.6, color=["#2C7A57" if v >= 0 else "#b3452c" for v in vals])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(mos); ax.set_xlabel("month")
    ax.set_ylabel("CRPSS vs mu"); ax.set_title(f"monthly CRPSS = 1 - CRPS/MAE_mu  {target}")
    fig.savefig(figd / "crpss_monthly.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4), constrained_layout=True)
    ax.bar(x, [monthly[m]["spread_ratio"] for m in mos], 0.6, color="#4878a8")
    ax.axhline(1.0, color="k", lw=1, ls="--", label="calibrated (=1)")
    ax.set_xticks(x); ax.set_xticklabels(mos); ax.set_xlabel("month")
    ax.set_ylabel("spread / RMSE(ens_mean)"); ax.set_title(f"monthly spread-skill ratio  {target}")
    ax.legend(frameon=False)
    fig.savefig(figd / "spread_ratio_monthly.png", dpi=140); plt.close(fig)

    # spread-skill 分箱曲线
    cnt, s_sum, e_sum = ss
    ok = cnt > 1000                                   # 样本过少的箱不画
    ms_, rm_ = s_sum[ok] / cnt[ok], np.sqrt(e_sum[ok] / cnt[ok])
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.plot(ms_, rm_, "o-", color="#4878a8", lw=1.6, label="binned RMSE vs spread")
    lim = max(ms_.max(), rm_.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect calibration")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"ensemble spread [{unit}]"); ax.set_ylabel(f"RMSE of ens_mean [{unit}]")
    ax.set_title(f"spread-skill  {target}  ({n_days} days, pixel-binned)")
    ax.legend(frameon=False)
    fig.savefig(figd / "spread_skill.png", dpi=140); plt.close(fig)

    if target == TD.PRECIP:
        fig, ax = plt.subplots(figsize=(8.6, 4), constrained_layout=True)
        ax.bar(x, [monthly[m]["clip_frac"] for m in mos], 0.6, color="#b3452c")
        ax.set_xticks(x); ax.set_xticklabels(mos); ax.set_xlabel("month")
        ax.set_ylabel("clipped pixel fraction")
        ax.set_title("monthly expm1-clamp fraction (red flag if >> 0)")
        fig.savefig(figd / "nclip_monthly.png", dpi=140); plt.close(fig)

    # 空间校准图: 出界频率 + 幅度比值(逐像素多天累计)
    if env_above is not None and land is not None:
        freq = (env_above + env_below) / max(n_days, 1)
        np.savez_compressed(Path(out_dir) / "spatial_maps.npz",
                            outside_freq=freq.astype(np.float32),
                            above_freq=(env_above / max(n_days, 1)).astype(np.float32),
                            below_freq=(env_below / max(n_days, 1)).astype(np.float32),
                            rms_err=np.sqrt(err2 / max(n_days, 1)).astype(np.float32),
                            rms_spread=np.sqrt(spread2 / max(n_days, 1)).astype(np.float32),
                            land=land)
        ref = 2.0 / (members + 1)                       # 完美校准的期望出界率
        fig, ax = plt.subplots(figsize=(9.2, 4.9), constrained_layout=True)
        im = ax.imshow(np.where(land, freq, np.nan), cmap="magma",
                       vmin=0, vmax=max(float(np.nanpercentile(np.where(land, freq, np.nan), 99)), ref * 2),
                       origin="lower", interpolation="nearest")
        ax.set_facecolor("0.85"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"envelope-miss frequency ({n_days} days; calibrated ref = {ref:.3f})",
                     fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(figd / "outside_freq.png", dpi=140); plt.close(fig)

        ratio = np.sqrt(err2 / max(n_days, 1)) / np.maximum(np.sqrt(spread2 / max(n_days, 1)), 1e-9)
        lg = np.log2(np.maximum(ratio, 1e-6))
        fig, ax = plt.subplots(figsize=(9.2, 4.9), constrained_layout=True)
        im = ax.imshow(np.where(land, lg, np.nan), cmap="RdBu_r", vmin=-2, vmax=2,
                       origin="lower", interpolation="nearest")
        ax.set_facecolor("0.85"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("log2[ RMS(truth-ens_mean) / RMS(spread) ]  "
                     "(0 = calibrated; red = under-dispersed)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(figd / "ratio_map.png", dpi=140); plt.close(fig)

    print(f"[big] 合并 {n_days} 天 -> bigcheck_{target}.json + figs/")
    print(f"[big] overall: CRPS={agg['crps_ens']:.4f} μMAE={agg['mae_mu']:.4f} "
          f"CRPSS={agg['crpss']:+.3f} | rmse μ/ens={agg['rmse_mu']:.4f}/{agg['rmse_ens_mean']:.4f} "
          f"| spread={agg['spread_land_mean']:.4f}{unit}")


def main():
    ap = argparse.ArgumentParser(description="阶段 B 大检验")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--diffusion-ckpt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2018, 2019])
    ap.add_argument("--all-days", action="store_true",
                    help="取 years 内全部日(365 日历/年), 替代每月 5/15/25 的抽样")
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--overlap", type=int, required=True)
    ap.add_argument("--boundary", type=int, required=True)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-data", type=float, required=True)
    ap.add_argument("--model-channels", type=int, default=64)
    ap.add_argument("--box", type=int, default=384, help="功率谱用的全陆地方框边长")
    ap.add_argument("--ss-bins", type=int, default=20)
    ap.add_argument("--ss-cap", type=float, default=0.0, help="spread 分箱上限; 0=按目标默认")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ti = TD.TARGETS.index(a.target)
    unit = "mm/day" if a.target == TD.PRECIP else "K"
    out = Path(a.out)
    if a.merge_only:
        merge(out, a.target, unit, a.members)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    ss_cap = a.ss_cap if a.ss_cap > 0 else (25.0 if a.target == TD.PRECIP else 5.0)

    days = ([(y, t) for y in a.years for t in range(365)] if a.all_days
            else default_days(a.years))
    mine = days[rank::ntasks]
    cache = MuCache(a.cache, [a.target]); cache.verify({a.target: a.stage_a_ckpt})
    stats = TD.Stats(a.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, a.years, TD.DEFAULT_IN, TD.TARGETS, stats)
    H, W = d.H, d.W
    (out / "parts").mkdir(parents=True, exist_ok=True)
    print(f"[big] rank {rank}/{ntasks} device={device} 分到 {len(mine)} 天", flush=True)

    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_in_channels=41 + 100, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=a.model_channels,
        channel_mult=[1, 2, 2], attn_resolutions=[16],
        N_grid_channels=100, gridtype="learnable",
        sigma_data=a.sigma_data).to(device).eval()
    net.load_state_dict(torch.load(a.diffusion_ckpt, map_location=device)["model"])
    patching = GridPatching2D(img_shape=(H, W), patch_shape=(a.patch, a.patch),
                              overlap_pix=a.overlap, boundary_pix=a.boundary)

    lmask = d.mask[a.years[0]]
    by, bx, bs = TD.pick_land_box(lmask, a.box)       # 掩膜恒定 -> 各 rank/日同框
    rng_tie = np.random.default_rng(a.seed + 977 + rank)

    rank_counts = np.zeros(a.members + 1)
    psd_sum = None
    ss = np.zeros((3, a.ss_bins))                      # count / spread 和 / 平方误差和
    edges = np.linspace(0, ss_cap, a.ss_bins + 1)
    # 逐像素空间校准累计: 包络出界次数与幅度比值的分子分母
    env_above = np.zeros((H, W))
    env_below = np.zeros((H, W))
    err2_map = np.zeros((H, W))
    spread2_map = np.zeros((H, W))
    land_ref = None
    metrics = {}
    for y, day in mine:
        t0 = time.time()
        cond, tgt, mask, hr = d.full(y, day)
        land = mask[0] > 0.5
        truth = hr[ti].astype(np.float64)
        if a.target == TD.PRECIP:
            truth = truth * stats.precip_scale
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
        mu_t = pin_ocean(torch.from_numpy(
            cache.get(a.target, y, day)[None, None]).float().to(device), land_t)

        members_norm = []
        for m in range(a.members):
            torch.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
            lat = torch.randn(1, 1, H, W, device=device)
            with torch.no_grad():
                r = stochastic_sampler(
                    net=net, latents=lat, img_lr=cond_t, patching=patching,
                    mean_hr=mu_t, num_steps=a.steps, sigma_min=a.sigma_min,
                    sigma_max=a.sigma_max, rho=7, S_churn=0, S_noise=1)
            members_norm.append(mu_t[0, 0].cpu().numpy() + r[0, 0].float().cpu().numpy())
        members_norm = np.stack(members_norm, 0)

        mu_norm = mu_t[0, 0].cpu().numpy()
        mu_phys = mu_norm * stats.d_std[ti] + stats.d_mean[ti]
        if a.target == TD.PRECIP and stats.precip_log:
            mu_phys = TD.precip_inv(mu_phys, stats.precip_scale) * stats.precip_scale
            mu_phys = np.where(mu_phys < stats.precip_clip, 0.0, mu_phys)
            mem_phys = TD.precip_inv(members_norm * stats.d_std[ti] + stats.d_mean[ti],
                                     stats.precip_scale) * stats.precip_scale
            # 融合后 drizzle 截断(与预处理 clip 同口径): 恢复零质量, member 与
            # 被 clip 过的真值同一约定比较
            mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)
        else:
            mem_phys = members_norm * stats.d_std[ti] + stats.d_mean[ti]
        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)

        err = (ens_mean - truth)[land]
        met = {
            "year": y, "day": day, "month": month_of(y, day),
            "rmse_mu": float(np.sqrt(((mu_phys - truth)[land] ** 2).mean())),
            "mae_mu": float(np.abs((mu_phys - truth)[land]).mean()),
            "rmse_ens_mean": float(np.sqrt((err ** 2).mean())),
            "rmse_member_avg": float(np.mean([np.sqrt(((mem_phys[m] - truth)[land] ** 2).mean())
                                              for m in range(a.members)])),
            "crps_ens": TD.crps_ensemble(mem_phys[:, None], truth[None], land)[0],
            "spread_land_mean": float(spread[land].mean()),
            "seconds": round(time.time() - t0, 1),
        }
        if a.target == TD.PRECIP:
            logs = members_norm * stats.d_std[ti] + stats.d_mean[ti]
            met["clip_frac"] = float((logs > TD.PRECIP_LOG_MAX).mean())

        # rank histogram(逐像素): 平局按标准做法均匀劈分 —— 真值与 k 个 member 并列时
        # 名次在并列区间内均匀随机。降水截断口径下精确零平局占多数像素, 全有/全无式
        # 劈分会把平局质量灌进两端, 必须均匀劈
        mem_l = mem_phys[:, land]
        tr_l = truth[land]
        ties = (mem_l == tr_l[None]).sum(0)
        ranks = (mem_l < tr_l[None]).sum(0) + rng_tie.integers(0, ties + 1)
        rank_counts += np.bincount(np.clip(ranks, 0, a.members), minlength=a.members + 1)

        # 功率谱(固定框): 降水用 log1p(mm) 空间, 温度用物理空间
        if a.target == TD.PRECIP:
            f_truth = np.log1p(np.maximum(truth, 0.0))
            f_mu = np.log1p(np.maximum(mu_phys, 0.0))
            f_ens = np.log1p(np.maximum(ens_mean, 0.0))
            f_mem = np.log1p(np.maximum(mem_phys[0], 0.0))
        else:
            f_truth, f_mu, f_ens, f_mem = truth, mu_phys, ens_mean, mem_phys[0]
        cur = np.stack([TD.radial_psd(f[by:by + bs, bx:bx + bs], bs)
                        for f in (f_truth, f_mu, f_ens, f_mem)], 0)
        psd_sum = cur if psd_sum is None else psd_sum + cur

        # 逐像素空间累计(出界频率图与幅度比值图的原料)
        land_ref = land
        env_above += (truth > mem_phys.max(0)) & land
        env_below += (truth < mem_phys.min(0)) & land
        err2_map += np.where(land, (ens_mean - truth) ** 2, 0.0)
        spread2_map += np.where(land, spread.astype(np.float64) ** 2, 0.0)

        # spread-skill 分箱累积
        sp_l = np.clip(spread[land], 0, ss_cap - 1e-9)
        idx = np.digitize(sp_l, edges) - 1
        for b in range(a.ss_bins):
            sel = idx == b
            n = int(sel.sum())
            if n:
                ss[0, b] += n
                ss[1, b] += float(sp_l[sel].sum())
                ss[2, b] += float((err[sel] ** 2).sum())

        metrics[f"{y}-d{day}"] = met
        print(f"  [rank{rank}] {y}-d{day}: CRPS={met['crps_ens']:.4f} μMAE={met['mae_mu']:.4f} "
              f"spread={met['spread_land_mean']:.3f}{unit}  {met['seconds']:.0f}s", flush=True)

    json.dump(metrics, open(out / "parts" / f"big_part_rank{rank}.json", "w"),
              indent=1, ensure_ascii=False)
    np.savez_compressed(out / "parts" / f"big_part_rank{rank}.npz",
                        rank_counts=rank_counts, psd_sum=psd_sum, ss=ss,
                        env_above=env_above, env_below=env_below,
                        err2_map=err2_map, spread2_map=spread2_map,
                        n_days=np.array(len(metrics)), land=land_ref)
    print(f"[big] rank {rank} 完成 {len(metrics)} 天", flush=True)
    if ntasks == 1:
        merge(out, a.target, unit, a.members)


if __name__ == "__main__":
    main()
