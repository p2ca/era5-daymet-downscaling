#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_ensemble_check.py — 阶段 B 权重的集合检验(验证集)
============================================================================
对若干验证日 x 若干 (overlap, boundary) 融合配置, 各采样 N 个 member(官方 EDM 二阶
随机采样器 + GridPatching2D 加权融合), 在物理单位、仅陆地口径下输出:

  * μ-only(阶段 A 均值)的 RMSE/MAE —— 确定性预报的 CRPS 恰等于其 MAE;
  * ensemble mean 的 RMSE/MAE 与 ensemble CRPS;
  * 逐 member 的 RMSE/MAE 明细与 member spread。

member 潜变量种子只依赖 (年, 日, member), 与融合配置无关 —— 不同配置间成对可比,
差异纯粹来自切块与融合方式, 服务于 overlap/boundary 的选型(seam 最低且
CRPS/均值误差不恶化)。

多卡: 在 SLURM 多任务下按 (配置, 日) 拆分工作项, 以配置的 patch 数为权重做负载
均衡; 各 rank 独立写 parts/ 分片, 结束后用 --merge 汇总(无进程间通信)。
============================================================================
"""
import argparse
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


def to_phys(x_norm, stats, ti, target):
    """归一化目标 -> 物理单位。温度: z-score 逆变换(K); 降水: 逆 z-score 后
    log1p(mm) -> mm/day, 且融合后按 drizzle 阈值(与预处理 clip 同口径)把
    < clip 的毛毛雨截回精确 0 —— 恢复被融合平均破坏的零质量, 使 member 与
    被 clip 过的真值同一约定比较。"""
    x = x_norm * stats.d_std[ti] + stats.d_mean[ti]
    if target == TD.PRECIP and stats.precip_log:
        x = TD.precip_inv(x, stats.precip_scale) * stats.precip_scale
        x = np.where(x < stats.precip_clip, 0.0, x)
    return x


def masked_rmse(a, b, land):
    d = (a - b)[land]
    return float(np.sqrt((d * d).mean()))


def masked_mae(a, b, land):
    return float(np.abs(a - b)[land].mean())


def parse_days(tokens):
    """'YYYY-DDD'(0 基日序) -> [(year, day), ...]"""
    out = []
    for tk in tokens:
        y, d = tk.split("-")
        out.append((int(y), int(d)))
    return out


def build_combos(overlaps, boundaries, days):
    """工作项 = (overlap, boundary, year, day); 权重 = 该配置的 patch 数。"""
    combos = []
    for ov, bd in zip(overlaps, boundaries):
        for y, d in days:
            combos.append((ov, bd, y, d))
    return combos


def assign_rank(combos, ntasks, patch, H, W):
    """按 patch 数(采样成本)做最长作业优先的静态均衡, 返回 {rank: [combo,...]}。
    分配只依赖参数, 各 rank 独立计算得到一致结果。"""
    def weight(c):
        ov, bd = c[0], c[1]
        import math
        nx = math.ceil(W / (patch - ov - bd)); ny = math.ceil(H / (patch - ov - bd))
        return nx * ny
    order = sorted(combos, key=weight, reverse=True)
    loads = [0.0] * ntasks
    out = {r: [] for r in range(ntasks)}
    for c in order:
        r = int(np.argmin(loads))
        loads[r] += weight(c)
        out[r].append(c)
    return out


def merge(out_dir, target, unit, ref_cfg="ov48_bd2"):
    """汇总 parts/ 分片: 逐 (配置, 日) 明细 + 各配置对日平均, 并给出相对参考配置的差。"""
    entries = {}
    for fp in sorted(glob.glob(str(Path(out_dir) / "parts" / "ens_part_rank*.json"))):
        entries.update(json.load(open(fp)))
    cfgs = sorted({k.split("|")[0] for k in entries})
    agg = {}
    for cfg in cfgs:
        es = [v for k, v in entries.items() if k.startswith(cfg + "|")]
        agg[cfg] = {m: float(np.mean([e[m] for e in es]))
                    for m in ("crps_ens", "mae_mu", "rmse_mu", "rmse_ens_mean",
                              "mae_ens_mean", "rmse_member_avg", "spread_land_mean")}
        agg[cfg]["n_days"] = len(es)
    ref = agg.get(ref_cfg)
    if ref:
        for cfg in cfgs:
            agg[cfg]["crps_vs_ref"] = agg[cfg]["crps_ens"] - ref["crps_ens"]
            agg[cfg]["rmse_ens_vs_ref"] = agg[cfg]["rmse_ens_mean"] - ref["rmse_ens_mean"]
    payload = {"target": target, "unit": unit, "per_config_day": entries,
               "per_config_mean": agg, "reference_config": ref_cfg}
    outp = Path(out_dir) / f"ens_metrics32_{target}.json"
    json.dump(payload, open(outp, "w"), indent=1, ensure_ascii=False)
    print(f"[ens] 合并 {len(entries)} 个 (配置,日) 条目 -> {outp}")
    for cfg in cfgs:
        a = agg[cfg]
        print(f"  {cfg}: CRPS={a['crps_ens']:.4f}  μMAE={a['mae_mu']:.4f}  "
              f"rmse μ/ens={a['rmse_mu']:.4f}/{a['rmse_ens_mean']:.4f}  "
              f"spread={a['spread_land_mean']:.4f}{unit}  (days={a['n_days']})")
    return outp


def main():
    p = argparse.ArgumentParser(description="阶段 B 集合检验")
    p.add_argument("--cache", required=True)
    p.add_argument("--stage-a-ckpt", required=True, help="用于校验 μ 缓存绑定")
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--target", default="2m_temperature_max")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--days", nargs="+", default=["2018-15", "2018-105", "2018-200", "2018-288",
                                                 "2019-15", "2019-105", "2019-200", "2019-288"],
                   help="'YYYY-DDD' 0 基日序")
    p.add_argument("--members", type=int, default=32)
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--overlaps", type=int, nargs="+", default=[48, 96, 96])
    p.add_argument("--boundaries", type=int, nargs="+", default=[2, 2, 8])
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-data", type=float, required=True)
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figs", type=int, default=0, help="1=每个 (配置,日) 出六联图")
    p.add_argument("--merge-only", action="store_true", help="只汇总 parts/, 不采样")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ti = TD.TARGETS.index(args.target)
    unit = "mm/day" if args.target == TD.PRECIP else "K"
    out = Path(args.out)
    if args.merge_only:
        merge(out, args.target, unit)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    days = parse_days(args.days)
    years = sorted({y for y, _ in days})
    combos = build_combos(args.overlaps, args.boundaries, days)

    cache = MuCache(args.cache, [args.target])
    cache.verify({args.target: args.stage_a_ckpt})
    stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, years, TD.DEFAULT_IN, TD.TARGETS, stats)
    H, W = d.H, d.W

    mine = assign_rank(combos, ntasks, args.patch, H, W)[rank]
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    print(f"[ens] rank {rank}/{ntasks} device={device} 分到 {len(mine)} 个工作项: "
          f"{[(f'ov{o}bd{b}', f'{y}-d{dd}') for o, b, y, dd in mine]}", flush=True)

    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_in_channels=41 + 100, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=args.model_channels,
        channel_mult=[1, 2, 2], attn_resolutions=[16],
        N_grid_channels=100, gridtype="learnable",
        sigma_data=args.sigma_data).to(device).eval()
    net.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device)["model"])

    day_cache = {}
    results = {}
    for ov, bd, y, day in mine:
        t0 = time.time()
        if (y, day) not in day_cache:
            cond, tgt, mask, hr = d.full(y, day)
            land = mask[0] > 0.5
            truth = hr[ti].astype(np.float64)
            if args.target == TD.PRECIP:
                truth = truth * stats.precip_scale        # 真值 m/day -> mm/day
            truth_norm = tgt[ti].astype(np.float32)       # 归一化空间真值(扩散工作空间)
            cond_t = torch.from_numpy(cond[None]).float().to(device)
            land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
            mu_t = pin_ocean(torch.from_numpy(
                cache.get(args.target, y, day)[None, None]).float().to(device), land_t)
            day_cache = {(y, day): (cond_t, land, truth, truth_norm, mu_t)}  # 只留当前日
        cond_t, land, truth, truth_norm, mu_t = day_cache[(y, day)]

        patching = GridPatching2D(img_shape=(H, W), patch_shape=(args.patch, args.patch),
                                  overlap_pix=ov, boundary_pix=bd)
        members_norm = []
        for m in range(args.members):
            # 种子只含 (年,日,member): 配置间共享潜变量, 成对可比
            torch.manual_seed(args.seed * 100003 + (y * 1000 + day) * 131 + m)
            lat = torch.randn(1, 1, H, W, device=device)
            with torch.no_grad():
                r = stochastic_sampler(
                    net=net, latents=lat, img_lr=cond_t, patching=patching,
                    mean_hr=mu_t, num_steps=args.steps, sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max, rho=7, S_churn=0, S_noise=1)
            members_norm.append(mu_t[0, 0].cpu().numpy() + r[0, 0].float().cpu().numpy())
        members_norm = np.stack(members_norm, 0)

        mu_phys = to_phys(mu_t[0, 0].cpu().numpy(), stats, ti, args.target)
        mem_phys = to_phys(members_norm, stats, ti, args.target)
        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)
        r_one = mem_phys[0] - mu_phys

        rmse_members = [masked_rmse(mem_phys[m], truth, land) for m in range(args.members)]
        mae_members = [masked_mae(mem_phys[m], truth, land) for m in range(args.members)]
        met = {
            "config": f"ov{ov}_bd{bd}", "year": y, "day": day,
            "rmse_mu": masked_rmse(mu_phys, truth, land),
            "mae_mu": masked_mae(mu_phys, truth, land),
            "rmse_ens_mean": masked_rmse(ens_mean, truth, land),
            "mae_ens_mean": masked_mae(ens_mean, truth, land),
            "rmse_member_avg": float(np.mean(rmse_members)),
            "rmse_members": [round(v, 5) for v in rmse_members],
            "mae_members": [round(v, 5) for v in mae_members],
            "crps_ens": TD.crps_ensemble(mem_phys[:, None], truth[None], land)[0],
            "spread_land_mean": float(spread[land].mean()),
            "residual_std_land": float(r_one[land].std()),
            "ens_shift_from_mu": masked_mae(ens_mean, mu_phys, land),
            "n_members": args.members,
            "seconds": round(time.time() - t0, 1),
        }
        key = f"ov{ov}_bd{bd}|{y}-d{day}"
        results[key] = met
        print(f"  [rank{rank}] {key}: CRPS={met['crps_ens']:.4f} vs μMAE={met['mae_mu']:.4f} | "
              f"rmse μ/ens={met['rmse_mu']:.3f}/{met['rmse_ens_mean']:.3f} | "
              f"spread={met['spread_land_mean']:.3f}{unit}  {met['seconds']:.0f}s", flush=True)

        fields = dict(truth=truth.astype(np.float32), mu=mu_phys.astype(np.float32),
                      ens_mean=ens_mean.astype(np.float32), spread=spread.astype(np.float32),
                      r_one=r_one.astype(np.float32), land=land)
        if args.target == TD.PRECIP:
            # 降水补归一化(log)空间字段与湿区量: 物理场经 expm1 不可逆推, 须单独落盘
            mu_norm = mu_t[0, 0].cpu().numpy().astype(np.float32)
            fields.update(truth_norm=truth_norm, mu_norm=mu_norm,
                          member_norm=members_norm[0].astype(np.float32),
                          r_one_norm=(members_norm[0] - mu_norm).astype(np.float32),
                          wet_prob=(mem_phys >= 0.1).mean(0).astype(np.float32))
            # expm1 钳制比例(红旗诊断): 反归一化后 log 值超上界的像素占比
            logs = members_norm * stats.d_std[ti] + stats.d_mean[ti]
            met["clip_frac"] = float((logs > TD.PRECIP_LOG_MAX).mean())
        np.savez_compressed(out / "fields" / f"ens_fields_{args.target}_{y}_d{day}_ov{ov}bd{bd}.npz",
                            **fields)

        if args.figs:
            off = 273.15 if unit == "K" else 0.0
            show = lambda a: np.where(land, a - off, np.nan)
            vmin, vmax = np.nanpercentile(show(truth), 1), np.nanpercentile(show(truth), 99)
            emax = max(1e-6, np.nanpercentile(np.abs(np.where(land, ens_mean - truth, np.nan)), 99))
            rmax = max(1e-6, np.nanpercentile(np.abs(np.where(land, r_one, np.nan)), 99))
            fig, ax = plt.subplots(2, 3, figsize=(19, 7.6), constrained_layout=True)
            panels = [
                (show(truth), f"truth (Daymet) [{'degC' if off else unit}]", "viridis", vmin, vmax),
                (show(mu_phys), "mu = stage-A regression mean", "viridis", vmin, vmax),
                (show(ens_mean), f"ensemble mean ({args.members} members)", "viridis", vmin, vmax),
                (np.where(land, r_one, np.nan), f"one-member residual r [{unit}]", "RdBu_r", -rmax, rmax),
                (np.where(land, ens_mean - truth, np.nan), f"ens-mean minus truth [{unit}]", "RdBu_r", -emax, emax),
                (np.where(land, spread, np.nan), f"member spread (std) [{unit}]", "magma", 0,
                 max(1e-6, np.nanpercentile(np.where(land, spread, np.nan), 99))),
            ]
            stride = args.patch - ov - bd
            for k, (fld, title, cmap, lo, hi) in enumerate(panels):
                a = ax[k // 3][k % 3]
                im = a.imshow(fld, cmap=cmap, vmin=lo, vmax=hi, origin="lower",
                              interpolation="nearest")
                a.set_facecolor("0.85")
                if k == 3:
                    for yy in range(stride, H, stride):
                        a.axhline(yy, color="k", lw=0.3, alpha=0.35)
                    for xx in range(stride, W, stride):
                        a.axvline(xx, color="k", lw=0.3, alpha=0.35)
                a.set_title(title, fontsize=10)
                a.set_xticks([]); a.set_yticks([])
                fig.colorbar(im, ax=a, shrink=0.8)
            fig.suptitle(f"{args.target}  {y} day {day}  ov{ov}/bd{bd}  "
                         f"(stage-B: {Path(args.diffusion_ckpt).parent.name})", fontsize=11)
            fig.savefig(out / f"ens_{args.target}_{y}_d{day}_ov{ov}bd{bd}.png", dpi=130)
            plt.close(fig)

    fp = out / "parts" / f"ens_part_rank{rank}.json"
    json.dump(results, open(fp, "w"), indent=1, ensure_ascii=False)
    print(f"[ens] rank {rank} 完成 {len(results)} 项 -> {fp}", flush=True)
    if ntasks == 1:
        merge(out, args.target, unit)


if __name__ == "__main__":
    main()
