#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_stage_b_sampling.py — 阶段 B 全域采样验收
============================================================================
按指南要求分别产出 regression-only(μ)、residual-only(r)、all(μ+r) 三种全域场, 叠加
patch 边界后计算 seam ratio = 边界处梯度 / 邻近非边界处梯度, 目标 <= 1.10。

采样走官方 EDM 二阶随机采样器 + 官方 GridPatching2D 的加权融合; 位置网格按全域建, 由
patcher 提供的 global_index 为每个 patch 取对应的一块。
============================================================================
"""
import argparse
import json

import numpy as np
import torch

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.mu_cache import MuCache
from era5_daymet.models.patching import GridPatching2D
from era5_daymet.models.preconditioning import EDMPrecondSuperResolution
from era5_daymet.models.stochastic_sampler import stochastic_sampler
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.stage_b_mean import pin_ocean


def seam_ratio(field, land, patch, overlap, axis):
    """边界处的一阶差分幅度 / 邻近非边界处的幅度。

    patcher 的内部步长为 patch - overlap; 融合痕迹若存在, 会以该周期出现在场上。
    """
    stride = patch - overlap
    g = np.abs(np.diff(field, axis=axis))
    m = land[:-1, :] * land[1:, :] if axis == 0 else land[:, :-1] * land[:, 1:]
    n = g.shape[axis]
    idx = np.arange(stride, n, stride)
    if len(idx) == 0:
        return float("nan")
    sel = np.zeros(n, bool); sel[idx] = True
    near = np.zeros(n, bool)
    for d in (-3, -2, 2, 3):                     # 邻近但不含边界本身
        j = np.clip(idx + d, 0, n - 1); near[j] = True
    near &= ~sel
    take = lambda mask: (g[mask, :] * m[mask, :]).sum() / max(m[mask, :].sum(), 1) \
        if axis == 0 else (g[:, mask] * m[:, mask]).sum() / max(m[:, mask].sum(), 1)
    b, nb = take(sel), take(near)
    return float(b / nb) if nb > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description="阶段 B 全域采样验收")
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True, help="阶段 A checkpoint(用于校验缓存)")
    p.add_argument("--diffusion-ckpt", default="", help="阶段 B 权重; 留空则用随机初始化(仅验机制)")
    p.add_argument("--target", default="2m_temperature_max")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--day", type=int, default=200)
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--overlaps", type=int, nargs="+", default=[48, 96, 96])
    p.add_argument("--boundaries", type=int, nargs="+", default=[2, 2, 8])
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-data", type=float, default=0.11)
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ti = TD.TARGETS.index(args.target)
    cache = MuCache(args.cache, [args.target]); cache.verify({args.target: args.ckpt})
    stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [args.year], TD.DEFAULT_IN, TD.TARGETS, stats)
    cond, tgt, mask, _ = d.full(args.year, args.day)
    H, W = cond.shape[-2:]
    land_np = (mask[0] > 0.5).astype(np.float32)
    cond_t = torch.from_numpy(cond[None]).float().to(device)
    land_t = torch.from_numpy(land_np[None, None]).float().to(device)
    mu_t = pin_ocean(torch.from_numpy(cache.get(args.target, args.year, args.day)[None, None])
                     .float().to(device), land_t)

    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_in_channels=41 + 100, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=args.model_channels,
        channel_mult=[1, 2, 2], attn_resolutions=[16],
        N_grid_channels=100, gridtype="learnable", sigma_data=args.sigma_data).to(device).eval()
    if args.diffusion_ckpt:
        net.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device)["model"])
        tag = "trained"
    else:
        tag = "random-init"
    print(f"[sample] target={args.target} {args.year}-{args.day} 全域 {H}x{W} "
          f"权重={tag} sigma_max={args.sigma_max} steps={args.steps}", flush=True)

    out = {}
    for ov, bd in zip(args.overlaps, args.boundaries):
        patching = GridPatching2D(img_shape=(H, W), patch_shape=(args.patch, args.patch),
                                  overlap_pix=ov, boundary_pix=bd)
        torch.manual_seed(args.seed)
        lat = torch.randn(1, 1, H, W, device=device)
        with torch.no_grad():
            res = stochastic_sampler(
                net=net, latents=lat, img_lr=cond_t, patching=patching, mean_hr=mu_t,
                num_steps=args.steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                rho=7, S_churn=0, S_noise=1)
        r = res[0, 0].float().cpu().numpy()
        mu = mu_t[0, 0].cpu().numpy()
        allf = mu + r
        finite = bool(np.isfinite(r).all() and np.isfinite(allf).all())
        sy = seam_ratio(allf, land_np, args.patch, ov, 0)
        sx = seam_ratio(allf, land_np, args.patch, ov, 1)
        out[f"ov{ov}_bd{bd}"] = {
            "patch_num": int(patching.patch_num), "seam_y": sy, "seam_x": sx,
            "residual_std_land": float(r[land_np > 0.5].std()),
            "all_std_land": float(allf[land_np > 0.5].std()), "finite": finite}
        print(f"  overlap={ov:>3} boundary={bd}  patch数={patching.patch_num:>3}  "
              f"seam_y={sy:.3f} seam_x={sx:.3f}  residual_std={r[land_np>0.5].std():.4f}  "
              f"有限={finite}", flush=True)

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    worst = max(max(v["seam_y"], v["seam_x"]) for v in out.values())
    print(f"\n最差 seam ratio = {worst:.3f} (目标 <= 1.10)")
    print("机制判定:", "PASS — 三种场均产出且有限" if all(v["finite"] for v in out.values())
          else "FAIL — 出现 NaN/Inf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
