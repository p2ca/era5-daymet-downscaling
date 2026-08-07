#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_stage_b_overfit.py — 阶段 B 的过拟合验收
============================================================================
在固定的 64 个 patch 上训练残差扩散网, 要求损失持续下降。这是正式训练前的硬性验收:
它一次性串起缓存 μ、官方 ResidualLoss、官方 RandomPatching2D 与 EDM 预条件, 任何一处
接错都会表现为损失不降。

训练时噪声水平与噪声本身逐步随机, 单步损失噪声很大; 因此另取一组固定的 (sigma, 噪声)
在同一批 patch 上周期性评估, 得到可判读的曲线。
============================================================================
"""
import argparse

import numpy as np
import torch

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.mu_cache import MuCache
from era5_daymet.models.corrdiff_loss import ResidualLoss
from era5_daymet.models.patching import RandomPatching2D
from era5_daymet.models.preconditioning import EDMPrecondSuperResolution
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.stage_b_mean import CachedRegressionMean, pin_ocean


def build(target, img_shape, device, model_channels, p_mean, p_std, sigma_data):
    # img_resolution 取**全域**尺寸: 可学位置网格按全域建, 由 patching 提供的 global_index
    # 为每个 patch 取出对应的那一块。传 patch 尺寸会让索引越界(表现为 GPU 非法访存)。
    ti = TD.TARGETS.index(target)
    net = EDMPrecondSuperResolution(
        img_resolution=list(img_shape), img_in_channels=41 + 100, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=model_channels,
        channel_mult=[1, 2, 2], attn_resolutions=[16],
        N_grid_channels=100, gridtype="learnable").to(device)
    reg = CachedRegressionMean(out_channels=1).to(device)
    loss_fn = ResidualLoss(regression_net=reg, P_mean=p_mean, P_std=p_std,
                           sigma_data=sigma_data, hr_mean_conditioning=True)
    return ti, net, reg, loss_fn


def main():
    p = argparse.ArgumentParser(description="阶段 B 过拟合验收")
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True, help="该目标的阶段 A checkpoint")
    p.add_argument("--target", default="2m_temperature_max")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--days", type=int, nargs="+", default=[2018, 100, 2019, 200],
                   help="成对给出 年 日")
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--n-patches", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--p-mean", type=float, default=-1.2)
    p.add_argument("--p-std", type=float, default=1.2)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--min-land", type=float, default=0.10)
    p.add_argument("--per-step", type=int, default=8, help="每天每步取几个 patch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = MuCache(args.cache, [args.target])
    cache.verify({args.target: args.ckpt})
    stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    ti = TD.TARGETS.index(args.target)

    # ---- 取固定的若干天, 组成一个常驻 batch ----
    conds, tgts, mus, masks = [], [], [], []
    pairs = list(zip(args.days[0::2], args.days[1::2]))
    for yr, day in pairs:
        d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [yr], TD.DEFAULT_IN, TD.TARGETS, stats)
        cond, tgt, mask, _ = d.full(yr, day)
        conds.append(cond)
        tgts.append(tgt[ti:ti + 1])
        mus.append(cache.get(args.target, yr, day)[None])
        masks.append((mask[0] > 0.5).astype(np.float32)[None])
    cond = torch.from_numpy(np.stack(conds)).float().to(device)
    tgt = torch.from_numpy(np.stack(tgts)).float().to(device)
    mu = torch.from_numpy(np.stack(mus)).float().to(device)
    land = torch.from_numpy(np.stack(masks)).float().to(device)
    mu = pin_ocean(mu, land)                      # 海洋上的 μ 钉 0 后再构造残差
    H, W = cond.shape[-2:]
    _, net, reg, loss_fn = build(args.target, (H, W), device, args.model_channels,
                                 args.p_mean, args.p_std, args.sigma_data)
    print(f"[overfit] target={args.target} 扩散网参数={sum(q.numel() for q in net.parameters()):,} "
          f"device={device} 全域={H}x{W} patch={args.patch}", flush=True)
    print(f"[overfit] 常驻 batch: {len(pairs)} 天, cond{tuple(cond.shape)} "
          f"tgt{tuple(tgt.shape)} mu{tuple(mu.shape)}", flush=True)

    # ---- 固定 patch 位置, 且每块陆地占比不低于阈值 ----
    per_day = args.n_patches // len(pairs)
    patching = RandomPatching2D(img_shape=(H, W), patch_shape=(args.patch, args.patch),
                                patch_num=per_day)
    rng = np.random.default_rng(args.seed)
    fixed = []
    tries = 0
    while len(fixed) < per_day and tries < 10000:
        tries += 1
        i = int(rng.integers(0, H - args.patch + 1))
        j = int(rng.integers(0, W - args.patch + 1))
        if float(land[:, :, i:i + args.patch, j:j + args.patch].mean()) >= args.min_land:
            fixed.append((i, j))
    if len(fixed) < per_day:
        raise RuntimeError(f"只找到 {len(fixed)} 个满足陆地占比 {args.min_land} 的 patch")
    # 固定池子共 n_patches 个; 单卡放不下同批 64 个 192x192x141 的 patch, 因此每步
    # 轮转一个子集, 若干步覆盖整个池子。生产配置是每 GCD 1 个 patch, 不存在此限制。
    per_step = min(args.per_step, per_day)
    patching.set_patch_num(per_step)
    print(f"[overfit] 固定池 {len(fixed)} 个位置/天 x {len(pairs)} 天 = "
          f"{len(fixed)*len(pairs)} 个 patch; 每步取 {per_step}/天 轮转", flush=True)

    def rotate(k):
        o = (k * per_step) % len(fixed)
        patching.patch_indices = [fixed[(o + i) % len(fixed)] for i in range(per_step)]

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    def eval_fixed():
        """固定 (sigma, 噪声) 下的损失, 用于判读趋势。"""
        net.eval()
        g = torch.Generator(device="cpu").manual_seed(1234)
        tot = 0.0
        with torch.no_grad():
            for k in range(len(fixed) // per_step):
                torch.manual_seed(1234 + k); rotate(k)
                reg.set(mu)
                l = loss_fn(net=net, img_clean=tgt, img_lr=cond, patching=patching,
                            use_patch_grad_acc=False)
                tot += float(l.mean())
        net.train()
        return tot / max(1, len(fixed) // per_step)

    hist = []
    base = eval_fixed()
    print(f"[overfit] step    0  固定评估 {base:.5f}", flush=True)
    for step in range(1, args.steps + 1):
        rotate(step)
        reg.set(mu)
        loss = loss_fn(net=net, img_clean=tgt, img_lr=cond, patching=patching,
                       use_patch_grad_acc=False)
        l = loss.mean()
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1e6)
        opt.step()
        if step % 50 == 0:
            e = eval_fixed()
            hist.append((step, e))
            print(f"[overfit] step {step:4d}  固定评估 {e:.5f}  "
                  f"(相对起点 {e/base:.3f}x)  单步 {float(l.detach()):.5f}", flush=True)

    ok = hist and hist[-1][1] < base * 0.7 and all(np.isfinite([h[1] for h in hist]))
    print(f"\n起点 {base:.5f} -> 终点 {hist[-1][1]:.5f}  "
          f"降幅 {(1-hist[-1][1]/base)*100:.1f}%")
    print("判定:", "PASS — 损失持续下降且无 NaN" if ok else "FAIL — 未达到预期下降")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
