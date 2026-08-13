#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_overfit.py — JiT 整幅条件扩散的过拟合验收
============================================================================
在少量固定整幅帧上训练小号 JiT, 要求固定 (t, 噪声) 的评估损失持续下降。这是正式
训练前的硬性验收: 一次性串起数据装配、海洋置零、v 空间损失、bf16 前向与主干本身,
任何一处接错都会表现为损失不降。结束后从首帧条件采样一次, 报告陆地 RMSE 供目检。

Run (login GPU):
    python -m era5_daymet.tests.test_jit_overfit --stats-dir runs/stats/train_dayofyear
============================================================================
"""
import argparse
import time

import torch

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.models.jit_backbone import JiT
from era5_daymet.models.jit_sampler import generate
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.train_jit import jit_vloss


def main():
    p = argparse.ArgumentParser(description="JiT 过拟合验收")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--target", default="2m_temperature_max")
    p.add_argument("--days", type=int, nargs="+",
                   default=[2018, 100, 2018, 300, 2019, 50, 2019, 200],
                   help="成对给出 年 日")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--noise-scale", type=float, default=4.0)
    p.add_argument("--moe", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    days = list(zip(args.days[0::2], args.days[1::2]))
    years = sorted({y for y, _ in days})
    stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    d = TD.DownscaleData(args.era5_dir, args.daymet_dir, years,
                         TD.DEFAULT_IN, TD.TARGETS, stats)
    ti = TD.TARGETS.index(args.target)

    conds, tgts, lands = [], [], []
    for y, t in days:
        cond, tgt, mask, _ = d.full(y, t)
        conds.append(torch.from_numpy(cond))
        tgts.append(torch.from_numpy(tgt[ti:ti + 1]))
        lands.append(torch.from_numpy((mask[0] > 0.5).astype("float32"))[None])
    cond = torch.stack(conds).to(device)
    land = torch.stack(lands).to(device)
    tgt = torch.stack(tgts).to(device) * land            # 海洋无监督, 目标定义为 0

    moe_config = None
    if args.moe:
        moe_config = {"num_experts": 16, "moe_intermediate_size": 2 * args.hidden,
                      "num_experts_per_tok": 2, "n_group": 2, "topk_group": 2,
                      "routed_scaling_factor": 2.5, "interleave": True,
                      "use_shared_expert": True, "proj_drop": 0.0}
    net = JiT(hw=(d.H, d.W), patch=args.patch, cond_ch=cond.shape[1], out_ch=1,
              hidden=args.hidden, depth=args.depth, num_heads=args.heads,
              bottleneck=args.bottleneck, moe_config=moe_config).to(device)
    pc = net.param_counts()
    print(f"[overfit] 帧数 {len(days)} 参数 {pc['total']:,} (激活 {pc['activated']:,}) "
          f"device={device}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def fixed_eval():
        g = torch.Generator(device=device); g.manual_seed(12345)
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device != "cpu")):
            v = float(jit_vloss(net, tgt, cond, land, args.noise_scale,
                                -0.8, 0.8, 0.05, generator=g))
        net.train()
        return v

    first = fixed_eval()
    print(f"[overfit] step 0  fixed_eval={first:.5f}", flush=True)
    t0, hist = time.time(), []
    for step in range(1, args.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device != "cpu")):
            loss = jit_vloss(net, tgt, cond, land, args.noise_scale, -0.8, 0.8, 0.05)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            v = fixed_eval()
            hist.append(v)
            print(f"[overfit] step {step:4d}  train={float(loss.detach()):.5f}  "
                  f"fixed_eval={v:.5f}  {time.time()-t0:.0f}s", flush=True)
    final = hist[-1]
    assert final < 0.5 * first, f"过拟合失败: {first:.5f} -> {final:.5f} 降幅不足一半"

    net.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device != "cpu")):
        s = generate(net, cond[:1], steps=50, noise_scale=args.noise_scale,
                     land=land[:1], generator=torch.Generator(device).manual_seed(0))
    rmse = float((((s - tgt[:1]) ** 2 * land[:1]).sum() / land[:1].sum()).sqrt())
    print(f"[overfit] 首帧采样陆地 RMSE(z 空间)={rmse:.4f} "
          f"(过拟合场景应接近 0)", flush=True)
    print(f"test_jit_overfit: 通过 ({first:.5f} -> {final:.5f})")


if __name__ == "__main__":
    main()
