#!/usr/bin/env python
# Packaged implementation; code/train_unet.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
train_unet.py — 确定性 UNet 回归 (ERA5 120x240 -> Daymet 720x1440, 6x)
============================================================================
独立训练 UNet。DDP 多节点 + 早停(val 连续 N 个 epoch 不提升就停, 默认 N=10)。
  - train = 1980-2017 (--train-years)   训练
  - val   = 2018-2019 (--val-years)     早停监控 (masked MSE)
  - test  = 2020      (--test-year)     最终评测(载入 best ckpt)
模型/数据管线/早停引擎/评测全部复用 train_downscale.py(它是核心库)。
UNet 是全卷积, 评测整帧一次出图(eval_tile=0), 不分块。

用法:
  自测(合成数据, 秒级, 需要 torch):
      python train_unet.py --smoke
  单卡:
      python train_unet.py --stats-dir stats_train --out runs/unet \
          --train-years $(seq 1980 2017) --val-years 2018 2019 --test-year 2020
  多节点(Frontier): 把下面这行放进 submit_ddp.sh 的 srun 后面
      srun -u python train_unet.py --stats-dir stats_train --out runs/unet
============================================================================
"""
import argparse
import os
import sys

from era5_daymet.training import train_downscale as TD


def main():
    p = argparse.ArgumentParser(description="独立 UNet 降尺度训练 (DDP + 早停)",
                                formatter_class=argparse.RawTextHelpFormatter)
    TD.add_common_args(p)                                  # 通用: 数据/年份/patch/lr/epochs/patience...
    p.add_argument("--base", type=int, default=64, help="UNet 通道基数(指南 U1/U2/U3 = 64/128/192)")
    # 整幅训练(与 train_vit.py 同名参数一致): UNet 全卷积, 感受野 ~54px << crop 192, 故整幅与裁块
    # 学到的函数几乎相同; 用整幅是为了和整幅 ViT 对比时消除"空间采样方式"这个混杂变量。
    p.add_argument("--full-frame", action="store_true",
                   help="整幅 720x1440 训练+整幅评测(不切窗); 用 FullFrameDS, eval_tile=0")
    p.add_argument("--epoch-frames", type=int, default=0,
                   help="整幅: 每 epoch 见多少帧; >0 时有效 steps=ceil(N/(world*batch)), 加节点真省墙钟")
    p.add_argument("--smoke", action="store_true", help="合成数据秒级自测")
    args = p.parse_args()

    if TD.torch is None:
        sys.exit("需要 PyTorch(在 Frontier GPU 节点上跑)。本地只能 --smoke 且需装 torch。")
    args.model = "unet"
    args.eval_tile = 0                                     # ★UNet 全卷积: 整帧评测, 不分块

    if args.smoke:
        TD.apply_smoke(args)                               # 覆写 dirs/years 为合成小数据
    if not args.full_frame:                                # 整幅不切窗, 无 crop 整除约束(720/1440 天然合法)
        TD.check_patch(args)                               # --patch 必须能被 FACTOR=6 整除

    rank, world, local, device, is_dist = TD.setup_ddp()
    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cin = TD.cond_channels(args.in_vars, args.out_vars, args.use_clim)  # bilinear(ERA5)+Δz+lc+lsm(+气候态 if --use-clim)
    Cout = len(args.out_vars)
    model = TD.UNet(Cin, Cout, base=args.base, temb=0).to(device)
    n_par = sum(x.numel() for x in model.parameters())
    if rank == 0:
        mode = f"full-frame(epoch_frames={args.epoch_frames})" if args.full_frame else f"crop{args.patch}"
        print(f"[UNet] {mode}  amp={'bf16' if args.amp else 'fp32'}  "
              f"base={args.base}  Cin={Cin} Cout={Cout}  params={n_par/1e6:.1f}M  "
              f"world={world}  train={args.train_years[0]}-{args.train_years[-1]}  "
              f"val={args.val_years}  test={args.test_year}  patience={args.patience}", flush=True)

    TD.fit_deterministic(model, stats, args, device, (rank, world, local, is_dist))
    if rank == 0:
        TD.evaluate(model, None, stats, args, device)      # diff=None -> 确定性评测

    if is_dist:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
