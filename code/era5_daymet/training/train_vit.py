#!/usr/bin/env python
# Packaged implementation; code/train_vit.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
train_vit.py — Vision-Transformer (SETR 风格) 降尺度 (ERA5 -> Daymet, 6x)
============================================================================
独立训练 ViT。和 UNet 分开, 因为 ViT 慢、要单独调超参。DDP + 早停(默认 10 epoch)。
  train=1980-2017 / val=2018-2019(早停) / test=2020(载入 best ckpt 评测)

★可调超参(都在命令行):
  --vit-patch  patch 边长   (默认 2; 越小细节越多、token 越多越慢)
  --dim        embedding 维 (默认 384)
  --depth      Transformer block 数 (默认 8)
  --heads      注意力头数   (默认 6)
  --mlp        MLP 扩张倍数 (默认 4.0)
  --patch      训练裁剪块边长(=eval 分块边长, 默认 64 -> token 数 (64/2)^2=1024)

结构: Conv patch-embed -> +可学习位置编码 -> N×(LN+MHSA+MLP) -> ConvTranspose 还原
      -> ★卷积细化头(两层 conv)消除 "一格一格" 的 patch 拼接伪影。
评测: ViT 的位置编码绑定训练 patch 尺寸, 所以整帧按 --patch 大小分块、重叠平均
      (det_predict, eval_tile=--patch), 保证每块 token 数与训练一致。

用法:
  自测(合成数据, 需 torch):  python train_vit.py --smoke
  单卡:  python train_vit.py --stats-dir stats_train --out runs/vit \
             --vit-patch 2 --dim 384 --depth 8 --heads 6 --patch 64
  多节点: 把  srun -u python train_vit.py --stats-dir stats_train --out runs/vit
          放进 submit_ddp.sh
============================================================================
"""
import argparse
import os
import sys

from era5_daymet.training import train_downscale as TD

if TD.torch is not None:
    import torch
    import torch.nn as nn

    # 网络主体定义在 era5_daymet.models.vit; 此处重新导出, 使 `TV.ViT` 等既有引用保持可用。
    # torch 缺失时本模块仍可导入(部分工具只用到这里的参数解析)。
    from era5_daymet.models.vit import (  # noqa: F401
        DropPath,
        TBlock,
        ViT,
        get_2d_sincos_pos_embed,
    )


def main():
    p = argparse.ArgumentParser(description="独立 ViT 降尺度训练 (DDP + 早停)",
                                formatter_class=argparse.RawTextHelpFormatter)
    TD.add_common_args(p)
    # 默认对齐指南 V3_LARGE 的参数预算(~15.2M), 做与 U-Net U3_BASE192(14.2M)公平的同参数量对比。
    # V3 名义超参 = patch2/dim384/depth6/heads12/mlp4(=>15.182M in 指南实现)。但本实现的上采样头更轻
    # (PixelShuffle+卷积细化头仅~1M, 该层不计入规模阶梯定义), depth=6 只有 11.6M;
    # 故 depth 取 8 补回头部差额 -> 15.19M ≈ 指南 V3。其余(dim/heads/patch/mlp)严格取 V3 值。
    p.add_argument("--vit-patch", type=int, default=2, help="patch 边长(V3=2)")
    p.add_argument("--dim", type=int, default=384, help="embedding 维度(V3=384)")
    p.add_argument("--depth", type=int, default=8, help="Transformer block 数(V3名义=6; 取8对齐V3参数预算~15.2M)")
    p.add_argument("--heads", type=int, default=12, help="注意力头数(V3=12)")
    p.add_argument("--mlp", type=float, default=4.0, help="MLP 扩张倍数(V3=4)")
    p.add_argument("--dropout", type=float, default=0.0, help="注意力/MLP/pos dropout(抗过拟合; 默认 0=旧口径)")
    p.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth, 沿深度线性 0->此值")
    p.add_argument("--pos-type", choices=["sincos", "learned"], default="sincos",
                   help="位置编码: sincos(固定, 任意尺寸, 整幅必需) / learned(绑 crop, 旧口径)")
    p.add_argument("--head-up", choices=["pixelshuffle", "convt"], default="pixelshuffle",
                   help="上采样头: pixelshuffle(无棋盘) / convt(旧口径 ConvTranspose)")
    p.add_argument("--full-frame", action="store_true",
                   help="整幅 720x1440 训练+整幅评测(不切窗); 用 FullFrameDS, eval_tile=0")
    p.add_argument("--epoch-frames", type=int, default=0,
                   help="整幅: 每 epoch 见多少帧; >0 时有效 steps=ceil(N/(world*batch)), 加节点真省墙钟")
    p.add_argument("--seq-parallel", action="store_true",
                   help="序列并行: SP 组内多 GCD 协同算一帧的全局注意力, 单步~1/sp-size(整幅提步数用)")
    p.add_argument("--sp-size", type=int, default=8,
                   help="SP 组大小(节点内 GCD 数, 建议=每节点GCD数=8); world 须被其整除")
    p.add_argument("--smoke", action="store_true", help="合成数据秒级自测")
    # ViT 专属默认: crop 小一点(token 数可控) + 梯度裁剪(Transformer 需要)
    # patch=60 而非 64: 必须被 FACTOR=6 整除(见 TD.check_patch), 64 会在真数据上崩
    # lr=1e-4 + grad_clip=1.0: 3e-4 无裁剪时发散(runs/exp/20260712-vit-d384, val 随 lr 单调变差)
    # LR 策略: 无 warmup, plateau 减半(见 TD.fit_deterministic; --lr-patience/--lr-factor/--min-lr)
    p.set_defaults(patch=60, batch=8, lr=1e-4, grad_clip=1.0)
    args = p.parse_args()

    if TD.torch is None:
        sys.exit("需要 PyTorch(在 Frontier GPU 节点上跑)。本地只能 --smoke 且需装 torch。")
    args.model = "vit"
    if args.full_frame and args.pos_type != "sincos":
        sys.exit("--full-frame 需 --pos-type sincos(learned 位置编码绑 crop 尺寸, 无法整幅)。")
    # 整幅: 整幅一次前向评测(不切窗); crop: 按训练 patch 尺寸分块重叠评测
    args.eval_tile = 0 if args.full_frame else args.patch

    if args.smoke:
        TD.apply_smoke(args)
        args.patch = 24; args.vit_patch = 2; args.dim = 64; args.depth = 2; args.heads = 4
        args.eval_tile = 0 if args.full_frame else 24

    if not args.full_frame:                                # 整幅不切窗, 无 crop 整除约束(720/1440 天然合法)
        if args.patch % args.vit_patch:
            sys.exit(f"--patch({args.patch}) 必须能被 --vit-patch({args.vit_patch}) 整除")
        TD.check_patch(args)                               # --patch 还必须能被 FACTOR=6 整除

    rank, world, local, device, is_dist = TD.setup_ddp()
    if args.seq_parallel and not args.full_frame:
        sys.exit("--seq-parallel 仅用于 --full-frame(切分整幅 token 的全局注意力)。")
    # 2D mesh(SP × DP): SP 组=节点内连续 GCD 协同算一帧; DP 组跨节点(不同帧)
    sp_ctx = None; sp_group = None; sp_size = 1; sp_rank = 0
    if args.seq_parallel and is_dist:
        from era5_daymet.models import seq_parallel_attn as SP
        if world % args.sp_size:
            sys.exit(f"world({world}) 必须被 --sp-size({args.sp_size}) 整除")
        sp_group, dp_group, sp_rank, dp_rank, dp_size = SP.build_2d_mesh(args.sp_size)
        sp_size = args.sp_size
        sp_ctx = dict(sp_group=sp_group, dp_group=dp_group, sp_rank=sp_rank, dp_rank=dp_rank, dp_size=dp_size)
        if rank == 0:
            print(f"[SP] 序列并行: sp_size={sp_size} dp_size={dp_size} (world={world}); "
                  f"单步~1/{sp_size}, 全局 batch={dp_size*args.batch}", flush=True)
    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cin = TD.cond_channels(args.in_vars, args.out_vars, args.use_clim)   # 默认20通道; --use-clim=23
    Cout = len(args.out_vars)
    model = ViT(Cin, Cout, img=args.patch, patch=args.vit_patch,
                dim=args.dim, depth=args.depth, heads=args.heads, mlp=args.mlp,
                dropout=args.dropout, drop_path=args.drop_path,
                pos_type=args.pos_type, head_up=args.head_up, full_frame=args.full_frame,
                sp_group=sp_group, sp_size=sp_size, sp_rank=sp_rank).to(device)
    n_par = sum(x.numel() for x in model.parameters())
    if rank == 0:
        if args.full_frame:
            gh, gw = 720 // args.vit_patch, 1440 // args.vit_patch; toks = f"整幅 {gh}x{gw}={gh*gw}"
        else:
            toks = f"crop={args.patch} tokens={(args.patch//args.vit_patch)**2}"
        print(f"[ViT] patch={args.vit_patch} dim={args.dim} depth={args.depth} heads={args.heads} "
              f"pos={args.pos_type} head_up={args.head_up} {toks}  params={n_par/1e6:.1f}M  "
              f"world={world}  train={args.train_years[0]}-{args.train_years[-1]} test={args.test_year} "
              f"patience={args.patience} lr={args.lr:.1e} lr_patience={args.lr_patience} lr_factor={args.lr_factor}", flush=True)

    TD.fit_deterministic(model, stats, args, device, (rank, world, local, is_dist), sp_ctx=sp_ctx)
    if args.eval_after_train and sp_ctx is not None:
        # 序列并行整幅评测: 保持 SP 开, 所有 rank 参与前向集合通信(~1/sp 快), 仅 rank0 算指标/写文件
        TD.evaluate(model, None, stats, args, device, is_main=(rank == 0))
    elif args.eval_after_train and rank == 0:
        TD.evaluate(model, None, stats, args, device)     # 非 SP: rank0 单进程整幅/切窗评测

    if is_dist:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
