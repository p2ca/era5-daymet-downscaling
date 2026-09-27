#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
train_unet.py — 确定性 UNet 回归 (ERA5 120x240 -> Daymet 480x960, 4x)
============================================================================
`--target` 选单目标(三个目标各训一个模型)或 all(三目标联合); 条件模式由 `--mode` 选:

  baseline_21        当天 ERA5 + 静态 + 年内相位, 用全部可用帧
  history_control_21 通道与 baseline 完全相同, 但只用历史齐全的帧
  history_51         再加 t-2/t-1 的 ERA5

**衡量历史通道的增量必须拿 history_51 去比 history_control_21**, 不能比 baseline_21:
后两者的通道数一样, 差别只在帧集合, 比错了会把"训练集变小"算进"加了历史通道"的效应里,
而不会有任何东西报错。

口径默认值(目标/规模/lr/wd/预算等)按 `--arch` 从 ARCH_DEFAULTS 回填: unet/corrdiff 是
plateau+早停口径, jit 是固定预算口径(lr 恒定、不早停、~16M 帧)。显式传参永远优先。

用法:
  自测(真实数据, 分钟级, 纯 CPU):
      python -m downscaling_4x.training.train_unet --out runs/_smoke/u1 --smoke
  2-rank DDP 自测(login node, gloo, 不占队列):
      torchrun --nproc_per_node=2 -m downscaling_4x.training.train_unet --out ... --smoke
  多节点(Frontier): 把下面这行放进 srun 后面
      python -m downscaling_4x.training.train_unet --out runs/exp/<id> --mode history_51 --amp
============================================================================
"""
import argparse

import torch.distributed as dist

from downscaling_4x import contract as C
from downscaling_4x.training import train_downscale as TD


# 各结构族的口径默认值; 解析后由 apply_arch_defaults 回填未显式给出的项。
# unet / corrdiff: plateau 减半 + 早停; jit: 固定预算(3906 x 4096 ≈ 16M 帧)跑满。
ARCH_DEFAULTS = {
    "unet":     dict(target="all", base=192, lr=1e-4, weight_decay=1e-4, grad_clip=1.0,
                     lr_patience=3, min_lr=1e-6, epochs=80, epoch_frames=3840, patience=12,
                     pos_grid=0),
    "corrdiff": dict(target=C.TARGETS[0], base=64, lr=2e-4, weight_decay=1e-4, grad_clip=1.0,
                     lr_patience=3, min_lr=1e-6, epochs=80, epoch_frames=3840, patience=12,
                     pos_grid=4),
    "jit":      dict(target=C.TARGETS[0], base=64, lr=1e-4, weight_decay=0.0, grad_clip=0.0,
                     epochs=3906, epoch_frames=4096, pos_grid=0),
}


def apply_arch_defaults(args):
    """按 --arch 回填未显式给出的口径默认值; 命令行显式传参永远优先。

    jit 的三个派生值单独解析: min_lr=lr 把 plateau 减半锁成空操作(lr 全程恒定),
    lr_patience 与 patience 取 epochs(不减半、不早停, 按固定预算跑满)。
    """
    for k, v in ARCH_DEFAULTS[args.arch].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.arch == "jit":
        if args.min_lr is None:
            args.min_lr = args.lr
        if args.lr_patience is None:
            args.lr_patience = args.epochs
        if args.patience is None:
            args.patience = args.epochs


def build_parser():
    p = argparse.ArgumentParser(description="4x UNet 降尺度训练 (DDP + 早停)",
                                formatter_class=argparse.RawTextHelpFormatter)
    TD.add_common_args(p)
    p.add_argument("--base", type=int, default=64, help="UNet 通道基数(U1/U2/U3 = 64/128/192); 仅 --arch unet")
    p.add_argument("--arch", choices=["unet", "corrdiff", "jit"], default="unet",
                   help="确定性主体的结构族; corrdiff = CorrDiff 阶段 A 的均值网 μ, "
                        "jit = 确定性 JiT(两阶段的阶段 A, 即 JDA/JMA)")
    p.add_argument("--model-channels", type=int, default=64, help="[corrdiff] 第 0 级通道数")
    p.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 2, 2, 2],
                   help="[corrdiff] 各级通道倍数; 长度决定层级数")
    p.add_argument("--num-blocks", type=int, default=4, help="[corrdiff] 每级 ResBlock 数")
    p.add_argument("--attn-bottleneck", type=int, choices=[0, 1], default=1,
                   help="[corrdiff] 瓶颈自注意力开关")
    # --- [jit] 结构 ---
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--patch", type=int, default=16, help="[jit] 切块边长")
    p.add_argument("--patch-margin", type=int, default=2, help="[jit] 块间读取重叠")
    p.add_argument("--rand-offset", type=int, choices=[0, 1], default=0,
                   help="[jit] 训练时每步重掷切块网格起点 (dy,dx)∈[0,patch)², 各 rank 互不相同; "
                        "0=恒定 (0,0)(缺省)。验证段的起点逐 batch 固定且跨 epoch 不变, val 仍可比")
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--proj-dropout", type=float, default=0.0)
    # --- [jit] MoE: 与阶段 B 逐字段相同的配置 ---
    p.add_argument("--moe", action="store_true", help="奇数序块的 FFN 换成 DeepSeek 稀疏层")
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--experts-per-tok", type=int, default=2)
    p.add_argument("--moe-intermediate", type=int, default=0)
    p.add_argument("--routed-scaling", type=float, default=1.0)
    p.add_argument("--moe-all-layers", action="store_true")
    p.add_argument("--moe-no-shared", action="store_true")
    p.add_argument("--router", choices=["tc", "ec"], default="tc",
                   help="[jit+moe] 专家选择方向: tc=每 token 挑 top-K 专家(DeepSeek); "
                        "ec=每专家帧内挑 top-C token(Expert-Choice), FLOPs 与 tc 对齐")
    p.add_argument("--gating", choices=["norm", "raw"], default="norm",
                   help="[jit+moe] 门控权重: norm=当选专家上归一(tc 固定此档); "
                        "raw=sigmoid 原值(仅 ec, 保留多专家 token 的幅度通道)")
    p.add_argument("--refine-head", type=int, choices=[0, 1], default=0,
                   help="[jit] 输出端全分辨率卷积精修头(静态通道直连); "
                        "0=关(与无此参数的版本逐字节相同)")
    p.add_argument("--pos-grid", type=int, choices=[0, 4], default=0,
                   help="主体内的正弦位置网格(0 关 / 4 开); 条件合同不含任何位置通道, "
                        "需要绝对位置信息的主体在此开启")
    # 口径默认值按 --arch 决定(ARCH_DEFAULTS): 先清成 None 哨兵, 解析后回填, 显式传参优先
    p.set_defaults(target=None, base=None, lr=None, weight_decay=None, grad_clip=None,
                   lr_patience=None, min_lr=None, epochs=None, epoch_frames=None,
                   patience=None, pos_grid=None)
    return p


def main():
    args = build_parser().parse_args()
    apply_arch_defaults(args)

    if args.smoke:
        TD.apply_smoke(args)

    ddp_info = TD.setup_ddp()
    rank, world, local, device, is_dist = ddp_info
    if rank == 0:
        cin = C.cond_channels(args.mode)
        if args.arch == "corrdiff":
            geom = (f"model_channels={args.model_channels} channel_mult={args.channel_mult} "
                    f"num_blocks={args.num_blocks} attn={args.attn_bottleneck}")
        elif args.arch == "jit":
            geom = (f"hidden={args.hidden} depth={args.depth} heads={args.heads} "
                    f"patch={args.patch} refine={args.refine_head} moe={args.moe}"
                    + (f" experts={args.experts}/{args.experts_per_tok} "
                       f"router={args.router}/{args.gating}" if args.moe else ""))
        else:
            geom = f"base={args.base}"
        print(f"[{args.arch}] mode={args.mode} Cin={cin} target={args.target} {geom} "
              f"pos_grid={args.pos_grid} amp={'bf16' if args.amp else 'fp32'} "
              f"device={device} world={world} "
              f"{'crop=' + str(args.crop) + '(smoke)' if args.crop else 'full-frame'}", flush=True)

    TD.fit(args, ddp_info)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
