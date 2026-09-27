#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_unet 按 --arch 回填口径默认值的检验。

默认值写错不会报错——训练照跑、loss 照降, 只是口径无声偏离既定配置。这里把三个
结构族的关键默认值逐项钉死, 并验证显式传参优先与 jit 的派生值(min_lr=lr 恒定、
耐心=epochs 不早停)。

纯 argparse, 不依赖数据与 torch 运行时。

运行: python -m downscaling_4x.tests.test_arch_defaults
"""
from downscaling_4x import contract as C
from downscaling_4x.training.train_unet import apply_arch_defaults, build_parser


def parse(*argv):
    args = build_parser().parse_args(["--out", "x", *argv])
    apply_arch_defaults(args)
    return args


def test_unet_defaults():
    a = parse("--arch", "unet")
    assert (a.target, a.base, a.pos_grid) == ("all", 192, 0)
    assert (a.lr, a.weight_decay, a.grad_clip) == (1e-4, 1e-4, 1.0)
    assert (a.lr_patience, a.min_lr, a.epochs, a.epoch_frames, a.patience) == \
        (3, 1e-6, 80, 3840, 12)


def test_corrdiff_defaults():
    a = parse("--arch", "corrdiff")
    assert (a.target, a.pos_grid, a.lr) == (C.TARGETS[0], 4, 2e-4)
    assert (a.weight_decay, a.grad_clip, a.lr_patience, a.min_lr) == (1e-4, 1.0, 3, 1e-6)
    assert (a.epochs, a.epoch_frames, a.patience) == (80, 3840, 12)
    assert (a.model_channels, a.channel_mult, a.num_blocks, a.attn_bottleneck) == \
        (64, [1, 2, 2, 2, 2], 4, 1)


def test_jit_defaults():
    a = parse("--arch", "jit")
    assert (a.patch, a.patch_margin, a.bottleneck) == (16, 2, 256)
    assert (a.hidden, a.depth, a.heads, a.mlp_ratio) == (384, 12, 6, 4.0)
    assert (a.target, a.pos_grid, a.weight_decay, a.grad_clip) == (C.TARGETS[0], 0, 0.0, 0.0)
    assert a.lr == 1e-4 and a.min_lr == a.lr, "jit 的 lr 必须恒定: min_lr=lr 锁死 plateau 减半"
    assert a.patience == a.epochs and a.lr_patience == a.epochs, "jit 不早停、不减半"
    assert (a.epochs, a.epoch_frames) == (3906, 4096)
    assert a.epochs * a.epoch_frames == 15_998_976, "固定预算 ~16M 帧"


def test_explicit_args_win():
    a = parse("--arch", "unet", "--lr", "5e-5", "--pos-grid", "4", "--epochs", "7")
    assert (a.lr, a.pos_grid, a.epochs) == (5e-5, 4, 7)
    b = parse("--arch", "jit", "--lr", "3e-4")
    assert b.min_lr == 3e-4, "jit 显式改 lr 后 min_lr 必须跟随, 否则恒定 lr 口径被无声破坏"
    c = parse("--arch", "jit", "--patience", "5", "--epochs", "10")
    assert (c.patience, c.lr_patience) == (5, 10), "显式 patience 优先; lr_patience 仍随 epochs"


def main():
    tests = [test_unet_defaults, test_corrdiff_defaults, test_jit_defaults,
             test_explicit_args_win]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
