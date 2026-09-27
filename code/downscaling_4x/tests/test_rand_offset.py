#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_rand_offset.py — 阶段 A JiT 的训练期随机切块起点(--rand-offset)
============================================================================
钉住五件事, 任何一件坏了都不会有东西报错(训练照跑、loss 照降、作业正常结束):

  1. 起点真的进了前向: 同种子下开与不开, 训出的权重必须不同 —— 只改了开关而权重逐位
     相同, 说明 offset 根本没传到模型, 而"启用了随机裁块"会被当成既成事实写进实验记录
  2. 训练起点逐步重掷, 验证起点逐 batch 固定且跨 epoch 不变(val 才跨轮可比)
  3. 各 rank 同一步抽到不同起点: 一个全局批覆盖多种网格相位
  4. 开着它续训仍逐位复现(起点只由 (seed, epoch, 步, rank) 决定, 不依赖任何断点状态)
  5. 续训契约钉住该开关: 接力中途翻档被拒; 早于该字段的旧断点按"固定 (0,0)"解释

需要真实数据, 纯 CPU, 约两分钟。
运行: python -m downscaling_4x.tests.test_rand_offset
============================================================================
"""
import argparse
import os
import tempfile

import torch

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.models.jit_backbone import draw_patch_offset
from downscaling_4x.training import train_downscale as TD

MODE = "history_51"
SINGLE = (0, 1, 0, torch.device("cpu"), False)


def make_args(out, epochs, resume_from="", **over):
    a = argparse.Namespace(
        era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR, out=out,
        target=C.TARGETS[0], mode=MODE, train_years=[2019], val_years=[2020],
        epochs=epochs, batch=1, epoch_frames=0, steps_per_epoch=2, val_steps=2,
        lr=2e-4, weight_decay=0.0, lr_patience=4, lr_factor=0.5, min_lr=1e-6,
        patience=10, grad_clip=0.0, amp=False, workers=0, seed=0,
        improve_criterion="abs", improve_tol=1e-4,
        resume_from=resume_from, crop=128, smoke=True, base=16, pos_grid=0,
        arch="jit", hidden=32, depth=2, heads=2, mlp_ratio=2.0, bottleneck=16,
        patch=16, patch_margin=0, attn_dropout=0.0, proj_dropout=0.0,
        moe=False, experts=4, experts_per_tok=2, moe_intermediate=0,
        routed_scaling=1.0, moe_all_layers=False, moe_no_shared=False,
        refine_head=0, router="tc", gating="norm", rand_offset=1)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _w(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def offsets(seed, ep, steps, rank=0, patch=16):
    g = torch.Generator()
    out = []
    for s in range(steps):
        g.manual_seed(TD.patch_offset_seed(seed, ep, s, rank))
        out.append(draw_patch_offset(patch, "cpu", g))
    return out


def test_seed_stream_shape():
    """训练逐步重掷、验证跨 epoch 固定、各 rank 互不相同。"""
    tr0, tr1 = offsets(0, 0, 8), offsets(0, 1, 8)
    assert len(set(tr0)) > 1, "同一 epoch 内的起点必须随步变化"
    assert tr0 != tr1, "不同 epoch 的起点序列应当不同"
    assert tr0 == offsets(0, 0, 8), "同一 (seed, epoch, 步) 必须复现同一起点"
    va0, va1 = offsets(0, -1, 8), offsets(0, -1, 8)
    assert va0 == va1, "验证段(ep=-1)的起点必须跨 epoch 恒定"
    assert offsets(0, 0, 8, rank=1) != tr0, "同一步各 rank 应落在不同起点"
    assert offsets(1, 0, 8) != tr0, "换 --seed 应换掉整条起点流"
    print("  起点流的形状 OK")


def test_offset_reaches_the_model():
    """同种子下开与不开, 权重必须不同 —— 否则 offset 没有真的进前向。"""
    with tempfile.TemporaryDirectory(prefix="randoff-") as root:
        on, off = os.path.join(root, "on"), os.path.join(root, "off")
        TD.fit(make_args(on, 1, rand_offset=1), SINGLE)
        TD.fit(make_args(off, 1, rand_offset=0), SINGLE)
        wa, wb = _w(os.path.join(on, "last.pt")), _w(os.path.join(off, "last.pt"))
        same = [k for k in wa if torch.equal(wa[k], wb[k])]
        assert len(same) < len(wa), "开关翻档后权重逐位相同: 随机起点没有进到模型前向"
    print("  起点确实进了前向 OK")


def test_resume_bit_exact_with_rand_offset():
    """开着随机起点时, 跑 2 轮 vs 跑 1 轮再续 1 轮, 权重仍逐位相同。"""
    with tempfile.TemporaryDirectory(prefix="randoff-") as root:
        a_dir, b_dir = os.path.join(root, "a"), os.path.join(root, "b")
        TD.fit(make_args(a_dir, 2), SINGLE)
        TD.fit(make_args(b_dir, 1), SINGLE)
        TD.fit(make_args(b_dir, 2, resume_from=os.path.join(b_dir, "last.pt")), SINGLE)
        wa, wb = _w(os.path.join(a_dir, "last.pt")), _w(os.path.join(b_dir, "last.pt"))
        bad = [k for k in wa if not torch.equal(wa[k], wb[k])]
        assert not bad, f"开着 --rand-offset 续训后权重不逐位相同, {len(bad)}/{len(wa)} 个张量有差异"
    print("  开着随机起点的续训逐位复现 OK")


def test_contract_pins_the_switch():
    """接力中途翻档被拒; 早于该字段的旧断点按固定 (0,0) 解释。"""
    with tempfile.TemporaryDirectory(prefix="randoff-") as root:
        d = os.path.join(root, "c")
        TD.fit(make_args(d, 1), SINGLE)
        try:
            TD.fit(make_args(d, 2, rand_offset=0, resume_from=os.path.join(d, "last.pt")), SINGLE)
        except RuntimeError as e:
            assert "rand_offset" in str(e), f"拒绝理由里应点名 rand_offset: {e}"
        else:
            raise AssertionError("接力中途把随机起点关掉必须被契约拒绝")
    legacy = {"refine_head": 0, "router": "tc", "gating": "norm",
              "improve_criterion": "abs", "improve_tol": 1e-4}
    filled = TD.apply_legacy_defaults(legacy)
    assert "rand_offset" in filled and legacy["rand_offset"] == 0, "旧断点缺该键时应按固定起点解释"
    print("  续训契约钉住开关 OK")


def test_non_jit_rejects():
    """UNet / CorrDiff 的 forward 不收 offset, 开这个开关必须当场拒绝。"""
    with tempfile.TemporaryDirectory(prefix="randoff-") as root:
        try:
            TD.fit(make_args(os.path.join(root, "u"), 1, arch="unet", rand_offset=1), SINGLE)
        except SystemExit as e:
            assert "rand-offset" in str(e)
        else:
            raise AssertionError("--arch unet 配 --rand-offset 1 必须被拒")
    print("  非 JiT 主干拒绝该开关 OK")


if __name__ == "__main__":
    test_seed_stream_shape()
    test_non_jit_rejects()
    test_offset_reaches_the_model()
    test_resume_bit_exact_with_rand_offset()
    test_contract_pins_the_switch()
    print("test_rand_offset: 全部通过")
