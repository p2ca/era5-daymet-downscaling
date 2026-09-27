#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练循环的单进程检验: 断点续训逐位复现、契约守卫、以及取帧分片的无放回性质。

断点续训写错是典型的静默错误 —— 权重装得回去、loss 接着降、作业正常结束, 只是采样流或
优化器状态被悄悄重置, 等价于换了个实验。这里用"一次跑两轮"与"跑一轮再续一轮"对拍,
要求最终权重**逐位相同**; 不逐位相同就说明有状态没接上。

需要真实数据, 纯 CPU, 约一分钟。

运行: python -m downscaling_4x.tests.test_train_resume
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
import torch

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.training import train_downscale as TD

MODE = "history_51"
SINGLE = (0, 1, 0, torch.device("cpu"), False)


def make_args(out, epochs, resume_from="", **over):
    a = argparse.Namespace(
        era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR, out=out,
        target=C.TARGETS[0], mode=MODE,
        train_years=[2019], val_years=[2020],
        epochs=epochs, batch=1, epoch_frames=0, steps_per_epoch=2, val_steps=2,
        lr=2e-4, weight_decay=0.0, lr_patience=4, lr_factor=0.5, min_lr=1e-6,
        patience=10, grad_clip=0.0, amp=False, workers=0, seed=0,
        resume_from=resume_from, crop=128, smoke=True, base=16, pos_grid=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def test_resume_is_bit_exact():
    """一次跑 2 轮 vs 跑 1 轮再续 1 轮, 最终权重必须逐位相同。"""
    root = tempfile.mkdtemp(prefix="resume-")
    try:
        a_dir, b_dir = os.path.join(root, "a"), os.path.join(root, "b")
        TD.fit(make_args(a_dir, 2), SINGLE)
        TD.fit(make_args(b_dir, 1), SINGLE)
        TD.fit(make_args(b_dir, 2, resume_from=os.path.join(b_dir, "last.pt")), SINGLE)
        wa, wb = _weights(os.path.join(a_dir, "last.pt")), _weights(os.path.join(b_dir, "last.pt"))
        assert set(wa) == set(wb)
        bad = [k for k in wa if not torch.equal(wa[k], wb[k])]
        assert not bad, (f"续训后权重与一次跑到底不同, {len(bad)}/{len(wa)} 个张量有差异; "
                         f"最大差 {max(float((wa[k]-wb[k]).abs().max()) for k in bad):.3e}")
        # 负对照: 换个种子必须给出不同的权重, 否则"逐位相同"是恒真的, 上面的断言无意义
        c_dir = os.path.join(root, "c")
        TD.fit(make_args(c_dir, 2, seed=7), SINGLE)
        wc = _weights(os.path.join(c_dir, "last.pt"))
        assert any(not torch.equal(wa[k], wc[k]) for k in wa), \
            "换种子后权重仍逐位相同, 说明训练没有真正依赖初始化, 本测试无分辨力"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_rejects_changed_contract():
    """改了被 pin 的字段还想原样续训, 必须当场拒绝而不是接着跑。"""
    root = tempfile.mkdtemp(prefix="resume-")
    try:
        d = os.path.join(root, "a")
        TD.fit(make_args(d, 1), SINGLE)
        last = os.path.join(d, "last.pt")
        for field, value in (("base", 32), ("mode", "history_control_21"),
                             ("target", C.TARGETS[1]), ("lr", 1e-3), ("seed", 7)):
            try:
                TD.fit(make_args(d, 2, resume_from=last, **{field: value}), SINGLE)
            except RuntimeError as e:
                assert "不一致" in str(e) or "契约" in str(e), str(e)[:200]
                continue
            raise AssertionError(f"改了 {field} 仍被接受续训")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_rejects_outside_out_dir():
    """checkpoint 必须在本实验目录内, 否则等于把一个实验的状态写进另一个实验。"""
    root = tempfile.mkdtemp(prefix="resume-")
    try:
        d = os.path.join(root, "a")
        TD.fit(make_args(d, 1), SINGLE)
        other = os.path.join(root, "b")
        os.makedirs(other, exist_ok=True)
        shutil.copy(os.path.join(d, "last.pt"), os.path.join(other, "last.pt"))
        try:
            TD.fit(make_args(d, 2, resume_from=os.path.join(other, "last.pt")), SINGLE)
        except RuntimeError as e:
            assert "必须位于" in str(e)
            return
        raise AssertionError("跨实验目录的 checkpoint 没有被拒绝")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stream_is_shuffled_without_replacement():
    """训练流每遍数据恰好覆盖全部帧一次, 且同一 epoch 内各 rank 的分片互不重叠。"""
    fi = FrameIndex([2019], [2019, 2020], history_days=2, lags=C.HISTORY_LAGS, split="train")
    n = len(fi)
    world, steps, batch = 4, 8, 1
    span = world * steps * batch

    def idx(ep, r, i):
        ds = TD.FrameDS.__new__(TD.FrameDS)                  # 只用索引逻辑, 不碰数据
        ds.fi, ds.base_seed = fi, 1234
        ds.deterministic, ds.stream = False, True
        ds.index_offset, ds.epoch_span, ds.epoch = r * steps * batch, span, ep
        ds._pass_id, ds._pass_perm, ds.perm = -1, None, None
        return ds._index(i)

    n_ep = 3 * n // span + 1
    # 按全局序号 g = ep*span + r*steps + i 的顺序展开; 只有这个顺序才对应"第几遍数据"
    order = [idx(ep, r, i) for ep in range(n_ep) for r in range(world)
             for i in range(steps * batch)]
    for k in range(len(order) // n):
        assert sorted(order[k * n:(k + 1) * n]) == list(range(n)), f"第 {k} 遍不是无放回置换"
    checked = 0
    for ep in range(n_ep):
        lo, hi = ep * span, (ep + 1) * span - 1
        if lo // n != hi // n:
            continue                                          # 跨遍边界, 本就可能取到两份置换
        vals = [idx(ep, r, i) for r in range(world) for i in range(steps * batch)]
        assert len(set(vals)) == len(vals), f"epoch {ep} 各 rank 分片重叠"
        checked += 1
    assert checked >= 2, "没有检到足够多的不跨遍 epoch, 本测试无分辨力"


def test_val_stream_is_fixed_across_epochs():
    """验证集必须逐 epoch 完全固定, 否则早停与 LR 减半的判据带采样噪声。"""
    fi = FrameIndex([2020], [2019, 2020], history_days=2, lags=C.HISTORY_LAGS, split="val")
    idx = []
    for ep in range(3):
        ds = TD.FrameDS.__new__(TD.FrameDS)
        ds.fi, ds.base_seed = fi, 987
        ds.deterministic, ds.stream = True, False
        ds.index_offset, ds.epoch_span, ds.epoch = 0, 0, ep
        ds._pass_id, ds._pass_perm = -1, None
        ds.perm = np.random.default_rng(987).permutation(len(fi))
        idx.append([ds._index(i) for i in range(8)])
    assert idx[0] == idx[1] == idx[2], idx


def test_crop_requires_smoke():
    """切窗与整幅学到的不是同一个函数; 正式训练用 --crop 必须被拒绝。"""
    root = tempfile.mkdtemp(prefix="resume-")
    try:
        a = make_args(os.path.join(root, "a"), 1, smoke=False, crop=128)
        try:
            TD.fit(a, SINGLE)
        except SystemExit as e:
            assert "crop" in str(e)
            return
        raise AssertionError("正式训练用 --crop 没有被拒绝")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    tests = [
        test_stream_is_shuffled_without_replacement,
        test_val_stream_is_fixed_across_epochs,
        test_crop_requires_smoke,
        test_resume_is_bit_exact,
        test_resume_rejects_changed_contract,
        test_resume_rejects_outside_out_dir,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
