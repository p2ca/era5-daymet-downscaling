#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""走**真实训练入口**的 2-rank 检验: 训练后各 rank 权重是否一致、帧集合守卫是否生效。

用法(纯 CPU, gloo 后端, 不需要 GPU 也不占队列):
  torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_train_ddp

与 test_ddp_grad_sync 的分工: 那个用手搭的小循环验证 DDP 包装体本身, 这个跑完整的
`train_downscale.fit()`, 覆盖真实的取帧、autocast、优化器与保存路径。两处都要有 ——
接线正确不等于训练循环里用对了。

三件事:
  1) fit() 跑完后各 rank 的权重逐位一致(梯度确实跨 rank 同步了);
  2) 负对照: 换个种子重跑必须给出不同权重, 否则第 1 条是恒真的;
  3) 各 rank 帧集合不同时, 启动守卫必须当场抛错 —— 不拦的话会挂到 NCCL 看门狗超时,
     表现为作业莫名卡死 30 分钟后被杀。
"""
import argparse
import os
import shutil
import tempfile

import torch
import torch.distributed as dist

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.training import train_downscale as TD

MODE = "history_51"


def make_args(out, seed=0):
    return argparse.Namespace(
        era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR, out=out,
        target=C.TARGETS[0], mode=MODE, train_years=[2019], val_years=[2020],
        epochs=2, batch=1, epoch_frames=0, steps_per_epoch=2, val_steps=2,
        lr=2e-4, weight_decay=0.0, lr_patience=4, lr_factor=0.5, min_lr=1e-6,
        patience=10, grad_clip=0.0, amp=False, workers=0, seed=seed,
        resume_from="", crop=128, smoke=True, base=16, pos_grid=0)


def _all_same(model, world, label):
    v = torch.cat([p.detach().double().reshape(-1) for p in model.parameters()])
    buf = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(buf, v)
    for r in range(1, world):
        if not torch.equal(buf[0], buf[r]):
            raise AssertionError(
                f"{label}: rank0 与 rank{r} 权重不一致, 最大差 "
                f"{(buf[0] - buf[r]).abs().max().item():.3e}")
    return buf[0]


def main():
    ddp_info = TD.setup_ddp()
    rank, world, local, device, is_dist = ddp_info
    assert is_dist and world >= 2, "本测试要用 torchrun 起至少 2 个 rank"
    root = tempfile.mkdtemp(prefix="ddptrain-") if rank == 0 else None
    box = [root]
    dist.broadcast_object_list(box, src=0)
    root = box[0]

    try:
        _, model = TD.fit(make_args(os.path.join(root, "a")), ddp_info)
        w0 = _all_same(model, world, "seed=0 训练后")
        if rank == 0:
            print("[PASS] fit() 跑完各 rank 权重逐位一致", flush=True)

        _, model2 = TD.fit(make_args(os.path.join(root, "b"), seed=7), ddp_info)
        w1 = _all_same(model2, world, "seed=7 训练后")
        assert not torch.equal(w0, w1), \
            "换种子后权重仍完全相同, 说明第 1 条断言是恒真的, 本测试无分辨力"
        if rank == 0:
            print("[PASS] 负对照: 换种子给出不同权重", flush=True)

        # 各 rank 传入不同的可用年份 -> 丢帧数不同 -> 帧集合不同, 守卫必须拦下
        avail = [2019, 2020] if rank == 0 else [2020]
        fi = FrameIndex([2020], avail, history_days=2, lags=C.HISTORY_LAGS, split="neg")
        try:
            TD.assert_same_frames_across_ranks(fi, device, True, "负对照")
        except RuntimeError as e:
            assert "帧集合不同" in str(e), str(e)[:200]
            if rank == 0:
                print(f"[PASS] 帧集合不一致被拦下: rank0 {len(fi)} 帧", flush=True)
        else:
            raise AssertionError("各 rank 帧集合不同却没有被拦下")

        if rank == 0:
            print("ALL PASS", flush=True)
    finally:
        dist.barrier()
        if rank == 0 and root:
            shutil.rmtree(root, ignore_errors=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
