#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2-rank 梯度同步冒烟: 证明 DDP all-reduce 真的生效, 且各 rank 确实在喂不同数据。

用法(纯 CPU, gloo 后端, 不需要 GPU 也不需要提交作业):
  torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_ddp_grad_sync

验证四件事:
  1) 各 rank 拿到的帧互不重叠(分片正确, 否则等于重复训同一批数据);
  2) 各 rank 的本地 loss 互不相同(证明确实在喂不同数据, 检验本测试不是假阳性);
  3) backward 后各 rank 梯度逐位一致(all-reduce 生效);
  4) optimizer.step 后各 rank 参数逐位一致。

前向若被改成传裸 module(绕过 DDP 包装体), 梯度同步会静默失效且全程零报错 —— 作业正常
结束、loss 正常下降、checkpoint 正常保存, 多卡实际等效单卡。设 DDP_SYNC_TEST_BYPASS=1
可人为制造该错误来验证本测试自身有分辨力(预期以 AssertionError 退出)。
"""
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.models.unet import UNet, masked_mse

MODE = "history_51"
YEARS = [2019, 2020]
CROP = 128            # DDP 接线与帧大小无关; 用小块让 CPU 上秒级可跑
STEPS = 3


def _land_crop(land, step, rank, min_land=0.3):
    """确定性地挑一个陆地占比够高的裁块起点; 起点对齐到 FACTOR 网格。"""
    H, W = land.shape
    ny, nx = (H - CROP) // C.FACTOR, (W - CROP) // C.FACTOR
    for k in range(ny * nx):
        j = (step * 7919 + rank * 104729 + k * 3571) % (ny * nx)
        y0, x0 = (j // nx) * C.FACTOR, (j % nx) * C.FACTOR
        if land[y0:y0 + CROP, x0:x0 + CROP].mean() >= min_land:
            return y0, x0
    raise AssertionError("找不到陆地占比足够的裁块")


def _gather_equal(tensors, name, world, step):
    """把本 rank 的一串张量拼平后 all-gather, 要求各 rank 逐位相同。"""
    v = torch.cat([t.detach().double().reshape(-1) for t in tensors])
    buf = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(buf, v)
    for r in range(1, world):
        if not torch.equal(buf[0], buf[r]):
            d = (buf[0] - buf[r]).abs().max().item()
            raise AssertionError(f"step {step}: rank0 与 rank{r} 的 {name} 不一致, 最大差 {d:.3e}")


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)                       # 各 rank 同一初始权重
    np.random.seed(0)
    assert world >= 2, "本测试至少要 2 个 rank"

    stats = Stats()
    ds = DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, YEARS, stats, mode=MODE, era5_cache_years=2)
    fi = FrameIndex([2020], YEARS, history_days=C.pairing_history_days(MODE),
                    lags=C.history_lags(MODE), split="test")

    # 分片: 第 i 步本 rank 取第 (i*world + rank) 帧, 与其它 rank 天然不重叠
    mine = [fi.frames[i * world + rank] for i in range(STEPS)]
    buf = [None] * world
    dist.all_gather_object(buf, mine)
    flat = [f for part in buf for f in part]
    assert len(set(flat)) == len(flat), f"各 rank 的帧有重叠: {buf}"

    cin = C.cond_channels(MODE)
    assert cin == 51, cin
    model = UNet(cin, 1, base=16, pos_grid=0)
    net = model if os.environ.get("DDP_SYNC_TEST_BYPASS") == "1" else DDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    it = C.TARGETS.index("2m_temperature_max")

    losses = []
    for step in range(STEPS):
        y, d = mine[step]
        i = fi.frames.index((y, d))
        cond, tgt, mask, _ = ds.full(y, d, fi.history_of(i))
        y0, x0 = _land_crop(ds.mask, step, rank)
        cond, tgt, mask, _ = ds.crop(cond, tgt, mask, np.zeros_like(tgt), y0, x0, CROP)
        cb = torch.from_numpy(cond)[None]
        tb = torch.from_numpy(tgt[it:it + 1])[None]
        mb = torch.from_numpy(mask)[None]
        # 全海裁块会让 masked_mse 变成 0/0 -> loss 与梯度全零, 于是"各 rank 梯度一致"
        # 空洞成立。非平凡性必须显式断言, 否则本测试会以假阳性通过。
        assert mb.mean().item() >= 0.3, f"裁块陆地占比 {mb.mean().item():.3f} 过低"

        loss = masked_mse(net(cb), tb, mb)
        opt.zero_grad(); loss.backward()
        losses.append(float(loss.item()))
        assert losses[-1] > 0, f"step {step} loss 为 {losses[-1]}, 梯度一致性无意义"
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "没有梯度"
        gn = float(torch.cat([g.reshape(-1) for g in grads]).norm())
        assert gn > 1e-12, f"step {step} 梯度范数 {gn}, 同步检验空洞"
        _gather_equal(grads, "梯度", world, step)
        opt.step()
        _gather_equal(list(model.parameters()), "参数", world, step)
        if rank == 0:
            print(f"[PASS] step {step}: 梯度与参数全 rank 一致", flush=True)

    lb = [None] * world
    dist.all_gather_object(lb, losses)
    for step in range(STEPS):
        vals = [lb[r][step] for r in range(world)]
        assert len(set(vals)) == world, \
            f"step {step} 各 rank 本地 loss 相同 {vals}, 说明喂的是同一批数据, 本测试无分辨力"
    if rank == 0:
        print(f"[PASS] 各 rank 本地 loss 互不相同 {[round(v, 6) for v in lb[0][:1]]} ...", flush=True)
        print(f"[PASS] 帧分片互不重叠: {buf}", flush=True)
        print("ALL PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
