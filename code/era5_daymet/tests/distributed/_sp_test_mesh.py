#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""SP 2D-mesh(SP×DP)梯度同步正确性(补做: 之前只测过 dp_size=1)。
world=sp_size×dp_size(默认 2×2=4), gloo/CPU/fp64。每个 DP 组喂不同帧、SP 组喂同帧;
sp_sync_grads 后**所有 rank 梯度应 = dp_size 帧各自全帧梯度的平均**(与单进程参照逐一致)。
运行: python _sp_test_mesh.py [sp_size] [dp_size]"""
import os, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp


def run(rank, sp_size, dp_size, H, W):
    world = sp_size * dp_size
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29563")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)
    from era5_daymet.models import seq_parallel_attn as SP
    from era5_daymet.training import train_vit as TV

    sp_group, dp_group, sp_rank, dp_rank, dpz = SP.build_2d_mesh(sp_size, world=world, rank=rank)
    dim, depth, heads, Cin, Cout, patch, B = 32, 2, 4, 5, 3, 2, 1

    # 统一权重
    torch.manual_seed(0)
    def make(spg, spr):
        return TV.ViT(Cin, Cout, img=8, patch=patch, dim=dim, depth=depth, heads=heads,
                      pos_type="sincos", head_up="pixelshuffle", full_frame=True,
                      sp_group=spg, sp_size=(sp_size if spg is not None else 1), sp_rank=spr).double()
    m = make(sp_group, sp_rank)
    sd = m.state_dict()
    for k in sd: dist.broadcast(sd[k], src=0)
    m.load_state_dict(sd)

    # dp_size 个不同帧(每个 DP 组一帧, 全体一致地生成)
    frames = []
    for d in range(dp_size):
        torch.manual_seed(1000 + d); frames.append(torch.randn(B, Cin, H, W))

    # 本 rank 的输入 = 自己 DP 组那一帧
    x = frames[dp_rank].clone()
    y = m(x); y.sum().backward()
    sharded, redundant = m.sp_param_groups()
    SP.sp_sync_grads(sharded, redundant, sp_group, dp_group, dpz)

    # 参照(仅 rank0 算): 单进程全 ViT, 对 dp_size 帧各求全帧梯度再平均
    if rank == 0:
        ref = make(None, 0); ref.load_state_dict(m.state_dict())  # 同权重(注意 m 权重未被 opt 改)
        refg = {n: torch.zeros_like(p) for n, p in ref.named_parameters()}
        for d in range(dp_size):
            ref.zero_grad()
            ref(frames[d]).sum().backward()
            for n, p in ref.named_parameters(): refg[n] += p.grad / dp_size
        max_e = 0.0; worst = ""
        for n, p in m.named_parameters():
            e = (p.grad - refg[n]).abs().max().item()
            if e > max_e: max_e, worst = e, n
        ok = max_e < 1e-8
        print(f"[mesh sp={sp_size}×dp={dp_size}=world{world}] 同步后参数梯度 vs (dp帧全帧梯度均值) "
              f"max|Δ|={max_e:.2e} (最差 {worst}) -> {'PASS ✅' if ok else 'FAIL ❌'}")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    sp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    H, W = 8, 4 * sp                     # N=(H/2)(W/2)=2sp*4=... 确保被 sp 整除
    mp.spawn(run, args=(sp, dp, H, W), nprocs=sp * dp, join=True)
