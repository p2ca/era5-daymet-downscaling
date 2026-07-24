#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""验证"初始权重广播"修复: 2 rank SP(sp_size=2)对单个固定样本做迷你训练, 看 loss 是否随步数下降。
- 不广播初始权重(模拟 bug): 各 rank 权重不同 -> SP 组内 K/V 混不同权重 -> 前向不连贯 -> loss 卡住。
- 广播(修复): 权重一致 -> 正常连贯 -> loss 应显著下降。
运行: python _sp_test_converge.py [bcast=0/1]"""
import os, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp


def run(rank, world, bcast):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)
    from era5_daymet.models import seq_parallel_attn as SP
    from era5_daymet.training import train_vit as TV
    spg, dpg, spr, dpr, dpz = SP.build_2d_mesh(world, world=world, rank=rank)  # sp=world, dp=1

    # 各 rank 用不同 seed 构造 -> 初始权重不同(复现真实情况: 无 DDP 自动广播)
    torch.manual_seed(rank)
    m = TV.ViT(5,3,img=8,patch=2,dim=32,depth=2,heads=4,pos_type="sincos",head_up="pixelshuffle",
               full_frame=True, sp_group=spg, sp_size=world, sp_rank=spr).double()
    if bcast:                                   # ★修复: 广播 rank0 权重
        for p in m.parameters(): dist.broadcast(p.data, src=0)
    sharded, redundant = m.sp_param_groups()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # 多样本泛化(对不连贯权重更敏感): 拟合 N 个不同 (x,tgt), 看最终能拟合到多低
    torch.manual_seed(999)
    N = 24
    X = torch.randn(N,5,8,16); TGT = torch.randn(N,3,8,16)
    losses = []
    for step in range(300):
        i = step % N
        opt.zero_grad()
        loss = ((m(X[i:i+1]) - TGT[i:i+1])**2).mean()
        loss.backward()
        SP.sp_sync_grads(sharded, redundant, spg, dpg, dpz)
        opt.step()
        if step % 60 == 0 or step == 299:
            with torch.no_grad():
                full = sum(((m(X[j:j+1]) - TGT[j:j+1])**2).mean().item() for j in range(N)) / N
            losses.append(round(full, 4))
    if rank == 0:
        print(f"[converge bcast={bcast}] 全样本 loss 轨迹: {losses}  最终={losses[-1]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    bcast = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    mp.spawn(run, args=(2, bcast), nprocs=2, join=True)
