#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""SP-2 正确性测试: K 进程(gloo/CPU, fp64)的序列并行注意力, 前向输出与梯度
必须与单进程全注意力精确对齐。验证 all-gather-KV 前向 + autograd all_gather 反向
(含 param 梯度在 SP 维求和 = 全量梯度)。运行: python _sp_test_multiproc.py [K]"""
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world, dim, heads, B, N):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)
    from era5_daymet.models.seq_parallel_attn import SPSelfAttention

    # 1) 全体一致的权重(同 seed 构造 + 从 rank0 broadcast 兜底)
    torch.manual_seed(0)
    ref = SPSelfAttention(dim, heads, sp_group=None)        # 单进程全注意力 = 参照
    for p in ref.parameters():
        dist.broadcast(p.data, src=0)

    # 2) 全体一致的完整输入
    torch.manual_seed(123)
    x_full = torch.randn(B, N, dim)

    # 3) 参照: 本地整幅全注意力(每 rank 各算, 结果一致)
    xr = x_full.clone().requires_grad_(True)
    y_ref = ref(xr)
    y_ref.sum().backward()
    ref_out = y_ref.detach()
    ref_gw = ref.in_proj_weight.grad.clone()
    ref_go = ref.out_proj.weight.grad.clone()

    # 4) 分片序列并行: 同权重, sp_group=WORLD
    sp = SPSelfAttention(dim, heads, sp_group=dist.group.WORLD)
    with torch.no_grad():
        for a, b in zip(sp.parameters(), ref.parameters()):
            a.copy_(b)
    Nl = N // world
    xs = x_full[:, rank * Nl:(rank + 1) * Nl].clone().requires_grad_(True)
    y_local = sp(xs)
    fwd_err = (y_local.detach() - ref_out[:, rank * Nl:(rank + 1) * Nl]).abs().max().item()

    y_local.sum().backward()
    # param 梯度: 各 rank 持自己 shard 的贡献, SP 维求和 = 全量梯度
    gw = sp.in_proj_weight.grad.clone(); dist.all_reduce(gw, op=dist.ReduceOp.SUM)
    go = sp.out_proj.weight.grad.clone(); dist.all_reduce(go, op=dist.ReduceOp.SUM)
    gwerr = (gw - ref_gw).abs().max().item()
    goerr = (go - ref_go).abs().max().item()
    # 输入梯度: 各 rank 对应自己 shard, 直接比对
    xgerr = (xs.grad - xr.grad[:, rank * Nl:(rank + 1) * Nl]).abs().max().item()

    errs = torch.tensor([fwd_err, gwerr, goerr, xgerr])
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    if rank == 0:
        f, gw_, go_, xg = errs.tolist()
        ok = max(f, gw_, go_, xg) < 1e-8
        print(f"[SP={world}] (fp64, N={N}->每rank {Nl})")
        print(f"  前向 max|Δ|            = {f:.2e}")
        print(f"  in_proj 梯度(SP求和)  = {gw_:.2e}")
        print(f"  out_proj 梯度(SP求和) = {go_:.2e}")
        print(f"  输入梯度               = {xg:.2e}")
        print(f"  -> {'PASS ✅ 分片SP与单进程全注意力精确对齐(前向+反向)' if ok else 'FAIL ❌'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dim, heads, B, N = 64, 4, 2, 8 * K      # N 可被 K 整除
    mp.spawn(run, args=(K, dim, heads, B, N), nprocs=K, join=True)
