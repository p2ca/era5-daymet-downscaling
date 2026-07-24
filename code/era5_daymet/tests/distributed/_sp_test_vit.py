#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""SP-3 端到端正确性: K 进程序列并行 ViT 的前向输出 + 每个参数梯度, 必须与单进程整幅 ViT
逐一致(fp64, gloo/CPU)。验证 token 分片 / gather / 分类梯度同步(sharded SP-SUM vs
redundant 不 SUM)全部正确。dp_size=1(全部 rank 同一 SP 组)。运行: python _sp_test_vit.py [K]"""
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world, H, W):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)
    from era5_daymet.models import seq_parallel_attn as SP
    from era5_daymet.training import train_vit as TV

    dim, depth, heads, Cin, Cout, patch = 32, 2, 4, 5, 3, 2
    B = 2
    # 1) 参照: 单进程整幅 ViT(sp_group=None), 同 seed 构造 + broadcast 兜底一致
    torch.manual_seed(0)
    ref = TV.ViT(Cin, Cout, img=8, patch=patch, dim=dim, depth=depth, heads=heads,
                 pos_type="sincos", head_up="pixelshuffle", full_frame=True).double()
    sd = ref.state_dict()
    for k in sd:
        dist.broadcast(sd[k], src=0)
    ref.load_state_dict(sd)

    # 2) SP ViT: 同权重, sp_group=WORLD
    sp = TV.ViT(Cin, Cout, img=8, patch=patch, dim=dim, depth=depth, heads=heads,
                pos_type="sincos", head_up="pixelshuffle", full_frame=True,
                sp_group=dist.group.WORLD, sp_size=world, sp_rank=rank).double()
    sp.load_state_dict(ref.state_dict())      # 键名与 MHA 版一致, 可直接载

    # 3) 相同完整输入(所有 SP rank 协作同一帧)
    torch.manual_seed(123)
    x = torch.randn(B, Cin, H, W)

    xr = x.clone().requires_grad_(True); xs = x.clone().requires_grad_(True)
    y_ref = ref(xr); y_sp = sp(xs)
    fwd_err = (y_ref.detach() - y_sp.detach()).abs().max().item()

    y_ref.sum().backward()
    y_sp.sum().backward()
    # SP 梯度同步(dp_size=1 -> 只做 SP-SUM(sharded)/不动(redundant))
    sharded, redundant = sp.sp_param_groups()
    SP.sp_sync_grads(sharded, redundant, dist.group.WORLD, None, dp_size=1)

    # 4) 逐参数比对梯度
    ref_g = {n: p.grad for n, p in ref.named_parameters()}
    max_gerr = 0.0; worst = ""
    for n, p in sp.named_parameters():
        e = (p.grad - ref_g[n]).abs().max().item()
        if e > max_gerr: max_gerr, worst = e, n
    # 输入 x 在每 SP rank 冗余过 embed、只用本段 token -> xs.grad 是分片偏梯度, SP-SUM 后才是完整
    dist.all_reduce(xs.grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    xg_err = (xs.grad - xr.grad).abs().max().item()

    errs = torch.tensor([fwd_err, max_gerr, xg_err])
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    if rank == 0:
        f, g, xg = errs.tolist()
        ok = max(f, g, xg) < 1e-8
        print(f"[SP-ViT K={world}] H×W={H}×{W} tokens={(H//patch)*(W//patch)} 每rank={(H//patch)*(W//patch)//world}")
        print(f"  前向输出 max|Δ|   = {f:.2e}")
        print(f"  参数梯度 max|Δ|   = {g:.2e}  (最差参数: {worst})")
        print(f"  输入梯度 max|Δ|   = {xg:.2e}")
        print(f"  -> {'PASS ✅ SP-ViT 与单进程整幅 ViT 前向+全参数梯度一致' if ok else 'FAIL ❌'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    H, W = 8, 4 * K                      # gh=4, gw=2K, N=8K 可被 K 整除
    mp.spawn(run, args=(K, H, W), nprocs=K, join=True)
