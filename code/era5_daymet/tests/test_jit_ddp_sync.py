#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_ddp_sync.py — train_jit 训练接线的 2-rank 梯度同步冒烟
============================================================================
用法(纯 CPU, gloo 后端):
  torchrun --nproc_per_node=2 -m era5_daymet.tests.test_jit_ddp_sync

按 train_jit 的接线(损失前向走 DDP 包装体, broadcast_buffers=False,
MoE 时 find_unused_parameters=True)各 rank 喂不同数据连跑数步, 验证:

  1) backward 后梯度指纹全 rank 一致(含梯度缺席模式 —— MoE 专家非全命中);
  2) optimizer.step + EMA 后参数与 EMA 指纹全 rank 一致;
  3) 各 rank 本地 loss 互不相同(确实在喂不同数据, 检验不是假阳性);
  4) MoE 负载计数逐 rank 互异(未被 buffer 广播覆盖), all-reduce 聚合正确;
  5) 人为把损失前向改成裸网络(绕过 DDP)后, 首步自检必须当场报错 ——
     该绕过会让梯度同步静默失效, 训练照常收敛但多卡等效单卡。
============================================================================
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from era5_daymet.models.jit_backbone import JiT
from era5_daymet.training.train_jit import (_assert_ranks_synced, _drain_moe_load,
                                            _fingerprint, jit_vloss)

MC = {"num_experts": 8, "moe_intermediate_size": 16, "num_experts_per_tok": 2,
      "n_group": 2, "topk_group": 2, "routed_scaling_factor": 2.5,
      "interleave": True, "use_shared_expert": True, "proj_drop": 0.0}


def make_batch(rank, step, scale):
    g = torch.Generator().manual_seed(1000 * rank + step)
    cond = torch.randn(2, 5, 16, 24, generator=g) * scale
    tgt = torch.randn(2, 1, 16, 24, generator=g)
    land = (torch.rand(2, 1, 16, 24, generator=g) > 0.3).float()
    return cond, tgt * land, land


def run_phase(label, moe, rank, world, bypass=False, steps=3):
    torch.manual_seed(0)                       # 各 rank 同初始权重(DDP 构造时亦会广播)
    net = JiT(hw=(16, 24), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
              num_heads=2, bottleneck=8, moe_config=MC if moe else None)
    model = DDP(net, find_unused_parameters=moe, broadcast_buffers=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95))
    ema = {n: p.detach().clone() for n, p in net.named_parameters()}
    gen = torch.Generator().manual_seed(7 + rank)

    for step in range(1, steps + 1):
        cond, tgt, land = make_batch(rank, step, scale=1.0 + 4.0 * rank)
        fwd = model.module if bypass else model
        loss = jit_vloss(fwd, tgt, cond, land, 1.0, -0.8, 0.8, 0.05, generator=gen)
        opt.zero_grad(); loss.backward()
        _assert_ranks_synced(_fingerprint(net, grads=True), "cpu", world,
                             f"{label} step{step} 梯度")
        opt.step()
        for n, p in net.named_parameters():
            ema[n].mul_(0.999).add_(p.detach(), alpha=0.001)
        _assert_ranks_synced(_fingerprint(net), "cpu", world,
                             f"{label} step{step} 参数/buffer")
        ema_fp = torch.tensor([float(v.double().sum()) for v in ema.values()],
                              dtype=torch.float64)
        _assert_ranks_synced(ema_fp, "cpu", world, f"{label} step{step} EMA")

        lv = torch.zeros(world); lv[rank] = float(loss.detach())
        dist.all_reduce(lv)
        assert lv.unique().numel() == world, f"{label}: 各 rank loss 相同, 检验无效"

    if moe and not bypass:
        layers = net.moe_layers()
        local = torch.stack([m.load_acc.clone() for m in layers])
        others = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(others, local)
        assert not torch.equal(others[0], others[1]), \
            "各 rank 负载计数完全相同 —— 疑似被 buffer 广播覆盖"
        total = _drain_moe_load(layers, is_dist=True)
        assert torch.equal(total, others[0] + others[1]), "负载 all-reduce 聚合错误"
        expect = len(layers) * world * steps * 2 * 24 * MC["num_experts_per_tok"]
        assert int(total.sum()) == expect, \
            f"负载总数 {int(total.sum())} != 层数x步数x token 数x K = {expect}"
    return True


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    ok = lambda s: rank == 0 and print(f"  [PASS] {s}", flush=True)

    run_phase("dense", moe=False, rank=rank, world=world)
    ok("dense 接线: 梯度/参数/EMA 全 rank 一致, 各 rank loss 互异")
    run_phase("moe", moe=True, rank=rank, world=world)
    ok("moe 接线: 同上, 且负载计数逐 rank 独立并正确聚合")

    caught = False
    try:
        run_phase("bypass", moe=False, rank=rank, world=world, bypass=True, steps=1)
    except RuntimeError:
        caught = True
    flag = torch.tensor([1.0 if caught else 0.0]); dist.all_reduce(flag)
    assert flag.item() == world, "绕过 DDP 包装体未被首步自检捕获 —— 检验自身失效"
    ok("绕过 DDP 的人为错误被首步自检当场捕获")

    dist.barrier()
    if rank == 0:
        print("test_jit_ddp_sync: 全部通过")


if __name__ == "__main__":
    main()
