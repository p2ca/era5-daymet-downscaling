#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_ddp_sync.py — train_jit 训练接线的 2-rank 梯度同步冒烟
============================================================================
用法(纯 CPU, gloo 后端):
  torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_jit_ddp_sync

按 train_jit 的接线(损失前向走 DDP 包装体, broadcast_buffers=False,
MoE 时 find_unused_parameters=True)各 rank 喂不同数据连跑数步, 验证(dense / moe / dec / 两条流全开 四档):

  1) backward 后梯度指纹全 rank 一致(含梯度缺席模式 —— MoE 专家非全命中);
  2) optimizer.step + EMA 后参数与 EMA 指纹全 rank 一致;
  3) 各 rank 本地 loss 互不相同(确实在喂不同数据, 检验不是假阳性);
  4) MoE 负载计数逐 rank 互异(未被 buffer 广播覆盖), all-reduce 聚合正确;
  5) 人为把损失前向改成裸网络(绕过 DDP)后, 首步自检必须当场报错 ——
     该绕过会让梯度同步静默失效, 训练照常收敛但多卡等效单卡。
============================================================================
"""
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from downscaling_4x import contract as C
from downscaling_4x.models.jit_backbone import JiT
from downscaling_4x.training.train_jit import (_assert_ranks_synced, _drain_moe_load,
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


DEC = {"pool": True, "drop": True, "prior": True, "prior_max": 1.0, "prior_ramp": 0.0,
       "head_layer": 1}


TWO = {"rope_units": "km", "drop_outside": 1, "two_stream": 1, "coarse_doy": 1, "coarse_conv": 1,
       "coarse_blocks": 2, "cross_attn": 1, "cross_modulate": 1, "cross_mask_outside": 1,
       "lagrangian": 1, "tau_spec": "0,850:6", "expert_two_card": 1, "dense_two_card": 1,
       "two_card_dims": (12, 20), "wards": 2, "ward_topk": 1, "ward_key": 1,
       "terrain_key": 1, "terrain_key_dim": 8, "terrain_key_window": 6}


def make_batch_two(rank, step, scale):
    """两条流按通道名拆 cond, 因此用合同的完整 51 通道布局。"""
    g = torch.Generator().manual_seed(1000 * rank + step)
    cond = torch.randn(2, C.cond_channels(C.DEFAULT_MODE), 16, 24, generator=g) * scale
    tgt = torch.randn(2, 1, 16, 24, generator=g)
    land = (torch.rand(2, 1, 16, 24, generator=g) > 0.3).float()
    return cond, tgt * land, land


def run_phase(label, moe, rank, world, bypass=False, steps=3, dec=False, two=False):
    torch.manual_seed(0)                       # 各 rank 同初始权重(DDP 构造时亦会广播)
    mc = ({**MC, "router_mode": "dec", "dec": DEC} if dec else MC) if moe else None
    if two:
        # 两条流全开: 粗流 + 交叉问询(按风推移) + 两张卡 + 两级分诊(科室键、地形键) + 域外舍弃
        mc = {**MC, "n_group": 2, "topk_group": 1, "terrain_key_dim": 8, "drop_outside": True}
        net = JiT(hw=(16, 24), patch=4, cond_ch=C.cond_channels(C.DEFAULT_MODE), out_ch=1, hidden=32,
                  depth=4, num_heads=2, bottleneck=8, moe_config=mc, patch_margin=1, arch=TWO)
        ev = np.ones((16, 24), dtype=np.uint8); ev[:, :6] = 0
        net.set_data_constants(np.zeros(4), np.ones(4), ev)
    else:
        net = JiT(hw=(16, 24), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
                  num_heads=2, bottleneck=8, moe_config=mc)
    if dec:
        net.set_dec_progress(1.0)              # λ 升满, 难度先验真正参与选择
    model = DDP(net, find_unused_parameters=moe, broadcast_buffers=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95))
    ema = {n: p.detach().clone() for n, p in net.named_parameters()}
    gen = torch.Generator().manual_seed(7 + rank)

    for step in range(1, steps + 1):
        cond, tgt, land = (make_batch_two if two else make_batch)(rank, step, scale=1.0 + 4.0 * rank)
        fwd = model.module if bypass else model
        if two:
            loss = jit_vloss(fwd, tgt, cond, land, 1.0, -0.8, 0.8, 0.05, generator=gen,
                             patch=4, domain_mask=land)
        elif dec:
            loss, aux, extra = jit_vloss(fwd, tgt, cond, land, 1.0, -0.8, 0.8, 0.05, generator=gen,
                                         patch=4, domain_mask=land, dec_aux=True)
            assert extra and torch.isfinite(aux), "D-EC 辅助损失缺失或非有限"
            loss = loss + aux
        else:
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

    if moe and not bypass and not two:
        layers = net.moe_layers()
        local = torch.stack([m.load_acc.clone() for m in layers])
        others = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(others, local)
        # 专家挑 token 的 dec 档每专家名额由域内 token 数决定, 逐 rank 天然相同, 不据此判广播
        if not dec:
            assert not torch.equal(others[0], others[1]), \
                "各 rank 负载计数完全相同 —— 疑似被 buffer 广播覆盖"
        total = _drain_moe_load(layers, is_dist=True)
        assert torch.equal(total, others[0] + others[1]), "负载 all-reduce 聚合错误"
        # token 数取模型自报的切块网格: 随机起点要求网格补到能容下任意起点,
        # 因此它不等于 H/patch x W/patch。
        n_tok = net.x_embedder.gh * net.x_embedder.gw
        expect = len(layers) * world * steps * 2 * n_tok * MC["num_experts_per_tok"]
        if dec:
            # drop 成分只给域内 token 名额: 总数必须小于 ec/tc 的满额, 且 D-EC 统计各 rank 互异
            assert int(total.sum()) < expect, "D-EC 的 drop 成分未减少路由名额"
            st = net.pop_dec_stats()
            others_st = [torch.empty_like(st) for _ in range(world)]
            dist.all_gather(others_st, st)
            assert not torch.equal(others_st[0], others_st[1]), "D-EC 统计各 rank 完全相同"
        else:
            assert int(total.sum()) == expect, \
                f"负载总数 {int(total.sum())} != 层数x步数x token 数({n_tok})x K = {expect}"
    return True


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    ok = lambda s: rank == 0 and print(f"  [PASS] {s}", flush=True)

    run_phase("dense", moe=False, rank=rank, world=world)
    ok("dense 接线: 梯度/参数/EMA 全 rank 一致, 各 rank loss 互异")
    run_phase("moe", moe=True, rank=rank, world=world)
    ok("moe 接线: 同上, 且负载计数逐 rank 独立并正确聚合")
    run_phase("dec", moe=True, rank=rank, world=world, dec=True)
    ok("dec 接线: 掩膜与难度头随前向, 头的参数/梯度全 rank 一致, 名额少于满额且统计逐 rank 独立")
    run_phase("two-stream", moe=True, rank=rank, world=world, two=True)
    ok("两条流全开接线: 粗流/交叉问询/两张卡/两级分诊/地形键的梯度、参数与 buffer 全 rank 一致")

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
