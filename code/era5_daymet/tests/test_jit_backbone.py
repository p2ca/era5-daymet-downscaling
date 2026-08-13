#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_backbone.py — JiT 主干的几何与初始化契约
============================================================================
矩形网格下的 2D RoPE 平移相对性、patchify/unpatchify 往返、adaLN-zero 的零初始输出、
MoE 层交替位置、dropout 中段规则与参数计量。

Run: python -m era5_daymet.tests.test_jit_backbone
============================================================================
"""
import torch

from era5_daymet.models.jit_backbone import JiT, Rope2D, rotate_half
from era5_daymet.models.moe_ffn import DSMoE

torch.manual_seed(0)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def rope_at(rope, vec, idx):
    return vec * rope.cos[idx] + rotate_half(vec) * rope.sin[idx]


def main():
    # --- 2D RoPE: 注意力分数只依赖相对位移 (矩形网格) ---
    gh, gw, hd = 5, 7, 16
    rope = Rope2D(hd, gh, gw)
    q, k = torch.randn(hd), torch.randn(hd)
    def score(p_q, p_k):
        return float(rope_at(rope, q, p_q[0] * gw + p_q[1])
                     @ rope_at(rope, k, p_k[0] * gw + p_k[1]))
    s1 = score((1, 1), (2, 4))
    s2 = score((2, 2), (3, 5))          # 同 (Δh, Δw)=(1, 3)
    s3 = score((0, 3), (1, 6))
    check("平移不变: 同位移分数一致", abs(s1 - s2) < 1e-4 and abs(s1 - s3) < 1e-4)
    s4 = score((1, 1), (4, 2))          # 不同位移
    check("不同位移分数不同", abs(s1 - s4) > 1e-3)

    # --- 模型: 矩形 24x32, patch 4 ---
    net = JiT(hw=(24, 32), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
              num_heads=2, bottleneck=8)
    z = torch.randn(2, 1, 24, 32)
    cond = torch.randn(2, 5, 24, 32)
    t = torch.rand(2)
    out = net(z, t, cond)
    check("输出形状 (B,1,H,W)", out.shape == (2, 1, 24, 32))
    check("零初始化: 初始 x 预测恒为 0", bool((out == 0).all()))

    img = torch.randn(2, 1, 24, 32)
    p, ghh, gww = 4, 6, 8
    pat = img.reshape(2, 1, ghh, p, gww, p)
    pat = torch.einsum("nchpwq->nhwpqc", pat).reshape(2, ghh * gww, p * p * 1)
    check("unpatchify 与 patchify 互逆", torch.allclose(net.unpatchify(pat), img))

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss = ((net(z, t, cond) - torch.randn_like(z)) ** 2).mean()
    loss.backward(); opt.step()
    check("一步优化后输出脱离零", bool((net(z, t, cond) != 0).any()))

    # --- MoE 交替位置与参数计量 ---
    mc = {"num_experts": 4, "moe_intermediate_size": 16, "num_experts_per_tok": 2,
          "n_group": 2, "topk_group": 2, "routed_scaling_factor": 2.5,
          "interleave": True, "use_shared_expert": True, "proj_drop": 0.0}
    nm = JiT(hw=(24, 32), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
             num_heads=2, bottleneck=8, moe_config=mc)
    kinds = [isinstance(b.mlp, DSMoE) for b in nm.blocks]
    check("MoE 只在奇数索引块 (第 0 块稠密)", kinds == [False, True, False, True])
    nm_all = JiT(hw=(24, 32), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
                 num_heads=2, bottleneck=8, moe_config={**mc, "interleave": False})
    check("interleave=False 时全部块为 MoE",
          all(isinstance(b.mlp, DSMoE) for b in nm_all.blocks))
    pc, pd = nm.param_counts(), net.param_counts()
    check("MoE 激活参数 < 总参数; 稠密两者相等",
          pc["activated"] < pc["total"] and pd["activated"] == pd["total"])
    out_m = nm(z, t, cond)
    check("MoE 前向形状与零初始化一致",
          out_m.shape == (2, 1, 24, 32) and bool((out_m == 0).all()))

    # --- dropout 中段规则 ---
    nd = JiT(hw=(24, 32), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=8,
             num_heads=2, bottleneck=8, attn_drop=0.5, proj_drop=0.5)
    on = [b.attn.attn_drop for b in nd.blocks]
    check("dropout 只作用于中段块", on == [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])

    print("test_jit_backbone: 全部通过")


if __name__ == "__main__":
    main()
