#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_moe.py — DSMoE 稀疏 FFN 的数值对拍与路由不变量
============================================================================
参考实现按逐 token 逐专家的朴素循环独立写出(与向量化派发无共享代码路径), 两者在同一
份权重上必须逐元素一致。另验证: 偏置只进选择不进门控、门控归一、负载计数与免辅助
损失偏置更新方向、无共享专家时的残差行为。

Run: python -m era5_daymet.tests.test_jit_moe
============================================================================
"""
import torch

from era5_daymet.models.moe_ffn import DSMoE

torch.manual_seed(0)
E, D, S, K = 8, 32, 48, 2


def naive_forward(m, x):
    """逐 token 朴素参考: 与 DSMoE.forward 相同语义, 不同实现路径。"""
    B, N, d = x.shape
    flat = x.reshape(-1, d)
    scores = torch.sigmoid(flat.float() @ m.gate.weight.float().t())
    bias = m.gate.e_score_correction_bias
    out = torch.zeros_like(flat)
    for i in range(flat.shape[0]):
        s = scores[i]
        choice = torch.topk(s + bias, m.top_k).indices
        w = s[choice]
        w = w / (w.sum() + 1e-20) * m.routed_scaling_factor
        for j, e in enumerate(choice.tolist()):
            out[i] += m.experts[e](flat[i][None])[0] * w[j]
    out = out.view(B, N, d)
    if m.use_shared_expert:
        return out + m.shared_experts(x)
    return out + x


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def main():
    x = torch.randn(2, 7, D)

    m = DSMoE(E, D, S, num_experts_per_tok=K)
    y_fast, y_ref = m(x), naive_forward(m, x)
    check("对拍: shared=True 逐元素一致",
          torch.allclose(y_fast, y_ref, atol=1e-5, rtol=1e-5))

    m0 = DSMoE(E, D, S, num_experts_per_tok=K, use_shared_expert=False)
    y_fast, y_ref = m0(x), naive_forward(m0, x)
    check("对拍: shared=False(带输入残差)一致",
          torch.allclose(y_fast, y_ref, atol=1e-5, rtol=1e-5))

    flat = x.reshape(-1, D)
    scores = torch.sigmoid(m.gate(flat))
    idx, w = m.route(scores)
    check("门控在 routed_scaling 前归一为 1",
          torch.allclose((w / m.routed_scaling_factor).sum(-1),
                         torch.ones(flat.shape[0]), atol=1e-5))
    manual = torch.topk(scores + m.gate.e_score_correction_bias, K, dim=-1).indices
    check("top-K 选择与手工计算一致",
          all(set(a.tolist()) == set(b.tolist()) for a, b in zip(idx, manual)))

    mb = DSMoE(E, D, S, num_experts_per_tok=K)
    mb.gate.e_score_correction_bias[0] = 100.0
    idx_b, w_b = mb.route(torch.sigmoid(mb.gate(flat)))
    check("大偏置强制专家 0 全被选", bool((idx_b == 0).any(dim=-1).all()))
    s0 = torch.sigmoid(mb.gate(flat)).gather(1, idx_b)
    w_expect = s0 / (s0.sum(-1, keepdim=True) + 1e-20) * mb.routed_scaling_factor
    check("偏置不进入门控权重(仍由原始亲和分归一)",
          torch.allclose(w_b, w_expect, atol=1e-6))

    m.pop_load()
    m(x); m(x)
    load = m.pop_load()
    check("负载计数 = 2 次前向 x token 数 x K",
          int(load.sum()) == 2 * flat.shape[0] * K)
    check("pop_load 后清零", int(m.load_acc.sum()) == 0)

    b_before = m.gate.e_score_correction_bias.clone()
    counts = torch.zeros(E); counts[0] = 100.0
    m.update_bias(0.01, counts)
    db = m.gate.e_score_correction_bias - b_before
    check("偏置更新方向: 过载减、欠载增",
          db[0] < 0 and (db[1:] > 0).all())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        yb = m(x)
    check("bf16 autocast 下前向有限", bool(torch.isfinite(yb).all()))

    print("test_jit_moe: 全部通过")


if __name__ == "__main__":
    main()
