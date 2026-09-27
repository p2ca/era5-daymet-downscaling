#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 JiT 残差阶段 B(JDB/JMB)的三处口径: 残差尺度、有效域外置零、μ 进条件。

这三件事写错都**不会报错**: 尺度错了只是噪声调度偏掉、训不好; 域外没置零只是让扩散网去
拟合一片无意义的值并经感受野污染域内边缘; μ 没进条件只是少了一路信息而通道数照样对得上
(因为宽度是按同一个式子算的)。所以逐条独立对拍, 并给出负对照。

纯 CPU, 不需要数据。

运行: python -m downscaling_4x.tests.test_jit_stage_b
"""
import torch

from downscaling_4x import contract as C
from downscaling_4x.training.train_jit import make_residual

B, CH, H, W = 2, 6, 8, 12


def _case(sigma_r=0.25):
    torch.manual_seed(0)
    y = torch.randn(B, 1, H, W)
    mu = torch.randn(B, 1, H, W)
    land = (torch.rand(B, 1, H, W) > 0.4).float()
    cond = torch.randn(B, CH, H, W)
    return y * land, mu, land, cond, sigma_r          # y 已在域外置零(与训练侧一致)


def test_residual_is_zero_outside_domain():
    y, mu, land, cond, sr = _case()
    r, _ = make_residual(y, mu, land, cond, sr)
    off = land < 0.5
    assert float(r[off].abs().max()) == 0.0, "有效域外残差必须恒为 0"
    assert off.any(), "构造里必须有域外像素, 否则本条没有分辨力"


def test_residual_matches_independent_formula():
    y, mu, land, cond, sr = _case()
    r, _ = make_residual(y, mu, land, cond, sr)
    want = (y - mu * land) / sr                        # 独立重算
    assert torch.allclose(r, want, atol=1e-6)


def test_sigma_r_roundtrip_recovers_y():
    """y = μ + σ_r·r 必须在有效域上还原原值; 采样端就是这么加回去的。"""
    y, mu, land, cond, sr = _case(sigma_r=0.137)
    r, _ = make_residual(y, mu, land, cond, sr)
    back = mu * land + sr * r
    on = land > 0.5
    assert torch.allclose(back[on], y[on], atol=1e-5)


def test_sigma_r_actually_scales():
    """负对照: 不同 σ_r 必须给出不同残差, 否则"归一"是空操作。"""
    y, mu, land, cond, _ = _case()
    r1, _ = make_residual(y, mu, land, cond, 1.0)
    r2, _ = make_residual(y, mu, land, cond, 0.25)
    assert not torch.allclose(r1, r2), "σ_r 没有起作用"
    assert torch.allclose(r2, r1 / 0.25, atol=1e-6)


def test_cond_gains_exactly_mu():
    y, mu, land, cond, sr = _case()
    _, c = make_residual(y, mu, land, cond, sr)
    assert c.shape[1] == cond.shape[1] + 1, "条件应恰好多一个通道"
    assert torch.allclose(c[:, :-1], cond), "原有条件通道不得被改动"
    assert torch.allclose(c[:, -1:], mu * land), "新增通道应是域外钉零后的 μ"


def test_stage_b_cond_width_is_layout_plus_one():
    """阶段 B 的条件宽度 = 合同通道数 + 1(μ)。两处各算各的差一个就装不回权重。"""
    for mode in C.MODES:
        y, mu, land, cond, sr = _case()
        cond = torch.randn(B, C.cond_channels(mode), H, W)
        _, c = make_residual(y, mu, land, cond, sr)
        assert c.shape[1] == C.cond_channels(mode) + 1, mode


def test_unpinned_mu_would_leak_outside_domain():
    """负对照: 若不把 μ 在域外钉零, 残差在域外就不是 0 —— 证明那一步不是摆设。"""
    y, mu, land, cond, sr = _case()
    bad = (y - mu) / sr                                # 少了 * land
    off = land < 0.5
    assert float(bad[off].abs().max()) > 0.0, "构造的 μ 在域外应非零, 否则本条无分辨力"
    good, _ = make_residual(y, mu, land, cond, sr)
    assert float(good[off].abs().max()) == 0.0
    assert not torch.allclose(bad, good), "两种做法必须不同"


def main():
    tests = [
        test_residual_is_zero_outside_domain,
        test_residual_matches_independent_formula,
        test_sigma_r_roundtrip_recovers_y,
        test_sigma_r_actually_scales,
        test_cond_gains_exactly_mu,
        test_stage_b_cond_width_is_layout_plus_one,
        test_unpinned_mu_would_leak_outside_domain,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
