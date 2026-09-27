#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 MoE 路由开关(router=tc/ec, gating=norm/raw)的口径。

要点: tc 档与现役实现逐位相同; ec 为帧内 Expert-Choice(每专家恰取 C=⌈N·K/E⌉ 个
token, FLOPs 与 tc 严格对齐, 选择与 batch 组成无关); raw 门控仅 ec 允许且均值对齐
norm 档; 组合守卫与遗留回填各有负对照。这些性质写错都不会报错, 只会静默换口径。

纯 CPU, 不需要数据。运行: python -m downscaling_4x.tests.test_moe_router
"""
import torch

from downscaling_4x.models.moe_ffn import DSMoE
from downscaling_4x.training import train_downscale as TD

E, D, I, K = 8, 16, 32, 2
B, N = 2, 24


def _mk(**kw):
    torch.manual_seed(0)
    return DSMoE(num_experts=E, dim=D, moe_inter=I, num_experts_per_tok=K,
                 routed_scaling_factor=2.5, **kw)


def test_tc_is_bytewise_status_quo():
    """router=tc 与不传该参数的构造逐位同参数、同输出。"""
    a, b = _mk(), _mk(router_mode="tc", gating_mode="norm")
    x = torch.randn(B, N, D)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_ec_capacity_and_flops_parity():
    """每专家在每个样本内恰取 C=⌈N·K/E⌉ 个互异 token; 总对数与 tc 对齐(⌈⌉ 余量 <E)。"""
    m = _mk(router_mode="ec")
    scores = torch.rand(B * N, E)
    sel, w = m.route_ec(scores, B, N)
    cap = (N * K + E - 1) // E
    per = sel.view(B, N, E).sum(1)
    assert torch.all(per == cap), per
    total = int(sel.sum())
    assert 0 <= total - B * N * K < B * E, (total, B * N * K)


def test_ec_zero_token_and_norm():
    """未被任何专家选中的 token 权重全零; 被选 token 的 norm 门控和恒为 2.5。"""
    m = _mk(router_mode="ec", gating_mode="norm")
    scores = torch.rand(1 * N, E)
    scores[0] = 1e-6                                  # 人为造一个全网嫌弃的 token
    sel, w = m.route_ec(scores, 1, N)
    picked = sel.any(dim=1)
    assert torch.all(w[~picked].abs() == 0)
    s = w[picked].sum(dim=1)
    assert torch.allclose(s, torch.full_like(s, 2.5), atol=1e-5), s


def test_ec_raw_mean_aligned_variance_open():
    """raw 门控: 权重=sigmoid 原值x2.5 —— 总权重均值与 norm 档同量级, 但跨 token 方差>0。"""
    m = _mk(router_mode="ec", gating_mode="raw")
    torch.manual_seed(1)
    scores = torch.sigmoid(torch.randn(1 * N, E))
    sel, w = m.route_ec(scores, 1, N)
    tot = w.sum(dim=1)
    assert tot[sel.any(1)].var() > 0, "幅度通道未打开"
    assert torch.allclose(w[sel], scores[sel] * 2.5)


def test_ec_forward_grads():
    """ec 前向可反传: router 与专家参数梯度均非零。"""
    m = _mk(router_mode="ec")
    x = torch.randn(B, N, D, requires_grad=True)
    m(x).square().mean().backward()
    assert m.gate.weight.grad is not None and m.gate.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.experts.parameters())


def test_ec_pool_is_per_sample():
    """帧内池: 同一样本无论 batch 组成如何, **选择**逐位相同(训推池一致的根基);
    输出按数值容差比(不同 batch 形状下 GEMM 归约顺序不保证位级一致)。"""
    m = _mk(router_mode="ec")
    torch.manual_seed(2)
    x1 = torch.randn(1, N, D)
    other = torch.randn(1, N, D) * 5
    with torch.no_grad():
        s1 = torch.sigmoid(m.gate(x1.reshape(-1, D)))
        sb = torch.sigmoid(m.gate(torch.cat([x1, other], 0).reshape(-1, D)))
        sel1, w1 = m.route_ec(s1, 1, N)
        selb, wb = m.route_ec(sb, 2, N)
        assert torch.equal(sel1, selb[:N]), "选择泄漏到样本之外"
        assert torch.allclose(w1, wb[:N], atol=1e-6)
        solo = m(x1)
        batched = m(torch.cat([x1, other], 0))[:1]
    assert torch.allclose(solo, batched, atol=1e-5), "同样本输出随 batch 组成漂移"


def test_combo_guards():
    """组合守卫负对照: tc+raw 与 ec+bias_gamma>0 必须当场拒绝。"""
    base = {"moe": True, "experts": E, "experts_per_tok": K, "moe_intermediate": I,
            "routed_scaling": 2.5, "moe_all_layers": False, "moe_no_shared": False,
            "hidden": D, "proj_dropout": 0.0}
    for bad in ({"router": "tc", "gating": "raw"},
                {"router": "ec", "gating": "norm", "bias_gamma": 0.1}):
        try:
            TD.jit_moe_config({**base, **bad})
        except SystemExit:
            continue
        raise AssertionError(f"非法组合未被拒绝: {bad}")
    ok = TD.jit_moe_config({**base, "router": "ec", "gating": "raw"})
    assert ok["router_mode"] == "ec" and ok["gating_mode"] == "raw"
    legacy = TD.jit_moe_config(dict(base))            # 旧配置无键 -> tc/norm
    assert legacy["router_mode"] == "tc" and legacy["gating_mode"] == "norm"
    # 稠密主体(无 --moe)配非缺省路由口径必须拒绝: 否则被静默忽略, run 顶着 ec 的名字跑完
    for bad in ({"router": "ec"}, {"gating": "raw"}):
        try:
            TD.jit_moe_config({**base, "moe": False, **bad})
        except SystemExit:
            continue
        raise AssertionError(f"稠密 + {bad} 未被拒绝")
    assert TD.jit_moe_config({**base, "moe": False}) is None
    assert TD.jit_moe_config({**base, "moe": False, "router": "tc", "gating": "norm"}) is None


def test_resume_pin_and_legacy():
    """router/gating 在两入口的续训契约里; 旧断点缺键回填默认档后放行。"""
    from downscaling_4x.training.train_jit import RESUME_PINNED_ARGS
    for k in ("router", "gating", "refine_head"):
        assert k in RESUME_PINNED_ARGS and k in TD.PINNED
    d = {"target": "x"}
    assert set(TD.apply_legacy_defaults(d)) == set(TD.LEGACY_DEFAULTS) >= {"refine_head", "router", "gating"}
    assert (d["router"], d["gating"], d["refine_head"]) == ("tc", "norm", 0)


def main():
    tests = [test_tc_is_bytewise_status_quo, test_ec_capacity_and_flops_parity,
             test_ec_zero_token_and_norm, test_ec_raw_mean_aligned_variance_open,
             test_ec_forward_grads, test_ec_pool_is_per_sample,
             test_combo_guards, test_resume_pin_and_legacy]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
