#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_routing_dump.py — 路由截获的回归测试(纯 CPU)
============================================================================
钉住四件事, 任何一件坏了都是静默错误(落盘照常写、数字照常有):

  1. 补丁只读: 装上截获前后, 三种路由(tc / ec / dec)的前向输出逐位相同
  2. 计数不变量: tc 下每 token 每次前向恰好 K 次选中; ec / dec 下每次前向的选中总数等于容量
  3. dec 自检: 由截获反推的域内 token 专家数直方图与 DSMoE 自己累计的 dec_acc 逐元素相等
  4. 输出范数钩子: 按截获还原的 token 顺序与 DSMoE 派发顺序一致, 钩子算出的
     ‖w·f_e(x)‖/‖x‖ 与手工逐专家重算的相同(tc 与 dec 各一遍)

另测 npz 落盘往返的 dtype 与数值, 以及 tok_to_pixel 的裁剪。
运行: python -m downscaling_4x.tests.test_routing_dump
============================================================================
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.models import moe_ffn
from downscaling_4x.models.moe_ffn import DSMoE


class Tiny(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def moe_layers(self):
        return list(self.layers)

    def forward(self, x, tok_mask=None, prior=None):
        for m in self.layers:
            x = m(x, tok_mask, prior) if m.router_mode == "dec" else m(x)
        return x


def make_layer(mode, E=4, d=16, inter=24, K=2, seed=0):
    torch.manual_seed(seed)
    cfg = {"pool": 0, "drop": 1, "prior": 1} if mode == "dec" else None
    m = DSMoE(E, d, inter, num_experts_per_tok=K, n_group=2, topk_group=2, router_mode=mode, dec_config=cfg)
    if mode == "dec":
        m.dec_lambda = 1.0
        m.set_dec_eval("frame", 1.0)
    return m.eval()


def inputs(B=2, N=12, d=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, d, generator=g)
    tok_mask = torch.rand(B * N, generator=g) > 0.3
    tok_mask[0] = True
    prior = torch.randn(B * N, generator=g)
    return x, tok_mask, prior


def run(net, x, tok_mask, prior, mode):
    with torch.no_grad():
        return net(x, tok_mask, prior) if mode == "dec" else net(x)


def test_patch_is_read_only_and_counts():
    for mode in ("tc", "ec", "dec"):
        net = Tiny([make_layer(mode, seed=0), make_layer(mode, seed=1)])
        x, tok_mask, prior = inputs()
        ref = run(net, x, tok_mask, prior, mode)
        cap = RD.RoutingCapture(net, norms=False).install()
        try:
            got = run(net, x, tok_mask, prior, mode)
            assert torch.equal(ref, got), f"{mode}: 装补丁后前向输出变了"
            E, K = net.layers[0].n_experts, net.layers[0].top_k
            T = x.shape[0] * x.shape[1]
            acc = RD.RoutingAccumulator(2, T, E, "cpu", scores=True, norms=False)
            cap.acc = acc
            for t in (0.1, 0.5, 0.9):
                run(net, x, tok_mask, prior, mode)
                acc.add_forward(t, cap)
            assert acc.nfwd == 3 and int(acc.nfwd_t.sum()) == 3
            if mode == "tc":
                acc.check("tc", K)
                assert bool((acc.sel_cnt.sum(-1) == K * 3).all())
            else:
                per_fwd = acc.sel_cnt.sum((1, 2)) / 3                    # 每层每次前向的选中总数
                if mode == "ec":
                    cap_tok = math.ceil(x.shape[1] * K / E) * E * x.shape[0]
                else:
                    cap_tok = sum(min(int(tok_mask.view(x.shape[0], -1)[b].sum()),
                                      math.ceil(int(tok_mask.view(x.shape[0], -1)[b].sum()) * K / E)) * E
                                  for b in range(x.shape[0]))
                assert bool((per_fwd == cap_tok).all()), f"{mode}: 选中总数 {per_fwd.tolist()} != 容量 {cap_tok}"
            assert bool((acc.sel_cnt_t.sum(1) == acc.sel_cnt).all())
        finally:
            cap.uninstall()
        assert moe_ffn.DSMoE.route.__name__ == "route", "补丁未复原"
        print(f"  [{mode}] 只读补丁与计数不变量 OK")


def test_dec_histogram_matches_dec_acc():
    net = Tiny([make_layer("dec", seed=3)])
    x, tok_mask, prior = inputs(seed=5)
    m = net.layers[0]
    m.pop_dec_stats()
    E = m.n_experts
    cap = RD.RoutingCapture(net, norms=False).install()
    try:
        hist = torch.zeros(E + 1, dtype=torch.float64)
        for t in (0.2, 0.6):
            run(net, x, tok_mask, prior, "dec")
            kv = cap.sel[0][tok_mask].sum(1)
            hist += torch.bincount(kv, minlength=E + 1)[:E + 1].double()
        ref = m.dec_acc[:E + 1].double()
        assert torch.equal(hist, ref), f"直方图 {hist.tolist()} != dec_acc {ref.tolist()}"
    finally:
        cap.uninstall()
    print("  [dec] 截获直方图与 dec_acc 逐元素相等 OK")


def test_norm_hook_matches_manual():
    for mode in ("tc", "dec"):
        net = Tiny([make_layer(mode, seed=7)])
        m = net.layers[0]
        x, tok_mask, prior = inputs(seed=9)
        E = m.n_experts
        T = x.shape[0] * x.shape[1]
        cap = RD.RoutingCapture(net, norms=True).install()
        try:
            acc = RD.RoutingAccumulator(1, T, E, "cpu", scores=False, norms=True)
            cap.acc = acc
            run(net, x, tok_mask, prior, mode)
            acc.add_forward(0.5, cap)
        finally:
            cap.uninstall()            # 手工重算前必须卸载: 截获在位时任何专家调用都会被钩子累加
        flat = x.reshape(-1, x.shape[-1])
        manual = torch.zeros(T, E)
        with torch.no_grad():
            for e in range(E):
                tok = cap.sel[0][:, e].nonzero().flatten()
                if tok.numel() == 0:
                    continue
                y = m.experts[e](flat[tok]) * cap.w[0][tok, e, None]
                manual[tok, e] = y.norm(dim=1) / flat[tok].norm(dim=1)
        assert torch.allclose(acc.norm_sum[0], manual, atol=1e-5), f"{mode}: 输出范数钩子与手工重算不符"
        assert bool(((acc.norm_sum[0] > 0) == cap.sel[0]).all()), f"{mode}: 范数非零位置与选择掩膜不一致"
        print(f"  [{mode}] 输出范数钩子 OK")


def test_save_load_roundtrip_and_tok_to_pixel():
    net = Tiny([make_layer("tc", seed=2)])
    x, tok_mask, prior = inputs(seed=4)
    E = net.layers[0].n_experts
    T = x.shape[0] * x.shape[1]
    cap = RD.RoutingCapture(net, norms=True).install()
    try:
        acc = RD.RoutingAccumulator(1, T, E, "cpu")
        cap.acc = acc
        for t in (0.3, 0.7):
            run(net, x, tok_mask, prior, "tc")
            acc.add_forward(t, cap)
    finally:
        cap.uninstall()
    arrays = acc.to_numpy((3, 5), tok_mask.numpy())
    assert arrays["sel_cnt"].dtype == np.uint8 and arrays["k_sum"].dtype == np.uint16 and arrays["gate_sum"].dtype == np.float16
    with tempfile.TemporaryDirectory() as td:
        RD.save_routing(td, 2020, 7, 0, arrays)
        got = RD.load_routing(td, 2020, 7, 0)
        assert got["nfwd"] == 2 and tuple(got["offset"]) == (3, 5)
        assert np.array_equal(got["sel_cnt"], acc.sel_cnt.numpy())
        assert np.allclose(got["gate_sum"], acc.gate_sum.numpy(), rtol=1e-2)
        assert RD.available_routing(td) == [(2020, 7, 0)]
    tok = np.arange(6, dtype=float).reshape(1, 6)                 # gh=2, gw=3 的 token 场
    px = RD.tok_to_pixel(tok, 2, 3, 2, (1, 1), 3, 5)
    expect = np.repeat(np.repeat(tok.reshape(2, 3), 2, 0), 2, 1)[1:4, 1:6]
    assert px.shape == (1, 3, 5) and np.array_equal(px[0], expect)
    print("  落盘往返与 tok_to_pixel OK")


if __name__ == "__main__":
    test_patch_is_read_only_and_counts()
    test_dec_histogram_matches_dec_acc()
    test_norm_hook_matches_manual()
    test_save_load_roundtrip_and_tok_to_pixel()
    print("test_routing_dump: 全部通过")
