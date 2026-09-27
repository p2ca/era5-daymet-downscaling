#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 D-EC 路由(router=dec)的口径: 成分全关等价于 ec; 域外 token 永不入选且容量按域内计;
批级池把名额跨帧分配; 先验只进选择不进门控; 难度头与主干互不传梯度; 掩膜、误差与路由三处
共用同一个 token 网格; 配置守卫与续训钉住; 推理期帧内规则与 batch 组成无关。

这些性质写错都不会报错: 域外泄漏只是多算一片海洋; 网格错位只是把地形喂成别的 token 的难度;
先验漏进门控只是让主损失偷偷训练难度头; 池没打通只是退回 ec 并顶着 D-EC 的名字跑完。

纯 CPU, 不需要数据。运行: python -m downscaling_4x.tests.test_dec_router
"""
import argparse
import math

import torch

from downscaling_4x.models.jit_backbone import JiT, token_domain_mask, token_pool
from downscaling_4x.models.moe_ffn import DSMoE, dec_pools, dec_standardize
from downscaling_4x.training import train_downscale as TD
from downscaling_4x.training.train_jit import RESUME_PINNED_ARGS, dec_aux_loss, jit_vloss

E, D, I, K = 8, 16, 32, 2
B, N = 2, 24
BASE = {"moe": True, "experts": E, "experts_per_tok": K, "moe_intermediate": I,
        "routed_scaling": 2.5, "moe_all_layers": False, "moe_no_shared": False,
        "hidden": D, "proj_dropout": 0.0}
MC = {"num_experts": E, "moe_intermediate_size": I, "num_experts_per_tok": K,
      "n_group": 2, "topk_group": 2, "routed_scaling_factor": 2.5,
      "interleave": True, "use_shared_expert": True, "proj_drop": 0.0}


def _moe(**kw):
    torch.manual_seed(0)
    return DSMoE(num_experts=E, dim=D, moe_inter=I, num_experts_per_tok=K,
                 routed_scaling_factor=2.5, **kw)


def _dec(pool=True, drop=True, prior=True, **kw):
    return _moe(router_mode="dec", dec_config={"pool": pool, "drop": drop, "prior": prior}, **kw)


def _jit(router, **dec):
    torch.manual_seed(0)
    mc = {**MC, "router_mode": router}
    if router == "dec":
        mc["dec"] = {"pool": False, "drop": False, "prior": True, "prior_max": 1.0,
                     "prior_ramp": 0.1, "head_layer": 1, **dec}
    return JiT(hw=(16, 24), patch=4, cond_ch=5, out_ch=1, hidden=32, depth=4,
               num_heads=2, bottleneck=8, moe_config=mc)


def _expect(fn, exc, what):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"未被拒绝: {what}")


def test_guards():
    """配置守卫: 全关、越权子开关、dec+bias、稠密+子开关、阶段 A 用 dec 一律拒绝; 缺掩膜/先验当场抛错。"""
    cfg = TD.jit_moe_config({**BASE, "router": "dec"})
    assert cfg["router_mode"] == "dec" and cfg["dec"] == {
        "pool": True, "drop": True, "prior": True, "prior_max": 1.0, "prior_ramp": 0.1, "head_layer": 1}
    assert TD.jit_moe_config({**BASE, "router": "ec"})["dec"] is None
    assert TD.jit_moe_config({**BASE, "router": "dec", "gating": "raw"})["gating_mode"] == "raw"
    _expect(lambda: TD.jit_moe_config({**BASE, "router": "dec", "dec_pool": 0, "dec_drop": 0, "dec_prior": 0}),
            SystemExit, "dec 三成分全关")
    _expect(lambda: TD.jit_moe_config({**BASE, "router": "ec", "dec_drop": 0}), SystemExit, "ec 带 dec 子开关")
    _expect(lambda: TD.jit_moe_config({**BASE, "router": "dec", "bias_gamma": 0.1}), SystemExit, "dec+bias_gamma")
    _expect(lambda: TD.jit_moe_config({**BASE, "moe": False, "dec_prior": 0}), SystemExit, "稠密+dec 子开关")
    _expect(lambda: TD.jit_moe_config({**BASE, "router": "dec", "dec_prior_ramp": 2.0}), SystemExit, "ramp 越界")
    _expect(lambda: TD.build_regressor(5, 1, {"arch": "jit", "patch": 4, "hidden": 32, "depth": 2, "heads": 2,
                                              "mlp_ratio": 4.0, "bottleneck": 8, "patch_margin": 0,
                                              "mode": "history_51", **BASE, "router": "dec"}, hw=(16, 24)),
            SystemExit, "阶段 A 用 dec")
    _expect(lambda: DSMoE(E, D, I, router_mode="dec", dec_config={"pool": 0, "drop": 0, "prior": 0}),
            ValueError, "DSMoE 全关")
    x = torch.randn(B, N, D)
    _expect(lambda: _moe(router_mode="tc")(x, tok_mask=torch.ones(B * N, dtype=torch.bool)),
            RuntimeError, "tc 收 tok_mask")
    _expect(lambda: _moe(router_mode="ec")(x, prior=torch.zeros(B * N)), RuntimeError, "ec 收 prior")
    _expect(lambda: _dec(drop=True)(x, prior=torch.zeros(B * N)), RuntimeError, "dec drop 缺 tok_mask")
    _expect(lambda: _dec(drop=False, prior=True)(x), RuntimeError, "dec prior 缺 prior")
    _expect(lambda: _jit("dec", drop=True)(torch.randn(1, 1, 16, 24), torch.rand(1), torch.randn(1, 5, 16, 24)),
            RuntimeError, "JiT dec drop 缺 domain_mask")
    _expect(lambda: _jit("dec", head_layer=3), ValueError, "难度头晚于首个 MoE 块")


def test_all_off_equals_ec():
    """drop 开但掩膜全 True、pool 关、λ=0: 选择、门控与输出都与 ec 逐位相同。"""
    ec, dec = _moe(router_mode="ec"), _dec(pool=False, drop=True, prior=True)
    assert all(torch.equal(a, b) for a, b in zip(ec.state_dict().values(), dec.state_dict().values()))
    torch.manual_seed(1)
    x = torch.randn(B, N, D)
    scores = torch.rand(B * N, E)
    all_on = torch.ones(B * N, dtype=torch.bool)
    sel_e, w_e = ec.route_ec(scores, B, N)
    for train in (True, False):
        dec.train(train)
        sel_d, w_d = dec.route_dec(scores, B, N, all_on, torch.randn(B * N))   # λ=0: 先验被忽略
        assert torch.equal(sel_e, sel_d) and torch.allclose(w_e, w_d)
    with torch.no_grad():
        assert torch.allclose(ec(x), dec(x, tok_mask=all_on, prior=torch.randn(B * N)), atol=1e-6)


def test_drop_excludes_outside_domain():
    """域外 token 选择恒为 0, 门控恒为 0, 输出只剩共享专家; 每专家每帧恰取 ⌈n_in·K/E⌉。"""
    m = _dec(pool=False, drop=True, prior=False)
    torch.manual_seed(2)
    x = torch.randn(B, N, D)
    scores = torch.rand(B * N, E)
    tok = torch.rand(B * N) > 0.4
    assert (~tok).any()
    sel, w = m.route_dec(scores, B, N, tok, None)
    assert int(sel[~tok].sum()) == 0 and float(w[~tok].abs().sum()) == 0.0
    for b in range(B):
        sl = slice(b * N, (b + 1) * N)
        n_in = int(tok[sl].sum())
        per = sel[sl].sum(0)
        assert torch.all(per == math.ceil(n_in * K / E)), per
    with torch.no_grad():
        out = m(x, tok_mask=tok, prior=None).reshape(-1, D)
        shared = m.shared_experts(x).reshape(-1, D)
    assert torch.allclose(out[~tok], shared[~tok], atol=1e-6), "域外 token 不该有路由输出"
    assert not torch.allclose(out[tok], shared[tok]), "域内 token 应有路由输出"


def test_pool_moves_capacity_across_frames():
    """批级池: 名额跨帧流动, 池内每专家合计恰为 C_pool, 分数高的帧拿得多; 帧内池下每帧恰为 C。"""
    torch.manual_seed(3)
    scores = torch.cat([torch.rand(N, E) * 0.5, 0.5 + torch.rand(N, E) * 0.5], 0)   # 帧 1 整体更高
    pooled, framed = _dec(pool=True, drop=False, prior=False), _moe(router_mode="ec")
    pooled.train()
    sel_p, _ = pooled.route_dec(scores, B, N, None, None)
    sel_f, _ = framed.route_ec(scores, B, N)                      # 帧内池 = ec
    c_pool, c_frame = math.ceil(B * N * K / E), math.ceil(N * K / E)
    assert torch.all(sel_p.sum(0) == c_pool) and torch.all(sel_f.view(B, N, E).sum(1) == c_frame)
    f0, f1 = sel_p[:N].sum(0), sel_p[N:].sum(0)
    assert torch.all(f1 > f0), (f0, f1)
    # 帧序无关: 交换两帧, 选择随之交换
    sel_s, _ = pooled.route_dec(torch.cat([scores[N:], scores[:N]], 0), B, N, None, None)
    assert torch.equal(sel_s, torch.cat([sel_p[N:], sel_p[:N]], 0))
    # eval 缺省退回帧内规则: 与 batch 组成无关
    pooled.eval()
    sel_e, _ = pooled.route_dec(scores, B, N, None, None)
    assert torch.equal(sel_e, sel_f)


def test_prior_selection_only():
    """先验: λ=0 不改变选择; 大先验使 token 被全部专家选中、负先验使其落选; 门控只由原始亲和分决定。"""
    m = _dec(pool=False, drop=False, prior=True)
    m.train()
    torch.manual_seed(4)
    scores = torch.rand(B * N, E)
    prior = torch.randn(B * N)
    m.dec_lambda = 0.0
    sel0, w0 = m.route_dec(scores, B, N, None, prior)
    assert torch.equal(sel0, m.route_ec(scores, B, N)[0])
    m.dec_lambda = 100.0
    hard, easy = 3, 5
    prior2 = prior.clone(); prior2[hard] = 1e3; prior2[easy] = -1e3
    sel1, w1 = m.route_dec(scores, B, N, None, prior2)
    assert int(sel1[hard].sum()) == E and int(sel1[easy].sum()) == 0
    assert not torch.equal(sel0, sel1)
    ref = scores.masked_fill(~sel1, 0.0)
    ref = ref / ref.sum(-1, keepdim=True).clamp_min(1e-20) * 2.5
    assert torch.allclose(w1, ref), "门控权重必须只由原始亲和分与选择决定"
    # 标准化: 逐池零均值单位方差、截断 ±3、无效处为 0
    valid = torch.ones(B * N, dtype=torch.bool); valid[0] = False
    z = dec_standardize(prior2, valid, dec_pools(B, N, False))
    assert float(z[0]) == 0.0 and float(z.abs().max()) <= 3.0 + 1e-6
    for sl in dec_pools(B, N, False):
        zz = z[sl][valid[sl]]
        assert abs(float(zz.mean())) < 0.5      # 截断后均值略偏, 但量级为 1


def test_token_grid_alignment():
    """掩膜与误差图经同一 token_pool: 单块掩膜恰落到一个 token, 单点误差落在同一个 token(随机起点)。"""
    H, W, patch = 32, 48, 8
    grid = (-(-(H + patch - 1) // patch) * patch, -(-(W + patch - 1) // patch) * patch)
    gw = grid[1] // patch
    g = torch.Generator().manual_seed(5)
    for _ in range(6):
        dy, dx = [int(v) for v in torch.randint(0, patch, (2,), generator=g)]
        gy, gx = int(torch.randint(1, grid[0] // patch - 1, (1,), generator=g)), int(torch.randint(1, gw - 1, (1,), generator=g))
        y0, x0 = gy * patch - dy, gx * patch - dx                # 该 token 在未 pad 图上的左上角
        mask = torch.zeros(1, 1, H, W); mask[0, 0, y0:y0 + patch, x0:x0 + patch] = 1
        tm = token_domain_mask(mask, patch, (dy, dx), grid)
        assert int(tm.sum()) == 1 and bool(tm[0, gy * gw + gx]), (dy, dx, gy, gx)
        err = torch.zeros(1, 1, H, W); err[0, 0, y0 + patch // 2, x0 + patch // 2] = 7.0
        te = token_pool(err, patch, (dy, dx), grid)
        assert int((te > 0).sum()) == 1 and float(te[0, gy * gw + gx]) > 0


def test_head_stopgrad_and_lambda_schedule():
    """λ=0 时 D-EC 主体与 ec 主体输出逐位相同; 辅助损失只到难度头, 主损失不到难度头; λ 按进度线性升满。"""
    net_ec, net = _jit("ec"), _jit("dec", drop=False, pool=False, prior=True)
    for k, v in net_ec.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), f"主干参数被扰动: {k}"
    torch.manual_seed(6)
    z, t, cond = torch.randn(2, 1, 16, 24), torch.rand(2), torch.randn(2, 5, 16, 24)
    land = (torch.rand(2, 1, 16, 24) > 0.3).float()
    assert net.set_dec_progress(0.0) == 0.0 and abs(net.set_dec_progress(0.05) - 0.5) < 1e-9
    assert net.set_dec_progress(0.2) == 1.0 and net.moe_layers()[0].dec_lambda == 1.0
    net.set_dec_progress(0.0)
    with torch.no_grad():
        assert torch.equal(net_ec(z, t, cond), net(z, t, cond, domain_mask=land))
    net.set_dec_progress(1.0)
    net.train()
    loss, aux, extra = jit_vloss(net, land * torch.randn(2, 1, 16, 24), cond, land, 1.0, -0.8, 0.8, 0.05,
                                 generator=torch.Generator().manual_seed(1), patch=4,
                                 domain_mask=land, dec_aux=True)
    assert torch.isfinite(loss) and torch.isfinite(aux) and -1.0 <= extra["corr"] <= 1.0
    assert abs(float(aux) - extra["aux"]) < 1e-6, "辅助项须与扩散损失分开返回"
    # 只反传辅助项: 主干参数梯度全零/缺席, 难度头有梯度
    out, info = net(z, t, cond, domain_mask=land, return_dec=True)
    aux, _ = dec_aux_loss(net, info, torch.randn(2, 1, 16, 24), land, 4, info["offset"])
    net.zero_grad(); aux.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.dec_head.parameters())
    for n, p in net.named_parameters():
        if not n.startswith("dec_head."):
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, f"辅助损失泄漏到主干: {n}"
    # 只反传主损失: 难度头无梯度
    net.zero_grad()
    out = net(z, t, cond, domain_mask=land)
    (out.square() * land).mean().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in net.dec_head.parameters()), \
        "主损失泄漏进难度头"
    # 非 D-EC 主体不接受 domain_mask 之外的任何 dec 接口影响: 旧调用逐字可用
    assert jit_vloss(net_ec, land * torch.randn(2, 1, 16, 24), cond, land, 1.0, -0.8, 0.8, 0.05,
                     generator=torch.Generator().manual_seed(1), patch=4).ndim == 0


def test_eval_rule_capacity_and_batch_independence():
    """推理期帧内规则: 同一帧单独或批内决策逐位相同; 容量系数 f 使每专家取 ⌈n_in·K·f/E⌉。"""
    m = _dec(pool=True, drop=True, prior=False)
    m.eval()
    torch.manual_seed(7)
    scores = torch.rand(B * N, E)
    tok = torch.rand(B * N) > 0.4
    sel_b, _ = m.route_dec(scores, B, N, tok, None)
    sel_0, _ = m.route_dec(scores[:N], 1, N, tok[:N], None)
    assert torch.equal(sel_b[:N], sel_0), "推理期选择泄漏到帧外"
    m.set_dec_eval("frame", 0.5)
    sel_h, _ = m.route_dec(scores, B, N, tok, None)
    for b in range(B):
        sl = slice(b * N, (b + 1) * N)
        n_in = int(tok[sl].sum())
        assert torch.all(sel_h[sl].sum(0) == math.ceil(n_in * K * 0.5 / E))
    _expect(lambda: m.set_dec_eval("threshold", 1.0), ValueError, "未知推理规则")


def test_resume_pins_and_legacy():
    """dec 键全部钉进阶段 B 续训契约; 旧断点按缺省档回填; 续训段切换子开关被拒。"""
    keys = ("dec_pool", "dec_drop", "dec_prior", "dec_prior_max", "dec_prior_ramp", "dec_head_layer")
    assert all(k in RESUME_PINNED_ARGS for k in keys)
    d = {"router": "ec"}
    filled = TD.apply_legacy_defaults(d, TD.LEGACY_DEFAULTS_JIT)
    assert set(keys) <= set(filled) and d["dec_pool"] == 1 and d["dec_prior_ramp"] == 0.1
    assert "dec_pool" not in TD.LEGACY_DEFAULTS, "共用回填表不该带阶段 B 专属键"
    ck = {"args": {"router": "dec", "dec_pool": 1, "dec_prior": 1}}
    ok = argparse.Namespace(router="dec", dec_pool=1, dec_prior=1)
    TD.check_resume_args(ck, ok, ("router", "dec_pool", "dec_prior"))
    _expect(lambda: TD.check_resume_args(ck, argparse.Namespace(router="dec", dec_pool=0, dec_prior=1),
                                         ("router", "dec_pool", "dec_prior")), SystemExit, "续训切换 dec_pool")


def main():
    tests = [test_guards, test_all_off_equals_ec, test_drop_excludes_outside_domain,
             test_pool_moves_capacity_across_frames, test_prior_selection_only,
             test_token_grid_alignment, test_head_stopgrad_and_lambda_schedule,
             test_eval_rule_capacity_and_batch_independence, test_resume_pins_and_legacy]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
