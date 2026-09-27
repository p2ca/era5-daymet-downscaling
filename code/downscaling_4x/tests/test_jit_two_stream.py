#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_two_stream.py — 两条流 JiT 的开关契约(纯 CPU)
============================================================================
钉住的都是"写错不会报错"的地方:
  1. 全部开关处于缺省时 jit_stream_config 返回 None, 模型与单流版逐位相同(含 wards=1 的分组无操作);
  2. 开关矩阵: 合法组合都能建模、前向、反传; 非法组合被 jit_stream_config 当场拒绝;
  3. 通道拆分按名字: ERA5 当天与历史通道进粗流, 静态与年内相位进细流, 通道数逐项对上;
  4. 公里制坐标: 东西向按纬度余弦收缩; km 档 RoPE 的分数只依赖相对位移;
  5. 按风推位置章: 推移量 = 风速 x τ 换算成 token 间距, τ=0 的头用原位置;
  6. 粗 token 掩膜: 被掩掉的粗 token 无论内容如何都不影响交叉问询的输出;
  7. precompute 复用与逐步重算逐位相同; 起点不一致的缓存被拒绝;
  8. TC 域外舍弃: 域外 token 的路由专家输出恒为 0, 只剩共享专家;
  9. 两级分诊: 选中的专家同科室; 科室键足够大时全部落到该科室; wards=1 与旧路由逐位相同;
 10. 键的零初始化: 带键模型与不带键模型在同一套主干权重下输出逐位相同;
 11. 数据常量随 state_dict 保存与恢复; 全部开关都在续训契约里。
Run: python -m downscaling_4x.tests.test_jit_two_stream
============================================================================
"""
import itertools
import math

import numpy as np
import torch

from downscaling_4x import contract as C
from downscaling_4x.models.jit_backbone import JiT, CrossAttention, Rope2D, StreamContext, parse_tau_spec
from downscaling_4x.models.moe_ffn import DSMoE
from downscaling_4x.training import train_downscale as TD
from downscaling_4x.training.train_jit import build_model, RESUME_PINNED_ARGS

torch.manual_seed(0)
HW, PATCH = (64, 128), 16
MODE = C.DEFAULT_MODE
COND_CH = C.cond_channels(MODE)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def base_args(**over):
    a = {"patch": PATCH, "patch_margin": 2, "hidden": 32, "depth": 4, "heads": 4, "mlp_ratio": 4.0,
         "bottleneck": 16, "attn_dropout": 0.0, "proj_dropout": 0.0, "mode": MODE, "cond_ch": COND_CH,
         "moe": True, "experts": 4, "experts_per_tok": 2, "moe_intermediate": 32, "routed_scaling": 2.5,
         "moe_all_layers": False, "moe_no_shared": False, "bias_gamma": 0.0, "router": "tc", "gating": "norm",
         "refine_head": 0, **TD.DEC_DEFAULTS, **TD.ARCH_DEFAULTS}
    a.update(over)
    return a


def synth(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, 1, *HW, generator=g)
    cond = torch.randn(B, COND_CH, *HW, generator=g)
    land = (torch.rand(B, 1, *HW, generator=g) > 0.3).float()
    t = torch.rand(B, generator=g)
    return z, cond, land, t


def make(seed=0, **over):
    torch.manual_seed(seed)
    net = build_model(base_args(**over), HW)
    if net.two_stream:
        ev = np.ones(HW, dtype=np.uint8); ev[:, :24] = 0                 # 西侧一条无 ERA5 数据
        net.set_data_constants(np.zeros(4), np.ones(4), ev)
    return net.eval()


FULL = dict(rope_units="km", drop_outside=1, two_stream=1, lagrangian=1,
            tau_spec="0,850:6,500:12,500:24", expert_two_card=1, dense_two_card=1,
            wards=2, ward_key=1, terrain_key=1, terrain_key_window=20)


def main():
    z, cond, land, t = synth()

    # ---------- 1. 缺省 = 单流 ----------
    check("全缺省时 jit_stream_config 返回 None", TD.jit_stream_config(base_args()) is None)
    n0 = make(0)
    check("全缺省模型 arch 为 None, 无粗流/地形键/常量 buffer",
          n0.arch is None and n0.coarse is None and n0.terrain is None and not hasattr(n0, "wind_mean"))
    mc = TD.jit_moe_config(base_args())
    check("wards=1 保留 DeepSeek 缺省的 2 组选 2 组", (mc["n_group"], mc["topk_group"]) == (2, 2))

    # ---------- 2. 开关矩阵 ----------
    valid = [dict(two_stream=1), dict(two_stream=1, coarse_blocks=0, coarse_conv=0),
             dict(two_stream=1, rope_units="km", lagrangian=1, tau_spec="0,850:6,500:12,500:24"),
             dict(two_stream=1, cross_attn=0, expert_two_card=1),
             dict(two_stream=1, cross_attn=0, dense_two_card=1),
             dict(two_stream=1, expert_two_card=1, dense_two_card=1, two_card_dims="12,20"),
             dict(two_stream=1, wards=2, ward_key=1), dict(wards=2), dict(terrain_key=1),
             dict(rope_units="km"), dict(drop_outside=1), dict(two_stream=1, cross_modulate=0, cross_mask_outside=0),
             FULL, dict(moe=False, two_stream=1, dense_two_card=1)]
    for i, ov in enumerate(valid):
        net = make(i, **ov).train()
        out = net(z, t, cond, offset=(3, 5), domain_mask=land)
        out.float().square().mean().backward()
        assert out.shape == (2, 1, *HW) and torch.isfinite(out).all(), ov
    check(f"{len(valid)} 个合法组合都能建模、前向、反传", True)
    invalid = [dict(two_stream=0, cross_attn=0), dict(two_stream=0, coarse_blocks=2), dict(two_stream=0, lagrangian=1),
               dict(two_stream=1, cross_attn=0), dict(two_stream=1, lagrangian=1),
               dict(two_stream=1, rope_units="km", lagrangian=1, cross_attn=0, expert_two_card=1),
               dict(two_stream=1, cross_attn=0, expert_two_card=1, cross_mask_outside=0),
               dict(two_stream=1, ward_key=1), dict(two_stream=1, wards=2, ward_key=1, moe=False),
               dict(two_stream=1, expert_two_card=1, moe=False), dict(terrain_key=1, moe=False),
               dict(wards=3), dict(ward_topk=2), dict(terrain_key=1, terrain_key_window=15),
               dict(rope_units="miles"), dict(two_stream=1, two_card_dims="12", expert_two_card=1),
               dict(wards=2, router="ec"), dict(drop_outside=1, router="ec")]
    for ov in invalid:
        try:
            build_model(base_args(**ov), HW)
            raise AssertionError(f"非法组合未被拒绝: {ov}")
        except SystemExit:
            pass
    check(f"{len(invalid)} 个非法组合被当场拒绝", True)
    try:
        parse_tau_spec("0,850:6,500:12", 4)
        raise AssertionError("tau_spec 项数不等于头数未被拒绝")
    except ValueError:
        check("tau_spec 项数必须等于头数", True)

    # ---------- 3. 通道拆分 ----------
    nf = make(1, **FULL)
    layout = C.cond_layout(MODE)
    era5 = [i for i, n in enumerate(layout) if n in C.ERA5_IN or n.startswith(C.HISTORY_PREFIX)]
    fine = [i for i, n in enumerate(layout) if n in C.STATIC_ORDER or n in C.TIME_ORDER]
    check("ERA5 当天+历史通道进粗流, 静态+年内相位进细流", nf.idx_era5 == era5 and nf.idx_fine == fine)
    check("粗流 47 通道(含年内相位), 细流 7 通道(含噪声目标)",
          nf.coarse.embed.proj1.in_channels == len(era5) + 2 and nf.x_embedder.proj1.in_channels == 1 + len(fine))
    check("风通道索引按名字取到 850/500 的 u、v",
          [layout[i] for i in nf.idx_wind] == list(JiT.WIND_VARS))

    # ---------- 4. 公里制 ----------
    y, x = nf.coords((0, 0), "cpu")
    gh, gw = nf.x_embedder.gh, nf.x_embedder.gw
    lat = nf.lat0 + (torch.arange(gh) * PATCH + PATCH / 2.0) * nf.dlat
    xj = torch.arange(gw).float()[None, :] * torch.cos(torch.deg2rad(lat))[:, None]
    check("km 档: 南北为行序号, 东西 = 列序号 x cos(纬度)",
          torch.allclose(y, torch.arange(gh).float().repeat_interleave(gw)) and torch.allclose(x, xj.reshape(-1)))
    rope = Rope2D(8, gh, gw)
    q, k = torch.randn(8), torch.randn(8)
    def score(yq, xq, yk, xk):
        cq, sq = rope.tables(torch.tensor([yq]), torch.tensor([xq])); ck, sk = rope.tables(torch.tensor([yk]), torch.tensor([xk]))
        rq = q * cq[0] + torch.stack((-q[1::2], q[0::2]), -1).reshape(-1) * sq[0]
        rk = k * ck[0] + torch.stack((-k[1::2], k[0::2]), -1).reshape(-1) * sk[0]
        return float(rq @ rk)
    s1, s2, s3 = score(1.0, 0.7, 2.5, 3.2), score(4.0, 1.7, 5.5, 4.2), score(1.0, 0.7, 2.0, 3.2)
    check("km 档 RoPE: 同位移分数一致, 不同位移不同", abs(s1 - s2) < 1e-4 and abs(s1 - s3) > 1e-3)
    ctx = nf.precompute(cond, (0, 0))
    cg, sg = nf.rope.tables(y, x)
    check("km 档的 q 表 = 按 km 坐标现算的表", torch.equal(ctx.q_tabs[0], cg) and torch.equal(ctx.q_tabs[1], sg))

    # ---------- 5. 按风推位置章 ----------
    cw = cond.clone()
    iu, iv, iu5, iv5 = nf.idx_wind
    cw[:, iu] = 10.0; cw[:, iv] = 0.0; cw[:, iu5] = 0.0; cw[:, iv5] = -5.0       # 统计量 mean 0 / std 1
    ctx = nf.precompute(cw, (0, 0))
    B = cw.shape[0]
    tabs_base = nf.rope.tables(y[None].expand(B, -1), x[None].expand(B, -1))
    unit = 3.6 / nf.token_km
    spec = nf.tau_spec
    check("tau_spec 解析: 4 头 = 不推 / 850:6 / 500:12 / 500:24",
          spec == [("850", 0.0), ("850", 6.0), ("500", 12.0), ("500", 24.0)])
    from downscaling_4x.models.jit_backbone import token_pool
    frac = token_pool(torch.ones(B, 1, *HW), PATCH, (0, 0), nf.grid_hw)         # 块内图像像素占比
    inside = frac[0] == 1.0                                                    # 完整落在图像内的 token
    expect = []
    for level, hours in spec:
        u, v = (10.0, 0.0) if level == "850" else (0.0, -5.0)
        # 边缘 token 的风按块内图像像素取均值, 常数风场下仍等于常数; 纯补边 token 为 0
        uu, vv = u * (frac > 0).float(), v * (frac > 0).float()
        expect.append(nf.rope.tables(y[None] + vv * hours * unit, x[None] + uu * hours * unit))
    ok = all(torch.allclose(ctx.k_tabs[0][:, h], expect[h][0], atol=1e-6) and
             torch.allclose(ctx.k_tabs[1][:, h], expect[h][1], atol=1e-6) for h in range(4))
    check("K 表: 各头按 (风层, τ) 推移 = 风速 x τ x 3.6 / 一个 token 的公里数", ok)
    shift = 10.0 * 6.0 * 3.6 / nf.token_km
    c_in, s_in = nf.rope.tables(y[inside], x[inside] + shift)
    check("图像内 token 的 850:6h 推移量恰为 10 m/s x 6 h", torch.allclose(ctx.k_tabs[0][0, 1][inside], c_in, atol=1e-6))
    check("τ=0 的头用原位置", torch.allclose(ctx.k_tabs[0][:, 0], tabs_base[0]) and torch.allclose(ctx.k_tabs[1][:, 0], tabs_base[1]))
    cz = cond.clone(); cz[:, [iu, iv, iu5, iv5]] = 0.0
    ctx0 = nf.precompute(cz, (0, 0))
    check("风为零时全部头退回原位置", all(torch.allclose(ctx0.k_tabs[0][:, h], tabs_base[0]) for h in range(4)))
    n_nolag = make(1, **{**FULL, "lagrangian": 0})
    n_nolag.set_data_constants(np.zeros(4), np.ones(4), nf.era5_valid.numpy())
    ctxn = n_nolag.precompute(cw, (0, 0))
    check("lagrangian=0 时 K 表就是 q 表", ctxn.k_tabs is ctxn.q_tabs)

    # ---------- 6. 粗 token 掩膜 ----------
    ca = CrossAttention(32, 4)
    xq, xkv = torch.randn(2, 10, 32), torch.randn(2, 12, 32)
    qt = rope.tables(torch.arange(10).float(), torch.zeros(10))            # 细 token 的位置表
    kt = rope.tables(torch.arange(12).float() * 0.8, torch.ones(12))       # 粗 token 的位置表
    mask = torch.ones(2, 12, dtype=torch.bool); mask[:, 3] = False; mask[1, 7] = False
    o1 = ca(xq, xkv, rope, qt, kt, mask)
    xkv2 = xkv.clone(); xkv2[:, 3] += 100.0; xkv2[1, 7] -= 50.0
    o2 = ca(xq, xkv2, rope, qt, kt, mask)
    o3 = ca(xq, xkv2, rope, qt, kt, None)
    check("被掩掉的粗 token 内容任意变化不影响输出; 不掩则影响", torch.allclose(o1, o2) and not torch.allclose(o1, o3))
    valid_tok = ctx.coarse_valid[0].view(gh, gw)
    gi, gj = HW[0] // PATCH, HW[1] // PATCH                                # 图像内完整的 token 行列数
    check("ERA5 无数据的列与纯补边的行列都被掩掉, 其余保留",
          bool((~valid_tok[:, 0]).all()) and bool(valid_tok[:gi, 2:gj].all())
          and bool((~valid_tok[gi:, :]).all()) and bool((~valid_tok[:, gj:]).all()))

    # ---------- 7. 缓存等价 ----------
    with torch.no_grad():
        o_direct = nf(z, t, cw, offset=(3, 5), domain_mask=land)
        cache = nf.precompute(cw, (3, 5))
        o_cached = nf(z, t, cw, offset=(3, 5), domain_mask=land, coarse_cache=cache)
    check("precompute 复用与逐步重算逐位相同", torch.equal(o_direct, o_cached))
    try:
        nf(z, t, cw, offset=(4, 5), domain_mask=land, coarse_cache=cache)
        raise AssertionError("起点不一致的缓存未被拒绝")
    except RuntimeError:
        check("起点不一致的缓存被拒绝", True)

    # ---------- 8. TC 域外舍弃 ----------
    layer = DSMoE(4, 16, 24, num_experts_per_tok=2, drop_outside=True)
    xx = torch.randn(2, 6, 16)
    tm = torch.ones(12, dtype=torch.bool); tm[[1, 4, 9]] = False
    with torch.no_grad():
        yy = layer(xx, tok_mask=tm)
        shared = layer.shared_experts(xx)
    check("域外 token 只剩共享专家, 域内 token 有路由输出",
          torch.equal(yy.reshape(12, 16)[~tm], shared.reshape(12, 16)[~tm]) and not torch.allclose(yy.reshape(12, 16)[tm], shared.reshape(12, 16)[tm]))
    check("负载计数只数域内 token", int(layer.load_acc.sum()) == 9 * 2)

    # ---------- 9. 两级分诊 ----------
    lw = DSMoE(4, 16, 24, num_experts_per_tok=2, n_group=2, topk_group=1, ward_key_dim=3)
    scores = torch.rand(12, 4)
    idx, w = lw.route(scores, None)
    check("两位专家同科室(组 = 专家序号 // 2)", bool(((idx // 2)[:, 0] == (idx // 2)[:, 1]).all()))
    bias = torch.tensor([[-100.0, 100.0]]).expand(12, -1)
    idx2, _ = lw.route(scores, bias)
    check("科室键足够大时全部落到该科室", bool((idx2 // 2 == 1).all()))
    old = DSMoE(4, 16, 24, num_experts_per_tok=2)                     # 缺省 2 组选 2 组
    idx3, w3 = old.route(scores)
    top2 = torch.topk(scores, 2, dim=-1, sorted=False)[1]
    check("wards=1 的分组无操作 = 全局 top-2", torch.equal(torch.sort(idx3, -1)[0], torch.sort(top2, -1)[0]))

    # ---------- 10. 键的零初始化 ----------
    torch.manual_seed(3)
    na = build_model(base_args(**{**FULL, "ward_key": 0, "terrain_key": 0}), HW)
    torch.manual_seed(3)
    nb = build_model(base_args(**FULL), HW)
    sd = na.state_dict()
    missing, unexpected = nb.load_state_dict(sd, strict=False)
    check("带键模型比不带键模型只多出零初始化的键投影与地形键模块",
          not unexpected and all(("ward_proj" in k or "terrain_proj" in k or k.startswith("terrain.")) for k in missing))
    for k in missing:
        if "proj" in k and "terrain." not in k:
            assert float(nb.state_dict()[k].abs().sum()) == 0.0, k
    ev = np.ones(HW, dtype=np.uint8)
    na.set_data_constants(np.zeros(4), np.ones(4), ev); nb.set_data_constants(np.zeros(4), np.ones(4), ev)
    na.eval(); nb.eval()
    with torch.no_grad():
        oa = na(z, t, cw, offset=(1, 2), domain_mask=land); ob = nb(z, t, cw, offset=(1, 2), domain_mask=land)
    check("键投影为零时带键模型与不带键模型输出逐位相同", torch.equal(oa, ob))

    # ---------- 11. 常量与契约 ----------
    sd = nf.state_dict()
    check("风统计量与 ERA5 掩膜进 state_dict", "wind_mean" in sd and "era5_valid" in sd and "consts_ready" in sd)
    n_fresh = build_model(base_args(**FULL), HW)
    n_fresh.load_state_dict(sd)
    check("常量随 checkpoint 恢复", int(n_fresh.consts_ready) == 1 and torch.equal(n_fresh.era5_valid, nf.era5_valid))
    n_empty = build_model(base_args(**FULL), HW)
    try:
        n_empty.precompute(cond, (0, 0))
        raise AssertionError("缺常量的两条流模型未被拒绝")
    except RuntimeError:
        check("缺数据常量时 precompute 拒绝运行", True)
    check("全部结构开关都在续训契约里", all(k in RESUME_PINNED_ARGS for k in TD.ARCH_DEFAULTS))
    print("test_jit_two_stream: 全部通过")


if __name__ == "__main__":
    main()
