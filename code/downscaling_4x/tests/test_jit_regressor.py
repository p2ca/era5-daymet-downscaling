#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定确定性 JiT(阶段 A, 即 JDA/JMA)的结构契约。

阶段 A 的全部意义是"同量级下 transformer 对比 UNet", 所以**参数量口径不能被死参数污染**:
这里断言它既没有 TimestepEmbedder, patch 嵌入也不含加噪目标那一路输入。这两样在扩散版里
是必需的, 在回归里毫无贡献却会被计进参数量。

切块起点必须真的改变输出 —— 那是消接缝的唯一机制; 起点若不起作用, 网格伪影会在固定位置
累积, 而 loss 曲线一切正常。

纯 CPU, 不需要数据。

运行: python -m downscaling_4x.tests.test_jit_regressor
"""
import torch

from downscaling_4x import contract as C
from downscaling_4x.models.jit_backbone import JiT, TimestepEmbedder
from downscaling_4x.models.jit_regressor import JiTRegressor
from downscaling_4x.models.moe_ffn import DSMoE
from downscaling_4x.training.train_downscale import build_regressor, jit_moe_config

HW = (64, 128)
CFG = dict(patch=16, hidden=64, depth=4, heads=4, mlp_ratio=4.0, bottleneck=16,
           attn_dropout=0.0, proj_dropout=0.0, patch_margin=0, moe_intermediate=0,
           routed_scaling=1.0, moe_all_layers=False, moe_no_shared=False,
           experts=8, experts_per_tok=2)


def _build(moe, **over):
    cfg = dict(CFG, arch="jit", moe=moe, pos_grid=0, **over)
    return build_regressor(C.cond_channels("history_51"), 1, cfg, hw=HW)


def test_no_diffusion_only_parts():
    """回归版不得携带时间步嵌入, patch 嵌入也不得为加噪目标留输入通道。"""
    net = _build(False)
    assert not any(isinstance(m, TimestepEmbedder) for m in net.modules()), \
        "回归版不应有 TimestepEmbedder —— 它对回归无贡献却会被计进参数量"
    cin = C.cond_channels("history_51")
    assert net.x_embedder.proj1.in_channels == cin, \
        f"patch 嵌入输入应为 {cin}(纯条件), 实得 {net.x_embedder.proj1.in_channels}"
    # 对照: 扩散版必须是 cond_ch + out_ch, 且必须有 TimestepEmbedder
    dif = JiT(hw=HW, patch=16, cond_ch=cin, out_ch=1, hidden=64, depth=4, num_heads=4,
              bottleneck=16)
    assert dif.x_embedder.proj1.in_channels == cin + 1
    assert any(isinstance(m, TimestepEmbedder) for m in dif.modules())
    assert sum(p.numel() for p in net.parameters()) < sum(p.numel() for p in dif.parameters()), \
        "回归版参数应少于扩散版(少一个时间步嵌入与一路输入)"


def test_output_geometry_is_input_geometry():
    net = _build(False).eval()
    x = torch.randn(2, C.cond_channels("history_51"), *HW)
    for off in ((0, 0), (3, 7), (15, 15)):
        with torch.no_grad():
            y = net(x, offset=off)
        assert y.shape == (2, 1, *HW), (off, y.shape)


def test_offset_changes_output():
    """负对照: 起点若不改变输出, 消接缝机制等于没有, 而训练一切正常。"""
    net = _build(False).eval()
    torch.nn.init.normal_(net.final_layer.linear.weight, std=0.02)   # 打破零初始化
    x = torch.randn(1, C.cond_channels("history_51"), *HW)
    with torch.no_grad():
        a, b = net(x, offset=(0, 0)), net(x, offset=(5, 9))
    assert float((a - b).abs().max()) > 1e-6, "切块起点没有改变输出"


def test_zero_init_output():
    """输出层零初始化: 初始预测恒为 0。目标已 z-score, 这对回归是合理起点。"""
    net = _build(False).eval()
    x = torch.randn(1, C.cond_channels("history_51"), *HW)
    with torch.no_grad():
        assert float(net(x).abs().max()) == 0.0


def test_moe_placement_and_counts():
    d, m = _build(False), _build(True)
    assert len(d.moe_layers()) == 0
    kinds = [type(b.mlp).__name__ for b in m.blocks]
    assert kinds[0] != "DSMoE", "第 0 块必须恒稠密"
    assert kinds == ["SwiGLUFFN", "DSMoE", "SwiGLUFFN", "DSMoE"], kinds
    assert len(m.moe_layers()) == 2
    pd, pm = d.param_counts(), m.param_counts()
    assert pm["total"] > pd["total"], "MoE 总参数应更大"
    assert pm["activated"] < pm["total"], "激活参数应小于总参数"
    assert pm["routed_experts"] > 0


def test_moe_changes_output():
    """dense 与 MoE 必须给出不同结果, 否则 MoE 没有真正接进前向。"""
    d, m = _build(False).eval(), _build(True).eval()
    for net in (d, m):
        torch.nn.init.normal_(net.final_layer.linear.weight, std=0.02)
    x = torch.randn(1, C.cond_channels("history_51"), *HW)
    with torch.no_grad():
        assert float((d(x) - m(x)).abs().max()) > 1e-6, "dense 与 MoE 输出相同"


def test_moe_config_shared_with_stage_b():
    """阶段 A 与阶段 B 必须共用同一份 MoE 配置构造, 两处各写一份迟早分叉。"""
    cfg = dict(CFG, moe=True)
    c = jit_moe_config(cfg)
    assert c["num_experts"] == cfg["experts"]
    assert c["num_experts_per_tok"] == cfg["experts_per_tok"]
    assert c["interleave"] is True and c["use_shared_expert"] is True
    assert jit_moe_config(dict(cfg, moe=False)) is None


def test_forward_is_deterministic():
    net = _build(True).eval()
    x = torch.randn(1, C.cond_channels("history_51"), *HW)
    with torch.no_grad():
        assert torch.equal(net(x, offset=(2, 2)), net(x, offset=(2, 2)))


def main():
    tests = [
        test_no_diffusion_only_parts,
        test_output_geometry_is_input_geometry,
        test_offset_changes_output,
        test_zero_init_output,
        test_moe_placement_and_counts,
        test_moe_changes_output,
        test_moe_config_shared_with_stage_b,
        test_forward_is_deterministic,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
