#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定输出端精修头(RefineHead)的口径: 关=逐字节现状、开=初始恒等、梯度可达、
静态切片名字对拍、续训拒绝切换、双开关独立。

这些性质写错都不会报错: 关时多消耗一次随机流会让"关"与历史 run 不再逐位同构;
零初始化缺失会让开关打开的瞬间输出突变; 切片索引错位只是把地形喂成别的通道。

纯 CPU, 不需要数据。运行: python -m downscaling_4x.tests.test_refine_head
"""
import torch

from downscaling_4x import contract as C
from downscaling_4x.models.jit_backbone import JiT
from downscaling_4x.models.jit_regressor import JiTRegressor
from downscaling_4x.training import train_downscale as TD

HW, P, CH = (64, 128), 16, 8
IDX = [3, 4, 5, 6]                       # 合成用: 假装这 4 个通道是静态


def _mk(cls, refine, **kw):
    torch.manual_seed(0)
    if cls is JiT:
        return JiT(hw=HW, patch=P, cond_ch=CH, out_ch=1, hidden=64, depth=2,
                   num_heads=4, bottleneck=16, refine_head=refine,
                   refine_static_idx=IDX if refine else None, **kw)
    return JiTRegressor(hw=HW, patch=P, cond_ch=CH, out_ch=1, hidden=64, depth=2,
                        num_heads=4, bottleneck=16, refine_head=refine,
                        refine_static_idx=IDX if refine else None, **kw)


def test_off_is_bytewise_status_quo():
    """关 = 不构造子模块: 参数集合与随机流消耗同现状, 前向逐位相同。"""
    for cls in (JiTRegressor, JiT):
        a, b = _mk(cls, 0), _mk(cls, 0)
        assert all(k1 == k2 and torch.equal(v1, v2) for (k1, v1), (k2, v2)
                   in zip(a.state_dict().items(), b.state_dict().items()))
        assert not any("refine" in k for k in a.state_dict())
        on = _mk(cls, 1)
        extra = {k for k in on.state_dict()} - {k for k in a.state_dict()}
        assert extra and all(k.startswith("refine.") for k in extra), extra
        # 开侧的主干参数与关侧逐位相同(精修头在 final_layer 之后构造, 不动此前的随机流)
        for k, v in a.state_dict().items():
            assert torch.equal(v, on.state_dict()[k]), f"主干参数被扰动: {k}"


def test_on_starts_as_identity():
    """末层零初始化 ⇒ 打开开关的初始输出与关侧逐位相同。"""
    torch.manual_seed(7)
    cond = torch.randn(2, CH, *HW)
    for cls, args in ((JiTRegressor, (cond,)),
                      (JiT, (torch.randn(2, 1, *HW), torch.rand(2), cond))):
        off, on = _mk(cls, 0), _mk(cls, 1)
        with torch.no_grad():
            o1 = off(*args) if cls is JiTRegressor else off(args[0], args[1], args[2])
            o2 = on(*args) if cls is JiTRegressor else on(args[0], args[1], args[2])
        assert torch.equal(o1, o2), f"{cls.__name__}: 零初始化恒等被破坏"


def test_grad_reaches_head():
    """零初始化签名: 初始时只有 conv3 有梯度(前层被零权重截断——与 adaLN-zero 同理);
    conv3 离开零点后梯度必须贯通到 conv1/2。两相都验才有分辨力。"""
    m = _mk(JiTRegressor, 1)
    cond = torch.randn(2, CH, *HW)
    tgt = torch.randn(2, 1, *HW)          # 主干输出层也是零初始化, 须用非零目标造出上游梯度
    (m(cond) - tgt).square().mean().backward()
    assert m.refine.conv3.weight.grad.abs().sum() > 0
    assert m.refine.conv1.weight.grad.abs().sum() == 0    # 零初始化的正确签名
    m.zero_grad()
    with torch.no_grad():
        m.refine.conv3.weight.normal_(std=0.02)           # 离开零点
    (m(cond) - tgt).square().mean().backward()
    for n in ("conv1", "conv2", "conv3"):
        g = getattr(m.refine, n).weight.grad
        assert g is not None and g.abs().sum() > 0, f"梯度未达 {n}"


def test_static_slice_matters():
    """切片索引错位必须改变输出(把头参数随机化后对拍), 否则名字对拍无分辨力。"""
    m = _mk(JiTRegressor, 1)
    torch.manual_seed(3)
    for p in m.refine.parameters():
        p.data.normal_()
    cond = torch.randn(1, CH, *HW)
    with torch.no_grad():
        o1 = m(cond)
        m.refine_static_idx = [6, 5, 4, 3]           # 错位
        o2 = m(cond)
    assert not torch.equal(o1, o2), "静态切片错位未反映到输出"


def test_resume_rejects_toggle():
    """续训契约必须拒绝 refine_head 切换; 旧断点缺键按默认档回填后放行。"""
    import argparse
    mk = lambda v: argparse.Namespace(refine_head=v, router="tc", gating="norm",
                                      mode="history_51", improve_criterion="abs",
                                      improve_tol=1e-4, rand_offset=0)
    c0 = TD.resume_contract(mk(0), 1, 1, 1, {}, {})
    c1 = TD.resume_contract(mk(1), 1, 1, 1, {}, {})
    assert any("refine_head" in d for d in TD.contract_mismatches(c0, c1))
    assert TD.apply_legacy_defaults(dict(c1["values"])) == []   # 全键在场 -> 无回填
    legacy = dict(c1["values"]); legacy.pop("refine_head")
    filled = TD.apply_legacy_defaults(legacy)
    assert filled == ["refine_head"] and legacy["refine_head"] == 0
    keep = {"refine_head": 1, "router": "ec", "gating": "raw",
            "improve_criterion": "rel", "improve_tol": 1e-3, "rand_offset": 1}
    assert TD.apply_legacy_defaults(keep) == [] and keep["router"] == "ec"  # 不覆盖已有值


def test_build_regressor_wiring():
    """build_regressor 按合同名字推切片索引; 与 jit_moe_config(路由开关)互不影响。"""
    cfg = {"arch": "jit", "patch": 16, "hidden": 64, "depth": 2, "heads": 4,
           "mlp_ratio": 4.0, "bottleneck": 16, "attn_dropout": 0.0, "proj_dropout": 0.0,
           "patch_margin": 0, "moe": False, "mode": "history_51", "refine_head": 1}
    m = TD.build_regressor(C.cond_channels("history_51"), 1, cfg, hw=HW)
    layout = C.cond_layout("history_51")
    assert m.refine_static_idx == [layout.index(n) for n in C.STATIC_ORDER]
    cfg2 = dict(cfg, refine_head=0, moe=True, experts=4, experts_per_tok=2,
                moe_intermediate=0, routed_scaling=1.0, moe_all_layers=False,
                moe_no_shared=False, router="ec", gating="norm")
    m2 = TD.build_regressor(C.cond_channels("history_51"), 1, cfg2, hw=HW)
    assert m2.refine is None and m2.moe_layers()[0].router_mode == "ec"


def main():
    tests = [test_off_is_bytewise_status_quo, test_on_starts_as_identity,
             test_grad_reaches_head, test_static_slice_matters,
             test_resume_rejects_toggle, test_build_regressor_wiring]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
