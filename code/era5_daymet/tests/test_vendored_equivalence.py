#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_vendored_equivalence.py — 移植代码与官方实现的等价性回归测试
============================================================================
`models/` 下的 song_unet / preconditioning / patching / stochastic_sampler 由官方
PhysicsNeMo 源码移植而来, 只去掉了与计算无关的包依赖。移植当时以"同权重下输出逐比特
相同"验收, 本测试就是那道验收。

这些文件允许修改 —— 上游本身有缺陷, 逐比特跟随上游并非目的。但**每一处有意偏离都必须
登记进下面的 INTENTIONAL_DEVIATIONS**, 使"我们与上游差在哪里"成为一份可执行、被审查的
清单。未登记的偏离会让对照项当场失败。

参照来源优先取仓库内的快照, 其次才是外部源码树: 外部路径属于他人管理的只读区, 可能在
我们不知情时消失, 而"参照没了"绝不能表现为"测试全绿"。官方包缺可选依赖, 按需打桩,
只为取到网络定义。

用法:
    python -m era5_daymet.tests.test_vendored_equivalence
============================================================================
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
# 参照来源, 按优先级排列。仓库内快照与上游逐字节一致, 但按 physicsnemo/models_diffusion/
# 平铺存放, 需要搭一层命名空间才能导入(见 _load_snapshot); 外部源码树则是原生布局。
SNAPSHOT_ROOT = _HERE.parent / "reference_corrdiff_official"
EXTERNAL_ROOT = Path("/lustre/orion/atm112/proj-shared/patrickfan/physicsnemo")

# 我们相对上游的有意偏离: {模块路径: 偏离说明}。空表示当前与上游一致。
# 改动移植来的文件时必须在此登记, 否则对照项会失败而不知其所以然。
INTENTIONAL_DEVIATIONS = {
    "models/corrdiff_loss.py":
        "ResidualLoss 新增 n_constant_cond: 拼给每个 patch 的全域上下文副本排除条件张量"
        "末尾的空间常数通道(其副本与 patch 内那份逐点相同, 贡献恒为零)。默认取数据合同的"
        "TIME_ORDER 长度; 传 0 复现上游行为。",
    "models/stochastic_sampler.py":
        "stochastic_sampler 新增同名参数, 与训练侧共用 corrdiff_loss._global_context —— "
        "两条路径的条件宽度必须一致, 只改一边会让权重装不回去且训练时无任何提示。",
    "models/preconditioning.py":
        "EDMPrecondSuperResolution.forward 补上 class_labels(排在 force_fp32 之前)并透传给"
        "网络, 与同族的 EDMPrecond 等一致。上游此处缺该参数, 而采样器按 "
        "net(x, x_lr, t_hat, class_labels, ...) 传第 4 个位置参数, 标签会落进 force_fp32: "
        "恒为 None 时碰巧无害, 真传标签则静默强制 fp32 且标签丢失。"
        "label_dim=0(默认)时归一为 None, 默认路径与上游逐比特相同。",
}

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


class _Anything:
    """万能占位: 可当函数调用、可取任意属性、可作上下文管理器。用于 nvtx 一类的可选依赖。"""

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_ANY = _Anything()


class _StubMetaData:
    """占位 ModelMetaData: 上游是 dataclass, 子类只在其上追加字段, 空基类即可。"""


class _StubModule(torch.nn.Module):
    """占位 Module: 上游它是 nn.Module 子类且接受 meta= 关键字; 必须是真类型才能当基类,
    也必须是 nn.Module 才能取 parameters() 做参数量对照。"""

    def __init__(self, meta=None, *a, **k):
        super().__init__()


def _stub(name, **attrs):
    """给缺失的模块打一个空壳, 只为让网络定义能被导入。"""
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    if not attrs:
        m.__getattr__ = lambda k: _ANY
    sys.modules[name] = m


def _clear_physicsnemo():
    """清掉 physicsnemo 相关的模块记录。快照尝试会先塞进一批占位模块, 失败后若不清干净,
    后续从外部源码树导入真包时会被这些占位挡住 —— 表现为"两个来源都取不到"。"""
    for k in [k for k in sys.modules if k == "physicsnemo" or k.startswith("physicsnemo.")]:
        del sys.modules[k]


def _retry_with_stubs(load, tries=40):
    """反复调用 load(), 每遇到一个缺失模块就打桩后重试。"""
    for _ in range(tries):
        try:
            return load()
        except ModuleNotFoundError as e:
            _stub(e.name)
        except Exception:
            return None
    return None


def _load_snapshot():
    """加载仓库内快照。

    快照是平铺的 physicsnemo/models_diffusion/, 而其内部沿用上游的绝对导入路径
    physicsnemo.models.diffusion.*; 这里先搭出该命名空间再按文件加载, 从而不必改动
    快照本身 —— 快照与上游逐字节一致才有参照价值。
    """
    pkg_dir = SNAPSHOT_ROOT / "physicsnemo" / "models_diffusion"
    if not (pkg_dir / "__init__.py").exists():
        return None
    # 快照只复制了 models_diffusion/, 上游的两个基类不在其中, 按其真实语义补上。
    _stub("physicsnemo")
    _stub("physicsnemo.models")
    _stub("physicsnemo.models.meta", ModelMetaData=_StubMetaData)
    _stub("physicsnemo.models.module", Module=_StubModule)

    def load():
        spec = importlib.util.spec_from_file_location(
            "physicsnemo.models.diffusion", pkg_dir / "__init__.py",
            submodule_search_locations=[str(pkg_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["physicsnemo.models.diffusion"] = mod
        spec.loader.exec_module(mod)
        return mod

    mod = _retry_with_stubs(load)
    if mod is None:
        _clear_physicsnemo()
    return mod


def _load_external():
    """加载外部源码树(原生布局)。"""
    if not EXTERNAL_ROOT.exists():
        return None
    sys.path.insert(0, str(EXTERNAL_ROOT))

    def load():
        import physicsnemo.models.diffusion as d
        return d

    return _retry_with_stubs(load)


def _import_official():
    """返回 (官方模块, 来源说明); 两个来源都取不到时返回 (None, None)。"""
    mod = _load_snapshot()
    if mod is not None:
        return mod, f"仓库内快照 {SNAPSHOT_ROOT.name}/"
    mod = _load_external()
    if mod is not None:
        return mod, f"外部源码树 {EXTERNAL_ROOT}"
    return None, None


def test_patch_fuse_identity():
    """§5.5 P0 / §6.1: patch -> fuse 必须还原原张量, max|误差| < 1e-6。"""
    from era5_daymet.models.patching import GridPatching2D
    print("patch->fuse 恒等 (不依赖官方源码):")
    for img in [(720, 1440), (360, 720)]:
        for ps, ov, bd in [(192, 48, 2), (192, 96, 2), (192, 96, 8), (192, 4, 2), (256, 48, 2)]:
            p = GridPatching2D(img_shape=img, patch_shape=(ps, ps),
                               overlap_pix=ov, boundary_pix=bd)
            x = torch.randn(2, 3, *img)
            fu = p.fuse(p.apply(x), batch_size=2)
            err = float((fu - x).abs().max()) if fu.shape == x.shape else float("inf")
            check(f"{img} patch={ps} ov={ov} bd={bd}", err < 1e-6, f"max|err|={err:.2e}")


def _pair_equal(off_cls, our_cls, kwargs, forward, tag, seeds=(7, 99)):
    torch.manual_seed(0); a = off_cls(**kwargs).eval()
    torch.manual_seed(0); b = our_cls(**kwargs).eval()
    extra = sorted(set(b.state_dict()) - set(a.state_dict()))
    if extra:
        check(f"{tag}: 移植无多余参数", False, f"多出 {extra}")
        return
    sd = {k: v for k, v in a.state_dict().items() if not k.endswith("device_buffer")}
    b.load_state_dict(sd)
    na = sum(q.numel() for q in a.parameters()); nb = sum(q.numel() for q in b.parameters())
    check(f"{tag}: 参数量一致", na == nb, f"{na:,} vs {nb:,}")
    for s in seeds:
        torch.manual_seed(s)
        args = forward()
        with torch.no_grad():
            ya, yb = a(*args), b(*args)
        check(f"{tag}: seed={s} 逐比特相同", torch.equal(ya, yb),
              f"max|diff|={float((ya-yb).abs().max()):.2e}")


def test_against_official(off):
    from era5_daymet.models.song_unet import UNet as OurUNet
    from era5_daymet.models.preconditioning import (
        EDMPrecondSuperResolution as OurPrec)
    res = [144, 288]
    print("\n阶段A 回归包装 (骨干 = DDPM++/NCSN++ + 位置网格):")
    _pair_equal(off.UNet, OurUNet,
                dict(img_resolution=res, img_in_channels=24, img_out_channels=1,
                     model_type="SongUNetPosEmbd", model_channels=64,
                     channel_mult=[1, 2, 2, 2, 2], attn_resolutions=[16],
                     N_grid_channels=4, gridtype="sinusoidal", embedding_type="zero"),
                lambda: (torch.zeros(1, 1, *res), torch.randn(1, 20, *res)),
                "阶段A UNet")
    print("\n阶段B EDM 预条件:")
    _pair_equal(off.EDMPrecondSuperResolution, OurPrec,
                dict(img_resolution=192, img_in_channels=141, img_out_channels=1,
                     model_type="SongUNetPosEmbd", model_channels=64,
                     channel_mult=[1, 2, 2], attn_resolutions=[16],
                     N_grid_channels=100, gridtype="learnable"),
                lambda: (torch.randn(2, 1, 192, 192), torch.randn(2, 41, 192, 192),
                         torch.full((2, 1, 1, 1), 2.0)),
                "阶段B EDMPrecondSR")


def main():
    test_patch_fuse_identity()
    if INTENTIONAL_DEVIATIONS:
        print("\n已登记的有意偏离:")
        for mod, why in INTENTIONAL_DEVIATIONS.items():
            print(f"  - {mod}: {why}")
    off, src = _import_official()
    if off is None:
        # 参照缺失不能表现为全绿: 对照项没跑过, 结论就不成立。
        print("\n[未执行] 逐比特对照 —— 两个参照来源都取不到:")
        print(f"           {SNAPSHOT_ROOT}")
        print(f"           {EXTERNAL_ROOT}")
        print("\nPARTIAL: 自洽检查通过, 但与上游的逐比特对照未执行")
        return 2
    print(f"\n参照来源: {src}")
    test_against_official(off)
    print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
