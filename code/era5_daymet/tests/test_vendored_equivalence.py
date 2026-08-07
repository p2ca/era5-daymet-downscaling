#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_vendored_equivalence.py — 移植代码与官方实现的等价性回归测试
============================================================================
`models/` 下的 song_unet / preconditioning / patching / stochastic_sampler 均由官方
PhysicsNeMo 源码移植而来, 只去掉了与计算无关的包依赖。移植的可信度完全建立在"同权重
下输出逐比特相同"这一条上, 因此把该验证固化为可重复执行的测试。

需要能访问官方源码树才能做对照; 缺失时跳过对照项, 仍执行不依赖官方的自洽检查
(patch->fuse 恒等)。官方包缺可选依赖, 此处按需打桩, 只为取到网络定义。

用法:
    python -m era5_daymet.tests.test_vendored_equivalence
============================================================================
"""
import sys
import types
from pathlib import Path

import torch

OFFICIAL_ROOT = Path("/lustre/orion/atm112/proj-shared/patrickfan/physicsnemo")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def _import_official():
    """导入官方模块; 缺可选依赖时打桩, 失败则返回 None。"""
    if not OFFICIAL_ROOT.exists():
        return None
    sys.path.insert(0, str(OFFICIAL_ROOT))
    for _ in range(40):
        try:
            import physicsnemo.models.diffusion as d
            return d
        except ModuleNotFoundError as e:
            m = types.ModuleType(e.name)
            m.__path__ = []
            m.__getattr__ = lambda k: types.SimpleNamespace()
            sys.modules[e.name] = m
        except Exception:
            return None
    return None


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
    off = _import_official()
    if off is None:
        print("\n(未找到官方源码树, 跳过逐比特对照)")
    else:
        test_against_official(off)
    print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
