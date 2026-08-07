#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_model_contracts.py — 已训权重与网络定义之间的结构契约
============================================================================
网络主体从训练脚本搬进 models/ 之后, 只要 state_dict 的键集合或形状发生任何变化, 已经
训好的 checkpoint 就再也装不回去 —— 而这类改动往往在重构时悄无声息地发生, 不报错, 直到
需要复现或续训时才暴露。此处把契约钉死: 用真实 checkpoint 以 strict=True 装载。

不锁前向输出的哈希: 那会随 torch 版本、后端算子实现而变, 产生与结构无关的假警报。键集合
与形状则与版本无关, 且正是"能否装回去"的充要条件。

缺少某个 checkpoint 时跳过该项而非失败 —— 本测试要能在只有部分实验产物的机器上运行。

用法:
    python -m era5_daymet.tests.test_model_contracts
============================================================================
"""
from pathlib import Path

import torch

from era5_daymet.training import train_downscale as TD

RUNS = Path(__file__).resolve().parents[3] / "runs/exp"
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def _args_of(ck):
    a = ck.get("args", {})
    return a if isinstance(a, dict) else vars(a)


def _weights(ck):
    sd = ck.get("model", ck.get("state_dict"))
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load(path):
    p = RUNS / path
    if not p.exists():
        print(f"  [SKIP] {path} 不存在")
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def test_vit(path, expect_params):
    ck = _load(path)
    if ck is None:
        return
    from era5_daymet.models.vit import ViT
    a, sd = _args_of(ck), _weights(ck)
    cin = TD.cond_channels(a["in_vars"], a["out_vars"], a.get("use_clim", False))
    m = ViT(cin, len(a["out_vars"]), img=a["patch"], patch=a["vit_patch"], dim=a["dim"],
            depth=a["depth"], heads=a["heads"], mlp=a["mlp"], pos_type=a["pos_type"],
            head_up=a["head_up"], full_frame=a["full_frame"])
    try:
        m.load_state_dict(sd, strict=True)
        ok, why = True, f"{sum(q.numel() for q in m.parameters()):,} 参数"
    except RuntimeError as e:
        ok, why = False, str(e).split("\n")[1].strip() if "\n" in str(e) else str(e)
    check(f"ViT {path}", ok, why)
    if ok:
        n = sum(q.numel() for q in m.parameters())
        check(f"ViT {path} 参数量未漂移", n == expect_params, f"{n:,} (期望 {expect_params:,})")


def test_regressor(path):
    """阶段A / UNet 基线: 经 build_regressor 走与训练、评测同一条构建路径。"""
    ck = _load(path)
    if ck is None:
        return
    a, sd = _args_of(ck), _weights(ck)
    cin = TD.cond_channels(a["in_vars"], a["out_vars"], a.get("use_clim", False))
    m = TD.build_regressor(cin, len(a["out_vars"]), a)
    try:
        m.load_state_dict(sd, strict=True)
        ok, why = True, f"{sum(q.numel() for q in m.parameters()):,} 参数 " \
                        f"({type(m).__name__}, arch={a.get('arch', 'unet')})"
    except RuntimeError as e:
        ok, why = False, str(e).split("\n")[1].strip() if "\n" in str(e) else str(e)
    check(f"回归主体 {path}", ok, why)


def main():
    print("已训权重能否装回当前网络定义 (strict=True):")
    test_vit("20260729-vit-fullframe-resume3/ckpt.pt", 15_192_387)
    for p in ["20260727-unet-U3-fullframe-16node-v4/ckpt.pt",
              "20260801-corrdiffA-C1-tmax/ckpt.pt",
              "20260802-corrdiffA-C2-tmin/ckpt.pt",
              "20260802-corrdiffA-C3-precip/ckpt.pt"]:
        test_regressor(p)

    print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
