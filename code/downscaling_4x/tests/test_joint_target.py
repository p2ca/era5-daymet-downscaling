#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三目标联合训练(--target all)的检验: 输出通道数、逐通道梯度、形状守卫与断点契约。

联合训练最危险的静默错误是广播: (B,3,H,W) 的预测对上 (B,1,H,W) 的目标, MSE 照算不误、
loss 正常下降、作业正常结束, 只是三个输出通道都在拟合同一个目标。这里要求形状不一致
当场抛错, 并用负对照验证守卫与逐通道梯度检查各自真有分辨力。

需要真实数据(联合冒烟一节), 纯 CPU, 约一分钟。

运行: python -m downscaling_4x.tests.test_joint_target
"""
import os
import shutil
import tempfile

import torch

from downscaling_4x import contract as C
from downscaling_4x.models.unet import UNet, masked_mse
from downscaling_4x.training import train_downscale as TD
from downscaling_4x.tests.test_train_resume import SINGLE, make_args


def test_target_selection():
    """'all' 给全通道切片与 cout=3; 单目标给保通道维的一条切片与 cout=1。"""
    sel, cout = TD.target_selection("all")
    assert cout == len(C.TARGETS)
    t = torch.zeros(2, len(C.TARGETS), 4, 4)
    assert t[:, sel].shape == t.shape
    for i, name in enumerate(C.TARGETS):
        sel, cout = TD.target_selection(name)
        assert cout == 1
        assert t[:, sel].shape == (2, 1, 4, 4), "单目标切片必须保住通道维"


def test_shape_guard_has_discrimination():
    """形状不一致必须抛错; 一致时数值与 masked_mse 完全相同。"""
    p = torch.randn(2, 3, 8, 8)
    m = torch.ones(2, 1, 8, 8)
    try:
        TD.masked_mse_strict(p, torch.randn(2, 1, 8, 8), m)
    except RuntimeError as e:
        assert "形状" in str(e)
    else:
        raise AssertionError("(B,3) 预测对 (B,1) 目标未被拒绝 —— 广播正在吞掉通道错配")
    t3 = torch.randn(2, 3, 8, 8)
    v = TD.masked_mse_strict(p, t3, m)
    assert torch.isfinite(v) and torch.equal(v, masked_mse(p, t3, m))


def test_grad_reaches_every_output_channel():
    """联合 loss 的梯度必须到达全部输出通道; 负对照证明该检查有分辨力。"""
    torch.manual_seed(0)
    net = UNet(5, 3, base=16)
    cond = torch.randn(2, 5, 32, 32)
    tgt = torch.randn(2, 3, 32, 32)
    m = (torch.rand(2, 1, 32, 32) > 0.3).float()

    TD.masked_mse_strict(net(cond), tgt, m).backward()
    per_ch = net.outc[2].weight.grad.abs().flatten(1).sum(1)
    assert (per_ch > 0).all(), f"联合 loss 的梯度未到达全部输出通道: {per_ch.tolist()}"

    # 负对照: 只拿通道 0 算 loss, 其余输出通道的梯度必须精确为 0
    net.zero_grad()
    TD.masked_mse_strict(net(cond)[:, :1], tgt[:, :1], m).backward()
    per_ch = net.outc[2].weight.grad.abs().flatten(1).sum(1)
    assert per_ch[0] > 0 and per_ch[1] == 0 and per_ch[2] == 0, \
        f"逐通道梯度检查无分辨力: {per_ch.tolist()}"


def test_joint_smoke_and_ckpt_contract():
    """--target all 端到端冒烟: ckpt 记录 all 且输出 3 通道; 联合断点拒绝单目标续训。"""
    root = tempfile.mkdtemp(prefix="joint-")
    try:
        d = os.path.join(root, "a")
        TD.fit(make_args(d, 1, target="all"), SINGLE)
        ck = torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu", weights_only=False)
        assert ck["args"]["target"] == "all"
        w = ck["model"]["outc.2.weight"]
        assert w.shape[0] == len(C.TARGETS), f"联合模型输出通道 {w.shape[0]} != {len(C.TARGETS)}"
        try:
            TD.fit(make_args(d, 2, resume_from=os.path.join(d, "last.pt"),
                             target=C.TARGETS[0]), SINGLE)
        except RuntimeError as e:
            assert "不一致" in str(e)
        else:
            raise AssertionError("target=all 的断点被单目标续训接受")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    tests = [
        test_target_selection,
        test_shape_guard_has_discrimination,
        test_grad_reaches_every_output_channel,
        test_joint_smoke_and_ckpt_contract,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
