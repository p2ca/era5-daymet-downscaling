#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_mean.py — 把缓存的 μ 接入官方残差损失
============================================================================
官方 `ResidualLoss` 通过 `regression_net(zeros, y_lr)` 取全域回归均值 μ, 之后在全域上
形成残差、再切 patch。这里提供一个外观与回归网一致、但直接返回缓存值的壳, 从而在
**完全不改动移植来的损失代码**的前提下省掉每步的全域前向。

保持官方顺序不变是关键: μ 必须在全域算好后再裁块。若反过来把 192x192 的条件喂进回归网,
GroupNorm 的统计量与卷积边界都会变, 实测偏差可达阶段 A 自身误差的 3-4 倍。
============================================================================
"""
import torch


class CachedRegressionMean(torch.nn.Module):
    """伪装成阶段 A 回归网, 返回本步预取的全域 μ。

    调用方每步先 `set(mu)` 再走损失; 形状/设备不匹配一律抛错, 避免静默错配。
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = int(out_channels)
        self._mu = None

    def set(self, mu):
        if mu.ndim != 4 or mu.shape[1] != self.out_channels:
            raise ValueError(
                f"μ 形状应为 (B,{self.out_channels},H,W), 收到 {tuple(mu.shape)}")
        self._mu = mu

    def clear(self):
        self._mu = None

    def forward(self, x, y_lr, **kwargs):
        if self._mu is None:
            raise RuntimeError("本步未设置缓存 μ; 调用损失前需先 set()")
        if self._mu.shape[0] != x.shape[0] or self._mu.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"μ 与当前 batch 不匹配: μ={tuple(self._mu.shape)}, x={tuple(x.shape)}")
        if self._mu.device != x.device:
            raise ValueError(f"μ 在 {self._mu.device}, 而输入在 {x.device}")
        return self._mu.to(x.dtype)


def pin_ocean(mu, land_mask):
    """把 μ 在非陆地格点上钉为 0, 再用于构造残差。

    训练损失只监督陆地, 海洋上的 μ 从未被约束; 若原样进入残差, 扩散网会去拟合一片
    无意义的值, 并通过卷积感受野污染沿海陆地。残差空间里 0 表示"均值已说完, 无附加"。
    """
    return mu * land_mask
