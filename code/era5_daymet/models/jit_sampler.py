#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
jit_sampler.py — JiT 的 ODE 采样(x-prediction -> 速度换算)
============================================================================
与训练同一约定: t=0 为噪声端、t=1 为数据端, 插值 z = t*x + (1-t)*e。采样从
z ~ N(0, noise_scale^2 I) 出发, 沿 dz/dt = v 用固定步长积分到 t=1:

    v(z, t) = (x_hat(z, t) - z) / max(1 - t, t_eps)

前 steps-1 段用 Heun(两次网络前向), 最后一段用 Euler(t=1 处 v 无定义, Heun 的
终点修正取不到)。总网络前向数 NFE = 2*(steps-1) + 1。

land 非空时每次 x 预测后把海洋区钳到 0 —— 训练侧目标场在海洋恒为 0(见 train_jit),
此钳制使采样轨迹的海洋分布与训练分布一致, 防止无监督区域的漂移经注意力污染陆地。
============================================================================
"""
import torch


def _velocity(net, z, t, cond, t_eps, land):
    B = z.shape[0]
    tb = torch.full((B,), float(t), device=z.device)
    x_hat = net(z, tb, cond)
    if land is not None:
        x_hat = x_hat * land
    return (x_hat - z) / max(1.0 - float(t), t_eps)


@torch.no_grad()
def generate(net, cond, out_ch=1, steps=50, method="heun", noise_scale=1.0,
             t_eps=0.05, land=None, generator=None):
    """cond: (B, C, H, W) -> 采样目标场 (B, out_ch, H, W)。

    generator 给定时, 初始噪声由它产生(集合成员用不同种子的 generator 区分),
    网络前向不再消耗随机数, 故采样结果对 generator 完全可复现。
    """
    B, _, H, W = cond.shape
    z = noise_scale * torch.randn(B, out_ch, H, W, device=cond.device,
                                  dtype=cond.dtype, generator=generator)
    ts = torch.linspace(0.0, 1.0, steps + 1)
    for i in range(steps - 1):
        t, t_next = float(ts[i]), float(ts[i + 1])
        v1 = _velocity(net, z, t, cond, t_eps, land)
        if method == "heun":
            z_mid = z + (t_next - t) * v1
            v2 = _velocity(net, z_mid, t_next, cond, t_eps, land)
            z = z + (t_next - t) * 0.5 * (v1 + v2)
        elif method == "euler":
            z = z + (t_next - t) * v1
        else:
            raise ValueError(f"未知采样方法: {method}")
    t, t_next = float(ts[-2]), float(ts[-1])
    z = z + (t_next - t) * _velocity(net, z, t, cond, t_eps, land)   # 末段 Euler
    return z
