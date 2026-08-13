#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_sampler.py — ODE 采样器的解析检验与端到端分布回收
============================================================================
1. 常值 x 预测下 dz/dt = (c-z)/(1-t) 有解析解 z(1)=c, 检验 Heun/Euler 步进实现
   (末端 clamp 只留极小残差);
2. 海洋钳制: land=0 区域轨迹收敛到 0;
3. generator 给定时采样逐位可复现;
4. 端到端: 用 jit_vloss 训练一个小 MLP 拟合二维高斯, 采样应回收其均值与方差 ——
   损失与采样器的时间约定/换算若有任何不一致, 此检验会失败。

Run: python -m era5_daymet.tests.test_jit_sampler
============================================================================
"""
import torch
import torch.nn as nn

from era5_daymet.models.jit_sampler import generate
from era5_daymet.training.train_jit import jit_vloss

torch.manual_seed(0)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


class ConstNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, z, t, cond):
        return torch.full_like(z, self.c)


class ToyNet(nn.Module):
    """(z, t) -> x 预测的二维玩具网络; cond 为 0 通道占位。"""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, z, t, cond):
        inp = torch.cat([z.reshape(z.shape[0], 2), t.reshape(-1, 1)], dim=1)
        return self.f(inp).reshape(-1, 2, 1, 1)


def main():
    cond = torch.zeros(4, 0, 6, 8)

    for method in ("heun", "euler"):
        z = generate(ConstNet(0.7), cond, out_ch=1, steps=50, method=method,
                     noise_scale=1.0)
        err = float((z - 0.7).abs().max())
        check(f"常值场解析收敛 ({method}, 残差 {err:.4f} < 0.1)", err < 0.1)

    land = torch.ones(1, 1, 6, 8); land[..., :3, :4] = 0.0
    z = generate(ConstNet(0.7), cond, steps=50, land=land, noise_scale=1.0)
    check("land=0 区域被钳向 0", float((z * (1 - land)).abs().max()) < 0.1)
    check("land=1 区域仍收敛到常值",
          float(((z - 0.7) * land).abs().max()) < 0.1)

    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    za = generate(ConstNet(0.3), cond, steps=8, generator=g1)
    zb = generate(ConstNet(0.3), cond, steps=8, generator=g2)
    check("同 generator 采样逐位一致", torch.equal(za, zb))

    # --- 端到端: 二维高斯的均值/方差回收 ---
    mean = torch.tensor([0.8, -0.5]).view(1, 2, 1, 1)
    std = 0.3
    net = ToyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    w = torch.ones(256, 1, 1, 1)
    for it in range(1500):
        y = mean + std * torch.randn(256, 2, 1, 1)
        loss = jit_vloss(net, y, torch.zeros(256, 0, 1, 1), w,
                         noise_scale=1.0, p_mean=-0.8, p_std=0.8, t_eps=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s = generate(net, torch.zeros(4096, 0, 1, 1), out_ch=2, steps=50,
                     noise_scale=1.0, generator=torch.Generator().manual_seed(5))
    m_err = float((s.mean(0).flatten() - mean.flatten()).abs().max())
    s_err = float((s.std(0).flatten() - std).abs().max())
    check(f"均值回收 (误差 {m_err:.3f} < 0.1)", m_err < 0.1)
    check(f"方差回收 (误差 {s_err:.3f} < 0.12)", s_err < 0.12)

    print("test_jit_sampler: 全部通过")


if __name__ == "__main__":
    main()
