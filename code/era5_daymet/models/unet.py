#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
unet.py — 确定性 UNet 主体、可选位置网格与单阶段 DDPM 组件
============================================================================
ERA5 -> Daymet 降尺度的确定性回归主体。全卷积, 2 次下采样(720x1440 -> 360x720 -> 180x360),
每级 1 个 ResBlock, GroupNorm(8), 无注意力。规模只由 base 控制: 64 / 128 / 192 分别对应
1.59M / 6.33M / 14.21M 参数。

`UNet` 是全卷积的, 因此同一份权重能吃任意尺寸输入(整幅或裁块); 但 GroupNorm 的统计量是
逐样本在 (C/G, H, W) 上算的, **依赖空间窗口大小** —— 整幅训练出来的权重在裁块上前向,
得到的不是同一个函数。需要裁块结果时, 应在整幅上前向后再裁, 而不是把裁块喂进来。

`Diffusion` 是单阶段条件 DDPM(eps-prediction, 线性 beta, DDIM 采样), 与两阶段残差扩散
无关, 仅供旧的 `--model diffusion` 路径使用。
============================================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DOMAIN_HW = (720, 1440)          # Daymet 目标网格; 位置网格按它归一化, 与当前输入尺寸无关


def sinusoidal_emb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb=0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(temb, cout) if temb else None
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, t=None):
        h = self.c1(F.silu(self.n1(x)))
        if self.emb is not None and t is not None:
            h = h + self.emb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


def sincos_pos_grid(H, W, y0, x0, domain, device, dtype):
    """全域正弦位置网格, 4 通道: [sin(行), cos(行), sin(列), cos(列)]。

    角度 = 该像素在**全域**中的归一化坐标 × 2π。sin/cos 成对可唯一还原 [0,2π) 内的角度,
    所以 4 个通道就给出全域唯一的位置指纹 —— 纯卷积是平移等变的, 单靠卷积无法表达
    "这一像素在域内的绝对位置", 这几个通道正是补上这一维。

    归一化除的是 domain 而不是当前输入的 H/W: 裁块时传入该块左上角 (y0,x0), 得到的网格
    与整幅落在同一坐标系, 保证裁块训练与整幅推理的位置口径一致。
    """
    Hg, Wg = domain
    ry = (torch.arange(H, device=device, dtype=torch.float32) + y0) * (2 * math.pi / Hg)
    rx = (torch.arange(W, device=device, dtype=torch.float32) + x0) * (2 * math.pi / Wg)
    one_w = torch.ones(W, device=device)
    one_h = torch.ones(H, device=device)
    g = torch.stack([torch.outer(ry.sin(), one_w), torch.outer(ry.cos(), one_w),
                     torch.outer(one_h, rx.sin()), torch.outer(one_h, rx.cos())], 0)
    return g.to(dtype)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64, temb=0, pos_grid=0, domain=DOMAIN_HW):
        super().__init__()
        self.temb = temb
        self.pos_grid = int(pos_grid)
        if self.pos_grid not in (0, 4):
            raise ValueError(f"pos_grid 只支持 0(关闭) 或 4(正弦), 收到 {pos_grid}")
        self.domain = (int(domain[0]), int(domain[1]))
        self._pos_cache = {}          # 位置网格只依赖形状/位置, 缓存复用; 不进 state_dict
        if temb:
            self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inc = nn.Conv2d(in_ch + self.pos_grid, base, 3, padding=1)
        self.d1 = ResBlock(base, base, temb); self.p1 = nn.Conv2d(base, base, 4, 2, 1)
        self.d2 = ResBlock(base, base * 2, temb); self.p2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
        self.mid = ResBlock(base * 2, base * 2, temb)
        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1); self.r2 = ResBlock(base * 4, base, temb)
        self.u1 = nn.ConvTranspose2d(base, base, 4, 2, 1); self.r1 = ResBlock(base * 2, base, temb)
        self.outc = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, out_ch, 3, padding=1))
    def _pos(self, x, origin):
        key = (x.shape[-2], x.shape[-1], int(origin[0]), int(origin[1]), x.device, x.dtype)
        g = self._pos_cache.get(key)
        if g is None:
            g = sincos_pos_grid(x.shape[-2], x.shape[-1], origin[0], origin[1],
                                self.domain, x.device, x.dtype)
            self._pos_cache = {key: g}          # 只留最近一个形状, 避免逐 patch 位置堆积显存
        return g.expand(x.shape[0], -1, -1, -1)

    def forward(self, x, t=None, origin=(0, 0)):
        if self.pos_grid:
            x = torch.cat([x, self._pos(x, origin)], 1)
        te = self.tmlp(sinusoidal_emb(t, self.temb)) if self.temb else None
        x0 = self.d1(self.inc(x), te)
        x1 = self.d2(self.p1(x0), te)
        xm = self.mid(self.p2(x1), te)
        h = self.r2(torch.cat([self.u2(xm), x1], 1), te)
        h = self.r1(torch.cat([self.u1(h), x0], 1), te)
        return self.outc(h)


def masked_mse(pred, tgt, mask):
    d = (pred - tgt) ** 2 * mask
    return d.sum() / (mask.sum() * pred.size(1) + 1e-6)


class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        b = torch.linspace(1e-4, 0.02, T)
        self.T = T; self.ac = torch.cumprod(1 - b, 0).to(device)
    def loss(self, model, x0, cond, mask):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        ac = self.ac[t][:, None, None, None]
        xt = ac.sqrt() * x0 + (1 - ac).sqrt() * noise
        pred = model(torch.cat([xt, cond], 1), t)
        return masked_mse(pred, noise, mask)
    @torch.no_grad()
    def ddim(self, model, cond, shape, steps=50):
        dev = cond.device; x = torch.randn(shape, device=dev)
        ts = torch.linspace(self.T - 1, 0, steps).long().to(dev)
        for i, t in enumerate(ts):
            tb = torch.full((shape[0],), int(t), device=dev)
            eps = model(torch.cat([x, cond], 1), tb); ac = self.ac[t]
            x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
            x = (self.ac[ts[i + 1]].sqrt() * x0 + (1 - self.ac[ts[i + 1]]).sqrt() * eps) if i < len(ts) - 1 else x0
        return x
