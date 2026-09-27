#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pos_embed.py — 2D sin-cos 固定位置编码。

从上一代的 models/vit.py 原样摘出这一个函数: 它自足(只依赖 torch), 而整搬 vit.py 会带进
本线不用的 ViT 实现和一个未搬运的 seq_parallel 依赖。
"""
import torch


def get_2d_sincos_pos_embed(dim, gh, gw, device=None, dtype=None):
    """2D sin-cos 固定位置编码 (MAE 风格): 无参数、任意分辨率通用 -> 整幅训练不再需要
    绑尺寸的可学习 pos (整幅 259200 token 的可学习位置编码裸参数达 ~1 亿)。
    dim 一半编码 h 轴、一半编码 w 轴; 每轴 dim/4 个几何频率的 sin/cos 拼接。要求 dim%4==0。
    token 顺序与 embed 后 flatten(2) 一致(h 外 w 内, 即 idx=h*gw+w)。返回 (1, gh*gw, dim)。"""
    assert dim % 4 == 0, f"sincos pos embed 需 dim%4==0, 得到 dim={dim}"
    d4 = dim // 4
    omega = torch.arange(d4, device=device, dtype=torch.float32) / d4
    omega = 1.0 / (10000.0 ** omega)                                   # (d4,) 频率
    y = torch.arange(gh, device=device, dtype=torch.float32)
    x = torch.arange(gw, device=device, dtype=torch.float32)
    ey = torch.einsum("i,j->ij", y, omega)                             # (gh, d4)
    ex = torch.einsum("i,j->ij", x, omega)                             # (gw, d4)
    ey = torch.cat([ey.sin(), ey.cos()], dim=1)                        # (gh, dim/2)
    ex = torch.cat([ex.sin(), ex.cos()], dim=1)                        # (gw, dim/2)
    ey = ey[:, None, :].expand(gh, gw, dim // 2)
    ex = ex[None, :, :].expand(gh, gw, dim // 2)
    emb = torch.cat([ey, ex], dim=2).reshape(1, gh * gw, dim)          # (1, gh*gw, dim)
    return emb.to(dtype) if dtype is not None else emb
