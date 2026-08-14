#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
tiled_inference.py — 确定性模型的整帧 / 分块推理(羽化加权融合)
============================================================================
全卷积模型(UNet 系)可直接吃整帧; 固定 token 数的模型(ViT 系)必须分块, 分块就要处理
重叠区的融合。本模块只负责"把一个已训练模型跑成一张场", 训练与评测两侧共用:
训练循环的验证走它, 评测与诊断出图也走它, 两边必须是同一份实现, 否则 val 与 test
的数值不可比。
============================================================================
"""
import numpy as np

try:
    import torch
except ImportError:                              # 纯 numpy 环境下仍可导入本模块的窗口函数
    torch = None


def _feather_window(tile, ov):
    """羽化窗口 (tile,tile): 块边缘的 ov 像素线性降权, 内部为 1。

    为什么不用均匀平均: 均匀平均下重叠区的覆盖数从 1 跳到 2 是不连续的, 且每块在自己
    边缘处上下文最少、预测最差 -> 拼出规则的"一格一格"接缝。羽化窗口让每块只在重叠带里
    平滑过渡, 边缘降权、内部满权, 接缝被抹平。
    最小权重 = 1/(ov+1) > 0 (不到 0): 保证域边界(只被单块覆盖处)除法不为 0 且恢复原值。
    """
    r = (np.arange(ov) + 1) / (ov + 1)                     # (0,1) 之间, 不含端点
    w1 = np.ones(tile, np.float32)
    w1[:ov] = r; w1[-ov:] = r[::-1]
    return np.outer(w1, w1).astype(np.float32)             # (tile,tile)


def det_predict(model, cond, tile, device, tile_batch=32):
    """确定性预测 -> (Cout,H,W) numpy。tile=0: 整帧(全卷积 UNet); tile>0: 分块 + 羽化加权融合。

    tile>0 时按 tile_batch 攒批前向: 720x1440 在 tile=60 下数百块, 逐块 batch=1 会让 GPU
    空转在 kernel launch 上。攒批与逐块数值等价。
    融合用羽化窗口加权(见 _feather_window), 而非均匀平均 -> 消除分块接缝的"一格一格"。
    """
    if not tile:
        return model(cond)[0].detach().cpu().numpy()
    _, _, H, W = cond.shape; ov = max(tile // 4, 1); step = max(tile - ov, 1)
    win = _feather_window(tile, ov)                        # (tile,tile) 加权窗口
    ys = sorted(set(list(range(0, max(1, H - tile + 1), step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, max(1, W - tile + 1), step)) + [max(0, W - tile)]))
    coords = [(y0, x0) for y0 in ys for x0 in xs]
    out = wsum = None
    for i in range(0, len(coords), tile_batch):
        chunk = coords[i:i + tile_batch]
        crops = torch.cat([cond[:, :, y0:y0 + tile, x0:x0 + tile] for y0, x0 in chunk], 0)
        pred = model(crops).detach().cpu().numpy()
        if out is None:
            out = np.zeros((pred.shape[1], H, W), np.float32); wsum = np.zeros((H, W), np.float32)
        for (y0, x0), p in zip(chunk, pred):
            out[:, y0:y0 + tile, x0:x0 + tile] += p * win[None]      # 加权累加
            wsum[y0:y0 + tile, x0:x0 + tile] += win
    return out / np.maximum(wsum, 1e-6)[None]
