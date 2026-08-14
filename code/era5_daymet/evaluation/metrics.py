#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
metrics.py — 评测侧的打分与分析原语(纯 numpy, 不依赖 torch 与训练循环)
============================================================================
集合预报的 CRPS 与名次直方图、径向功率谱、分析窗口选取。这些量只在评测与诊断中使用,
与训练循环无关, 因此独立成模块: 改指标口径不必触碰训练代码。

口径要点:
  - CRPS 用公平估计式 E|x-y| - 0.5*E|x-x'|; 确定性方法(N=1)时第二项为 0, CRPS 恒等于 MAE;
  - 逐日 CRPS/MAE 取平均与全池等价, 但 RMSE 不等价, 跨日聚合前须确认用的是哪一种;
  - 所有函数都在陆地掩膜内取值, 掩膜外不参与统计。
============================================================================
"""
import numpy as np


def crps_ensemble(members, truth, mask, per_pixel=False):
    """集合 CRPS 的公平估计式 E|x-y| - 0.5*E|x-x'|, 逐通道在掩膜内取均值。

    per_pixel=True 时改为返回 (标量列表, 逐像素场), 场形状 (C,)+mask.shape, 掩膜外为 NaN。
    场与标量出自同一份计算, 标量恒等于场在掩膜内的均值。需要按区域、海拔或其他空间分层
    聚合 CRPS 时必须取场: 标量已经把空间维平掉, 事后无法再拆回各分层的贡献。
    """
    N = members.shape[0]; mb = mask > 0.5; out = []; fields = []
    for c in range(truth.shape[0]):
        mem = members[:, c][:, mb]; y = truth[c][mb]
        t1 = np.abs(mem - y[None]).mean(0)
        t2 = np.abs(mem[:, None] - mem[None, :]).mean((0, 1)) if N > 1 else 0.0
        px = t1 - 0.5 * t2
        out.append(float(px.mean()))
        if per_pixel:
            f = np.full(mb.shape, np.nan)
            f[mb] = px
            fields.append(f)
    return (out, np.stack(fields, 0)) if per_pixel else out


def rank_hist(members, truth, mask):
    """名次直方图: 真值在集合成员中的名次分布, 平坦即集合离散度校准良好。

    并列名次按随机整数打散, 避免离散取值(如降水零值)在某一档堆积成假峰。
    """
    N = members.shape[0]; mb = mask > 0.5
    mem = members[:, 0][:, mb]; y = truth[0][mb]
    ranks = (mem < y[None]).sum(0) + np.random.randint(0, 2, y.shape) * (mem == y[None]).sum(0)
    h, _ = np.histogram(ranks, bins=np.arange(N + 2))
    return (h / h.sum())


def radial_psd(img, sz):
    """径向平均功率谱: 去均值 + 汉宁窗后作二维 FFT, 按到中心的整数半径分箱平均。

    加窗是必需的: 不加窗时图像边界的不连续会在谱上引入十字形泄漏, 淹没高波数的真实差异。
    """
    img = img - img.mean(); w = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * w))) ** 2
    c0, c1 = np.array(P.shape) // 2; Y, X = np.indices(P.shape); r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def pick_land_box(mask, sz):
    """在掩膜内找一个 sz x sz 的全陆地方框, 返回 (y0, x0, sz)。

    用于需要纯陆地窗口的分析(如谱检验): 窗口内混入海洋会把常数填充区的零方差带进统计。
    步长 40 是覆盖率与搜索成本的折中; 找不到全陆地窗口时退回掩膜左上角。
    """
    ys, xs = np.where(mask)
    if len(ys) == 0: return (0, 0, sz)
    for y0 in range(ys.min(), max(ys.min() + 1, ys.max() - sz), 40):
        for x0 in range(xs.min(), max(xs.min() + 1, xs.max() - sz), 40):
            if mask[y0:y0 + sz, x0:x0 + sz].all(): return (y0, x0, sz)
    return (int(ys.min()), int(xs.min()), sz)
