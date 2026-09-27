#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
metrics.py — 评测侧的打分与分析原语(纯 numpy, 不依赖 torch 与训练循环)
============================================================================
集合预报的 CRPS 与名次直方图、径向功率谱、分析窗口选取、陆地掩膜 SSIM。这些量只在评测与诊断中使用,
与训练循环无关, 因此独立成模块: 改指标口径不必触碰训练代码。

口径要点:
  - CRPS 用公平估计式 E|x-y| - 0.5*E|x-x'|; 确定性方法(N=1)时第二项为 0, CRPS 恒等于 MAE;
  - 逐日 CRPS/MAE 取平均与全池等价, 但 RMSE 不等价, 跨日聚合前须确认用的是哪一种;
  - 所有函数都在陆地掩膜内取值, 掩膜外不参与统计;
  - SSIM 对窗口、data_range 与掩膜腐蚀极敏感, 三项口径连同常量一并放在本模块, 不得分散。
============================================================================
"""
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.metrics import structural_similarity as _ssim

from downscaling_4x import contract as C


def precip_log_mm(mm, clip_mm=C.PRECIP_CLIP_MM):
    """mm/day -> 合同的 log1p(mm) 空间: 小于 clip_mm 的毛毛雨置零后 log1p。

    评测侧所有 log 空间的量(逐点指标、SSIM、crps_log 场、log 空间的图)都必须经这一个函数,
    与训练目标的 precip_fwd 同一条变换。直接写 log1p(max(x,0)) 会漏掉置零: 真值在干区是
    精确的 0, 预测的零点几毫米毛毛雨在 log 空间里就成了结构差, 而统计基线那条路径是置零的
    —— 两边数字各自都对, 放到一张表里却不是同一个量。
    """
    return C.precip_fwd(mm, clip_mm=clip_mm, scale=1.0)


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


SSIM_SIGMA = 1.5          # Wang et al. 原始 SSIM: 11x11 高斯窗, sigma=1.5
SSIM_ERODE = 5            # 陆地掩膜腐蚀半径(px): 海岸窗口会吃到填充值, 剔掉
SSIM_TRUNCATE = 3.5       # 高斯窗的截断倍数; sigma=1.5 时即 11x11, 与 skimage 调用一致


def eroded_land_mask(mb):
    """SSIM 用的陆地掩膜: 按 SSIM_ERODE 腐蚀, 使高斯窗不会跨过海岸线吃到填充值。
    腐蚀半径必须大于高斯窗半径, 两者一起改才有意义, 因此与 ssim_masked 放在同一处。"""
    return binary_erosion(mb, iterations=SSIM_ERODE)


def ssim_masked(pred, truth, mb, mb_er):
    """陆地上的 SSIM。

    ★三个必须交代的口径决定(SSIM 对这些极其敏感, 不写清楚数字就没法复现):

    1) 海洋怎么办: SSIM 是局部窗口算的, 窗口一旦跨过海岸线就会吃到无效值。
       这里把 pred 和 truth 的非陆地区域填成★同一个值★(当日陆地均值) ——
       两边填一样 -> 不引入任何人为的结构差异; 再把 SSIM 图只在★腐蚀过★的陆地
       掩膜上平均(腐蚀半径 5px > 高斯窗半径), 彻底避开被填充值污染的海岸窗口。
       (若填不同值, 或不腐蚀直接在全陆地上平均, 海岸带会凭空产生结构差, 分数失真。)

    2) data_range: SSIM 的 C1/C2 正比于动态范围 L。这里取★当日 truth 在陆地上的
       max-min★。同一天里所有方法共用同一个 L -> ★方法之间的比较是公平的★
       (这正是我们要的); 跨天的 L 不同, 但最后是对天平均, 不影响方法排序。

    3) 公式: 高斯窗(sigma=1.5) + use_sample_covariance=False, 即 Wang 2004 原版,
       不是 skimage 的 7x7 均匀窗默认值。
    """
    S = ssim_field(pred, truth, mb)
    if S is None:
        return 1.0                                     # 该日陆地上是常数场(退化), 定义为完全相似
    m = mb_er if mb_er.any() else mb
    return float(S[m].mean())


def ssim_field(pred, truth, mb):
    """上面那三条口径下的 SSIM 图(未在任何掩膜上平均), 形状同 pred; 退化日返回 None。

    ssim_masked 就是本函数的结果在腐蚀掩膜上取均值, 两者只有一份实现。需要把 SSIM 按区域、
    按日子集重新聚合时必须取图: 标量已经把空间维平掉, 事后无法再拆回各分层的贡献。
    data_range 仍取★当日全域 truth 的陆地 max-min★, 与子集无关 —— 同一天里所有方法、所有
    子集共用同一把尺子, 子集之间才可比。
    """
    fill = float(truth[mb].mean())
    p = np.where(mb, pred, fill).astype(np.float64)
    t = np.where(mb, truth, fill).astype(np.float64)
    dr = float(truth[mb].max() - truth[mb].min())
    if dr <= 0:
        return None
    _, S = _ssim(t, p, data_range=dr, gaussian_weights=True, sigma=SSIM_SIGMA,
                 use_sample_covariance=False, full=True)
    return S


def ssim_components(pred, truth, mb):
    """SSIM 的三个乘性分量场 (l, c, s), 三者逐点相乘即 ssim_field 的 SSIM 图。

    l 亮度项(局部均值是否对), c 对比度项(局部起伏幅度是否对), s 结构项(局部图案是否对齐)。
    口径与 ssim_field 完全一致: 非陆地两边填同一个值(当日陆地均值), data_range 取当日 truth
    在陆地上的 max-min, 高斯窗 sigma=SSIM_SIGMA 且不做样本协方差修正。需要说清"结构差在哪一项"
    时取本函数; 只要标量或整幅 SSIM 图时取 ssim_masked / ssim_field。
    """
    fill = float(truth[mb].mean())
    t = np.where(mb, truth, fill).astype(np.float64)
    p = np.where(mb, pred, fill).astype(np.float64)
    L = float(truth[mb].max() - truth[mb].min())
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    g = lambda a: gaussian_filter(a, SSIM_SIGMA, truncate=SSIM_TRUNCATE)
    ut, up = g(t), g(p)
    vt = np.maximum(g(t * t) - ut * ut, 0.0)
    vp = np.maximum(g(p * p) - up * up, 0.0)
    vtp = g(t * p) - ut * up
    st, sp = np.sqrt(vt), np.sqrt(vp)
    l = (2 * ut * up + C1) / (ut * ut + up * up + C1)
    c = (2 * st * sp + C2) / (vt + vp + C2)
    st_ = (vtp + C2 / 2) / (st * sp + C2 / 2)
    return l, c, st_
