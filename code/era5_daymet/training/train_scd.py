#!/usr/bin/env python
# Packaged implementation; code/train_scd.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
SCD — Scale-Consistent Decomposition (EDM 两阶段, 出 ensemble)
ERA5(120x240) -> Daymet(720x1440) 6x 降尺度 · 多节点 DDP
============================================================================
本脚本是 train_downscale.py 的"方法升级版", 直接复用它的数据管线(Stats /
DownscaleData / PatchDS)、归一化约定、和评测(CRPS / rank hist / 功率谱),
只替换核心算法。要点(对应你 doc 里的三部分):

  ┌ 阶段一  偏差算子 f_θ (在一致性尺度 κ 上工作, 确定性, 只出一张图)
  │   输入 : 粗化到 κ 的条件 (ERA5 + 气候态 + 静态)
  │   输出 : μ_coarse(去偏后的粗场) 和 σ_coarse(异方差不确定性图)
  │   监督 : 在 κ 上的 heteroscedastic 高斯 NLL  ←─ calibration-from-operator
  │          (+ 可选 CoBi 错位鲁棒项, 容忍 κ 残留的空间位移)
  │
  └ 阶段二  EDM 条件生成器 G_θ (域内超分, 产生 ensemble)
      backbone: Karras EDM (preconditioning + 噪声调度 + 二阶 Heun 采样)
      条件 : [baseline 条件] + 上采样(μ_coarse) (+ 上采样(σ_coarse))
      目标 : 归一化后的 Daymet 距平(与 baseline target 同空间)
      一致性: coarsen_κ(D_θ) ≈ μ_coarse, 按 1/σ² 加权 (σ 大处放手, σ 小处钉住)
      ensemble: 同一输入用 N 个不同噪声种子采样 N 次

为什么是 EDM 而不是 baseline 的 DDPM:
  * sigma_data≈1 与你"目标归一到单位方差"天然吻合;
  * Heun 二阶采样 18 步 ≈ DDPM 1000 步质量, 出 ensemble 便宜;
  * 与 CorrDiff/GenCast 同框架, 便于对照。

用法:
  python train_scd.py --smoke --stage both          # 合成数据自测(CPU 可跑)
  python train_scd.py --estimate-kappa              # 打印 R²-随尺度 曲线, 定 κ
  python train_scd.py --stage corrector --out runs/scd
  python train_scd.py --stage generator --out runs/scd \
                      --corrector-ckpt runs/scd/corrector.pt
  python train_scd.py --stage both --out runs/scd   # 先训 f_θ 再训 G_θ, 末尾评测
  python train_scd.py --eval-only --out runs/scd    # 用 corrector.pt+generator.pt 评测

依赖: torch (+ matplotlib 仅评测画图)。本机无 torch 时只能做语法/数据检查。
============================================================================
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.paths import PROJECT_ROOT

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:
    torch = None

FACTOR = TD.FACTOR
TARGETS = TD.TARGETS
PRECIP = TD.PRECIP


# ===========================================================================
# 1. κ 尺度算子: 面积平均粗化 / 双线性上采样
# ===========================================================================
def coarsen(x, k):
    """面积平均池化到 κ 尺度 (B,C,H,W)->(B,C,H/k,W/k); D_κ 算子的可微实现。"""
    if k == 1:
        return x
    return F.avg_pool2d(x, k, k)


def upsample(x, size):
    """双线性上采样回 HR (用于把粗场 μ/σ 接回生成器条件)。"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def split_corrector(out, Cout):
    """阶段一输出 -> (μ, logvar)。logvar 截断保证数值稳定。"""
    mu = out[:, :Cout]
    logvar = out[:, Cout:2 * Cout].clamp(-8.0, 8.0)
    return mu, logvar


def _resize_np(a, Hc, Wc):
    """整数比例的最近邻上采样 / 块平均下采样 (κ 诊断用, 不依赖 torch)。"""
    H, W = a.shape
    if (H, W) == (Hc, Wc):
        return a
    if Hc >= H and Hc % H == 0 and Wc % W == 0:            # 上采样: 整数重复
        return np.repeat(np.repeat(a, Hc // H, 0), Wc // W, 1)
    if H % Hc == 0 and W % Wc == 0:                        # 下采样: 块平均
        return a.reshape(Hc, H // Hc, Wc, W // Wc).mean((1, 3))
    yi = (np.arange(Hc) * H // Hc).clip(0, H - 1)          # 兜底: 最近邻索引
    xi = (np.arange(Wc) * W // Wc).clip(0, W - 1)
    return a[yi][:, xi]


# ===========================================================================
# 2. CoBi (Contextual Bilateral) 错位鲁棒损失
# ===========================================================================
def cobi_loss(pred, tgt, mask, win=5, w_spatial=0.1):
    """
    对每个 pred 像素, 在 tgt 的 win×win 邻域里找"特征最近 + 空间最近"的匹配,
    取该最小代价作为损失 —— 内容对应、但位置有小位移时不被惩罚(对应 doc 创新点①)。
    这是 Zhang et al. CVPR'19 CoBi 的轻量版(用原始通道而非 VGG 特征, 无额外依赖)。
    pred,tgt: (B,C,h,w);  mask: (B,1,h,w)。在 κ 尺度上算, 很便宜。
    """
    B, C, h, w = pred.shape
    pad = win // 2
    tgt_un = F.unfold(tgt, win, padding=pad)              # (B, C*win*win, h*w)
    tgt_un = tgt_un.view(B, C, win * win, h * w)
    feat = ((pred.view(B, C, 1, h * w) - tgt_un) ** 2).sum(1)   # (B, win*win, h*w) 特征 L2
    dy = torch.arange(win, device=pred.device) - pad
    sd = (dy[:, None] ** 2 + dy[None, :] ** 2).float().view(1, win * win, 1)   # 空间距离²
    dmin = (feat + w_spatial * sd).min(1).values.view(B, 1, h, w)  # 双边代价取窗内最小
    return (dmin * mask).sum() / (mask.sum() + 1e-6)


# ===========================================================================
# 3. EDM (Karras 2022): preconditioning + 噪声调度 + Heun 采样
# ===========================================================================
class EDM:
    def __init__(self, sigma_data=1.0, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                 P_mean=-1.2, P_std=1.2):
        self.sd = sigma_data
        self.smin, self.smax, self.rho = sigma_min, sigma_max, rho
        self.P_mean, self.P_std = P_mean, P_std

    def _D(self, net, x, sigma, cond):
        """去噪器 D_θ(x;σ,cond) = c_skip·x + c_out·F_θ(c_in·x; c_noise, cond); 输出干净 y 的估计。"""
        if not torch.is_tensor(sigma) or sigma.ndim == 0:
            sigma = torch.full((x.size(0),), float(sigma), device=x.device)
        s = sigma.view(-1, 1, 1, 1)
        c_skip = self.sd ** 2 / (s ** 2 + self.sd ** 2)
        c_out = s * self.sd / (s ** 2 + self.sd ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + self.sd ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        Fx = net(torch.cat([c_in * x, cond], 1), c_noise)
        return c_skip * x + c_out * Fx

    def loss(self, net, y, cond, mask):
        """EDM 去噪损失: 采 ln σ~N(P_mean,P_std), 加噪, 用 λ(σ) 加权 ‖D-y‖²。返回 (loss, D_estimate)。"""
        sigma = (torch.randn(y.size(0), device=y.device) * self.P_std + self.P_mean).exp()
        n = torch.randn_like(y) * sigma.view(-1, 1, 1, 1)
        D = self._D(net, y + n, sigma, cond)
        w = ((sigma ** 2 + self.sd ** 2) / (sigma * self.sd) ** 2).view(-1, 1, 1, 1)
        se = w * (D - y) ** 2 * mask
        return se.sum() / (mask.sum() * y.size(1) + 1e-6), D

    def sample(self, net, cond, shape, steps=18):
        """EDM Algorithm-2 二阶 Heun 确定性采样。ensemble 多样性来自不同的初始噪声 x0。"""
        dev = cond.device
        i = torch.arange(steps, dtype=torch.float64, device=dev)
        a, b = self.smax ** (1 / self.rho), self.smin ** (1 / self.rho)
        sig = ((a + i / (steps - 1) * (b - a)) ** self.rho)
        sig = torch.cat([sig, torch.zeros(1, dtype=torch.float64, device=dev)]).float()  # σ_N=0
        x = torch.randn(shape, device=dev) * sig[0]
        for j in range(steps):
            s = sig[j]
            d = (x - self._D(net, x, s, cond)) / s
            x_next = x + (sig[j + 1] - s) * d
            if sig[j + 1] > 0:                          # 二阶修正(最后一步 σ=0 跳过)
                d2 = (x_next - self._D(net, x_next, sig[j + 1], cond)) / sig[j + 1]
                x_next = x + (sig[j + 1] - s) * 0.5 * (d + d2)
            x = x_next
        return x


# ===========================================================================
# 4. κ 估计: R²(ERA5, coarsen(Daymet)) 随尺度变化 (doc 里的"实验一")
# ===========================================================================
def estimate_kappa(args, device=None):
    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    data = TD.DownscaleData(args.era5_dir, args.daymet_dir, [args.test_year],
                            args.in_vars, args.out_vars, stats, use_clim=getattr(args, "use_clim", False))
    y = args.test_year
    days = list(range(0, data.ndays[y], max(1, data.ndays[y] // 12)))
    shared = [v for v in args.out_vars if v in args.in_vars]
    factors = [f for f in [1, 2, 3, 6, 12] if data.H % f == 0 and data.W % f == 0]
    print(f"\n[κ] 共享变量={shared}  天数={len(days)}  HR={data.H}x{data.W}  ERA5={data.Hl}x{data.Wl}")
    print(f"{'scale(km≈)':>12} {'factor_to_HR':>13} " + " ".join(f"{v[:10]:>12}" for v in shared))
    out = {}
    for f in factors:
        Hc, Wc = data.H // f, data.W // f
        accs = {v: dict(sp=0., st=0., spp=0., stt=0., spt=0., n=0) for v in shared}
        for t in days:
            mask = data.mask[y]
            mk = mask.reshape(Hc, f, Wc, f).mean((1, 3)) > 0.5 if f > 1 else mask
            for v in shared:
                hr = data._hr(y, v, t).astype(np.float32)
                hc = hr.reshape(Hc, f, Wc, f).mean((1, 3)) if f > 1 else hr   # coarsen Daymet
                lr = _resize_np(data.lr[y][v][t].astype(np.float32), Hc, Wc)   # ERA5 LR -> κ 网格
                p, q = lr[mk], hc[mk]
                a = accs[v]
                a["sp"] += p.sum(); a["st"] += q.sum(); a["spp"] += (p * p).sum()
                a["stt"] += (q * q).sum(); a["spt"] += (p * q).sum(); a["n"] += p.size
        row = {}
        for v in shared:
            a = accs[v]; n = max(a["n"], 1)
            cov = a["spt"] / n - (a["sp"] / n) * (a["st"] / n)
            vp = a["spp"] / n - (a["sp"] / n) ** 2; vq = a["stt"] / n - (a["st"] / n) ** 2
            row[v] = (cov / math.sqrt(vp * vq)) ** 2 if vp > 0 and vq > 0 else float("nan")
        out[f] = row
        km = 4.6 * f
        print(f"{km:>10.1f}   {f:>13} " + " ".join(f"{row[v]:>12.3f}" for v in shared))
    print("[κ] 取 R² 跌破阈值(如 0.85)的最细尺度作为 κ。precip 通常在更粗尺度才达标。\n")
    return out


# ===========================================================================
# 5. 训练: 阶段一(corrector) / 阶段二(generator)
# ===========================================================================
def _all_reduce_scalar(tot, nb, device, is_dist):
    if is_dist:
        lt = torch.tensor([tot, nb], device=device); dist.all_reduce(lt); return lt[0].item(), lt[1].item()
    return tot, nb


@contextlib.contextmanager
def fixed_rng(seed):
    """临时把 RNG 固定住, 退出时还原。

    EDM 的 val loss 每次都要现采 σ 和噪声 —— 不固定的话, 相邻 epoch 的 val 差异里混着
    采样噪声, 曲线没法读。固定种子后每个 epoch 见到的是同一批 (σ, 噪声), val 可比。
    """
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _corrector_loss(net, cond, tgt, m, args, k, Cout):
    """阶段一损失: 粗尺度 κ 上的 heteroscedastic 高斯 NLL (+ 可选 CoBi)。train/val 共用。"""
    cond_c, yc, mc = coarsen(cond, k), coarsen(tgt, k), coarsen(m, k)
    mu, logvar = split_corrector(net(cond_c), Cout)
    inv = torch.exp(-logvar)
    nll = 0.5 * (inv * (mu - yc) ** 2 + logvar) * mc
    loss = nll.sum() / (mc.sum() * Cout + 1e-6)
    if args.cobi_weight > 0:
        loss = loss + args.cobi_weight * cobi_loss(mu, yc, mc, args.cobi_win, args.cobi_spatial)
    return loss


def train_corrector(corrector, dl, va_dl, args, device, is_dist, local, is_main, k, Cout, ckpt):
    ddp = DDP(corrector, device_ids=[local], static_graph=True) if is_dist else corrector
    opt = torch.optim.AdamW(corrector.parameters(), lr=args.lr)
    best = float("inf")
    for ep in range(args.corrector_epochs):
        ddp.train(); t0 = time.time(); tot = 0.0; nb = 0
        for cond, tgt, m in dl:
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            loss = _corrector_loss(ddp, cond, tgt, m, args, k, Cout)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        tot, nb = _all_reduce_scalar(tot, nb, device, is_dist)

        # ---- val: 与训练同一个损失, 在 val_years 上算 ----
        vmsg = ""
        if va_dl is not None:
            corrector.eval(); vt = 0.0; vn = 0
            with torch.no_grad():
                for cond, tgt, m in va_dl:
                    cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                    vt += float(_corrector_loss(corrector, cond, tgt, m, args, k, Cout).item()); vn += 1
            vt, vn = _all_reduce_scalar(vt, vn, device, is_dist)
            vloss = vt / max(vn, 1)
            improved = vloss < best - 1e-4
            if improved:
                best = vloss
                if is_main:
                    torch.save({"model": corrector.state_dict(), "args": vars(args),
                                "val_best": best, "epoch": ep}, ckpt)
            vmsg = f" val={vloss:.4f} best={best:.4f}{' *' if improved else ''}"

        if is_main:
            print(f"  [corrector] epoch {ep+1}/{args.corrector_epochs} "
                  f"loss={tot/max(nb,1):.4f}{vmsg} {time.time()-t0:.0f}s", flush=True)

    if is_main and va_dl is None:                       # 无 val -> 只能存最后一轮
        torch.save({"model": corrector.state_dict(), "args": vars(args)}, ckpt)
    if is_dist:
        dist.barrier()                                  # 等 rank0 写完盘
    # ★所有 rank 都载回 best —— 阶段二要用冻结的 corrector 造条件, 权重必须全 rank 一致,
    #   只让 rank0 载会让 8 个 rank 拿着不同的 μ/σ 训练同一个生成器。
    if os.path.exists(ckpt):
        corrector.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if is_dist:
        dist.barrier()
    return best


def _generator_loss(net, corrector, edm, cond, tgt, m, args, k, Cout, use_sigma):
    """阶段二损失: EDM 去噪损失 + σ 加权一致性项。train/val 共用。返回 (total, cons)。"""
    with torch.no_grad():                                           # 冻结的 corrector 给条件
        mu, logvar = split_corrector(corrector(coarsen(cond, k)), Cout)
    sig_c = torch.exp(0.5 * logvar)
    hw = tgt.shape[-2:]
    cg = [cond, upsample(mu, hw)] + ([upsample(sig_c, hw)] if use_sigma else [])
    cond_g = torch.cat(cg, 1)
    loss, D = edm.loss(net, tgt, cond_g, m)
    cons = torch.zeros((), device=tgt.device)
    if args.cons_weight > 0:                                        # σ 加权一致性: coarsen(D)≈μ
        Dc, mc = coarsen(D, k), coarsen(m, k)
        cons = ((Dc - mu) ** 2 * torch.exp(-logvar) * mc).sum() / (mc.sum() * Cout + 1e-6)
        loss = loss + args.cons_weight * cons
    return loss, cons


def train_generator(gen, corrector, edm, dl, va_dl, args, device, is_dist, local, is_main,
                    k, Cout, use_sigma, ckpt):
    ddp = DDP(gen, device_ids=[local], static_graph=True) if is_dist else gen
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr)
    best = float("inf")
    for ep in range(args.epochs):
        ddp.train(); t0 = time.time(); tot = 0.0; cot = 0.0; nb = 0
        for cond, tgt, m in dl:
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            loss, cons = _generator_loss(ddp, corrector, edm, cond, tgt, m, args, k, Cout, use_sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); cot += float(cons.item()); nb += 1
        tot, nb = _all_reduce_scalar(tot, nb, device, is_dist)

        # ---- val: 同一个 EDM 损失, 但★固定 RNG★ —— 否则 σ/噪声每轮重采, val 曲线全是采样噪声
        vmsg = ""
        if va_dl is not None:
            gen.eval(); vt = 0.0; vn = 0
            with torch.no_grad(), fixed_rng(20260713 + ep * 0):     # 每个 epoch 用同一批 (σ, 噪声)
                for cond, tgt, m in va_dl:
                    cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                    vl, _ = _generator_loss(gen, corrector, edm, cond, tgt, m, args, k, Cout, use_sigma)
                    vt += float(vl.item()); vn += 1
            vt, vn = _all_reduce_scalar(vt, vn, device, is_dist)
            vloss = vt / max(vn, 1)
            improved = vloss < best - 1e-4
            if improved:
                best = vloss
                if is_main:
                    torch.save({"model": gen.state_dict(), "args": vars(args),
                                "val_best": best, "epoch": ep}, ckpt)
            vmsg = f" val={vloss:.4f} best={best:.4f}{' *' if improved else ''}"

        if is_main:
            print(f"  [generator] epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
                  f"cons={cot/max(nb,1):.4f}{vmsg} {time.time()-t0:.0f}s", flush=True)

    if is_main and va_dl is None:
        torch.save({"model": gen.state_dict(), "args": vars(args)}, ckpt)
    if is_dist:
        dist.barrier()
    if os.path.exists(ckpt):                                        # 载回 best 供评测
        gen.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if is_dist:
        dist.barrier()
    return best


# ===========================================================================
# 6. 评测: 两阶段链式 + 集合 (复用 train_downscale 的指标)
# ===========================================================================
def evaluate_scd(corrector, gen, edm, stats, args, device, k, Cout, use_sigma):
    from era5_daymet.evaluation import eval_common as EC
    test = TD.DownscaleData(args.era5_dir, args.daymet_dir, [args.test_year],
                            args.in_vars, args.out_vars, stats, use_clim=getattr(args, "use_clim", False))
    corrector.eval(); gen.eval(); y = args.test_year
    days = list(range(0, test.ndays[y], args.eval_stride))
    dstd = stats.d_std[:, None, None]; dmean = stats.d_mean[:, None, None]
    ev = EC.MultiMethodEval(["scd"], args.out_vars, test.H, test.W, test.mask[y],
                            precip_scale=stats.precip_scale, precip_log=stats.precip_log)
    t0 = time.time()
    with torch.no_grad():
        for di, t in enumerate(days):
            cond, _, m, hr = test.full(y, t)
            cb = torch.from_numpy(cond[None]).float().to(device)
            H, W = cond.shape[1:]
            mu, logvar = split_corrector(corrector(coarsen(cb, k)), Cout)
            sig_c = torch.exp(0.5 * logvar)
            cg = [cb, upsample(mu, (H, W))] + ([upsample(sig_c, (H, W))] if use_sigma else [])
            cond_g = torch.cat(cg, 1)
            mem = [edm.sample(gen, cond_g, (1, Cout, H, W), steps=args.edm_steps)[0].cpu().numpy()
                   for _ in range(args.ensemble)]
            members = np.stack(mem, 0) * dstd[None] + dmean[None]       # 反归一化 -> 物理
            if PRECIP in args.out_vars:
                pi = args.out_vars.index(PRECIP)
                members[:, pi] = (TD.precip_inv(members[:, pi], stats.precip_scale)
                                  if stats.precip_log else np.maximum(members[:, pi], 0.0))
            ev.add_day(hr, m, {"scd": members})
            if (di + 1) % 25 == 0:
                print(f"  eval {di+1}/{len(days)} days ({time.time()-t0:.0f}s)", flush=True)
    ev.finalize(args.out, args.test_year, eval_stride=args.eval_stride, tag="scd")


# ===========================================================================
# 7. 编排: DDP + 两阶段
# ===========================================================================
def run(args):
    rank, world, local, device, is_dist = TD.setup_ddp()
    is_main = (rank == 0)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"device={device} stage={args.stage} κ_factor={args.kappa_factor} "
              f"in={len(args.in_vars)} out={len(args.out_vars)} patch={args.patch} world={world}", flush=True)

    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cout = len(args.out_vars)
    Cin = TD.cond_channels(args.in_vars, args.out_vars, getattr(args, "use_clim", False))  # 默认20; --use-clim=23
    k = args.kappa_factor
    use_sigma = not args.no_sigma_cond

    corrector = TD.UNet(Cin, 2 * Cout, base=args.corrector_base, temb=0).to(device)   # 出 μ + logvar
    gen_in = Cout + Cin + Cout + (Cout if use_sigma else 0)  # 噪声 + 条件 + μ_up (+ σ_up)
    gen = TD.UNet(gen_in, Cout, base=args.base, temb=128).to(device)
    edm = EDM(sigma_data=args.sigma_data, sigma_min=args.sigma_min,
              sigma_max=args.sigma_max, rho=args.rho)

    corr_ckpt = args.corrector_ckpt or os.path.join(args.out, "corrector.pt")
    gen_ckpt = args.generator_ckpt or os.path.join(args.out, "generator.pt")

    # 数据(两阶段共用)
    need_train = not args.eval_only
    dl = va_dl = None
    if need_train:
        data = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.train_years,
                                args.in_vars, args.out_vars, stats, use_clim=args.use_clim)
        ds = TD.PatchDS(data, args.patch, args.steps_per_epoch * args.batch, seed=1234 + rank)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                         drop_last=True, pin_memory=True, worker_init_fn=TD.ds_worker_init)
        # ★val: 原脚本完全没有验证集 —— 两个阶段都盲跑固定轮数、只存最后一轮。
        #   而本任务已被证实在 ~ep10 后过拟合(见 docs/archive/results/2026-07-15-results.md),
        #   盲跑 40 轮等于故意存一个退化的模型。
        if args.val_steps > 0:
            va_data = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.val_years,
                                       args.in_vars, args.out_vars, stats, use_clim=args.use_clim)
            # 逐 epoch 固定 + 按 rank 分片(同 train_downscale 口径): 各卡评不同 patch, 墙钟不变
            va_ds = TD.PatchDS(va_data, args.patch, args.val_steps * args.batch, seed=987, deterministic=True,
                               index_offset=rank * args.val_steps * args.batch)
            va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch,
                                                num_workers=max(1, args.workers // 2), drop_last=True,
                                                worker_init_fn=TD.ds_worker_init)

    # ---- 阶段一: corrector ----
    if args.stage in ("corrector", "both") and need_train:
        if is_main:
            print(f"==> 阶段一: 训练偏差算子 f_θ (μ + σ)  val={'on' if va_dl else 'OFF'}", flush=True)
        b = train_corrector(corrector, dl, va_dl, args, device, is_dist, local, is_main, k, Cout, corr_ckpt)
        if is_main:
            print(f"  -> {corr_ckpt}  (best val={b:.4f})" if va_dl else f"  -> {corr_ckpt}", flush=True)
        if is_dist: dist.barrier()

    # 阶段二/评测前: 准备好(冻结的)corrector
    if args.stage in ("generator", "both") or args.eval_only:
        if args.stage == "generator" or args.eval_only:        # 单独训练/评测 -> 从盘加载
            corrector.load_state_dict(torch.load(corr_ckpt, map_location=device)["model"])
            if is_main: print(f"  loaded corrector <- {corr_ckpt}", flush=True)
        corrector.eval()
        for p in corrector.parameters():
            p.requires_grad_(False)

    # ---- 阶段二: generator ----
    if args.stage in ("generator", "both") and need_train:
        if is_main:
            print(f"==> 阶段二: 训练 EDM 条件生成器 G_θ  val={'on' if va_dl else 'OFF'}", flush=True)
        b = train_generator(gen, corrector, edm, dl, va_dl, args, device, is_dist, local, is_main,
                            k, Cout, use_sigma, gen_ckpt)
        if is_main:
            print(f"  -> {gen_ckpt}  (best val={b:.4f})" if va_dl else f"  -> {gen_ckpt}", flush=True)
        if is_dist: dist.barrier()

    if args.eval_only:
        gen.load_state_dict(torch.load(gen_ckpt, map_location=device)["model"])
        if is_main: print(f"  loaded generator <- {gen_ckpt}", flush=True)

    # ---- 评测(仅主进程) ----
    # --skip-eval: 训练与评测解耦。集合评测(16 成员 x 35 次整帧前向 x N 天)在单卡上很贵,
    # 与 generator 训练挤在同一个作业里会顶爆 debug 的 2 小时墙钟。拆开跑, 用 --eval-only 收尾。
    if is_main and not args.skip_eval and (args.stage in ("generator", "both") or args.eval_only):
        evaluate_scd(corrector, gen, edm, stats, args, device, k, Cout, use_sigma)
    elif is_main and args.skip_eval:
        print("  [skip-eval] 跳过评测 (用 --eval-only 单独跑)", flush=True)
    if is_dist:
        dist.barrier(); dist.destroy_process_group()


# ===========================================================================
# 8. CLI + 合成自测
# ===========================================================================
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", choices=["corrector", "generator", "both"], default="both")
    p.add_argument("--era5-dir", default=M.ERA5_DIR); p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default="stats_train")
    p.add_argument("--in-vars", nargs="+", default=TD.DEFAULT_IN)
    p.add_argument("--out-vars", nargs="+", default=TARGETS)
    p.add_argument("--use-clim", action="store_true",
                   help="保留 3 个逐日气候态条件通道 -> 23 通道(旧口径); 默认关=20 通道(指南口径)")
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"],
                   help="默认 1980-2017 全训练集; 内存紧张可自行缩小")
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"],
                   help="验证年份(默认 2018-2019), 用于监控过拟合 + 存 best ckpt")
    p.add_argument("--val-steps", type=int, default=50,
                   help="每 epoch 的 val 步数; 0=关闭验证(退回原来的盲跑固定轮数)")
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])
    p.add_argument("--out", default=str(PROJECT_ROOT / "runs/scd"))
    p.add_argument("--corrector-ckpt", default=""); p.add_argument("--generator-ckpt", default="")
    # 模型/尺度
    p.add_argument("--kappa-factor", type=int, default=FACTOR,
                   help="κ 相对 HR 的粗化倍数; 默认 6 = ERA5 网格分辨率")
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--base", type=int, default=64); p.add_argument("--corrector-base", type=int, default=48)
    p.add_argument("--no-sigma-cond", action="store_true", help="不把 σ 图喂进生成器条件")
    # 损失权重
    p.add_argument("--cobi-weight", type=float, default=0.0, help=">0 启用 CoBi 错位鲁棒项")
    p.add_argument("--cobi-win", type=int, default=5); p.add_argument("--cobi-spatial", type=float, default=0.1)
    p.add_argument("--cons-weight", type=float, default=0.1, help="σ 加权一致性项权重")
    # EDM
    p.add_argument("--sigma-data", type=float, default=1.0); p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0); p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--edm-steps", type=int, default=18, help="Heun 采样步数")
    # 优化
    p.add_argument("--batch", type=int, default=16); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=40, help="generator 训练轮数")
    p.add_argument("--corrector-epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=500); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument("--eval-stride", type=int, default=1,
                   help="1=完整 test 年(默认, 汇报必须用此)。>1=子采样, 结果会被 eval_common 标 __SUBSAMPLED")
    # 模式
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="训练完不评测(集合评测很贵, 拆成单独作业用 --eval-only 跑)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--estimate-kappa", action="store_true")
    args = p.parse_args()

    if torch is None:
        sys.exit("需要 torch(Frontier GPU)。本机仅可做数据/语法检查。")

    if args.smoke:
        import tempfile
        root = tempfile.mkdtemp()
        args.era5_dir, args.daymet_dir, args.stats_dir = TD.make_synth(root)
        args.in_vars = TARGETS; args.out_vars = TARGETS
        args.train_years = [2018, 2019]; args.test_year = 2020
        args.patch = 48; args.base = 16; args.corrector_base = 16; args.batch = 4
        args.epochs = 1; args.corrector_epochs = 1; args.steps_per_epoch = 8; args.workers = 0
        args.ensemble = 4; args.eval_stride = 10; args.edm_steps = 6
        args.cobi_weight = 1.0; args.cons_weight = 0.1; args.kappa_factor = FACTOR
        args.out = root + "/out"
        print("[smoke] 合成数据+stats 自测 (EDM 两阶段) ...", flush=True)

    if args.estimate_kappa:
        estimate_kappa(args)
        return
    run(args)


# ---------------------------------------------------------------------------
# DDP / SLURM (Frontier) 示例:
#   srun -N4 --ntasks-per-node=8 --gpus-per-node=8 \
#        python train_scd.py --stage both --out runs/scd \
#        --era5-dir $ERA5 --daymet-dir $DAYMET --stats-dir $STATS \
#        --epochs 80 --corrector-epochs 30 --batch 8 --patch 192 \
#        --cobi-weight 0.5 --cons-weight 0.1 --ensemble 32
# 推理(单卡即可):  python train_scd.py --eval-only --out runs/scd --ensemble 32
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
