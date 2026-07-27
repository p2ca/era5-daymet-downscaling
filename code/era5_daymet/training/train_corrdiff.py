#!/usr/bin/env python
# Packaged implementation; code/train_corrdiff.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
train_corrdiff.py — CorrDiff 复现 (Mardani et al., Comms Earth Env 2025, 6:124)
ERA5(120x240) -> Daymet(720x1440), 6x, CONUS
============================================================================
核心是★两阶段残差分解★(论文式 1):

      x  =  E[x|y]  +  (x - E[x|y])
             └ 回归        └ 生成(扩散)

  阶段一 回归: UNet 学条件均值 μ = E[x|y]，masked-MSE，确定性，只出一张图
  阶段二 扩散: EDM 学★残差★ r = x - μ 的分布，条件 = [cond, μ]
  采样    : x = μ + EDM_sample(r | cond, μ)，同一输入采 N 次 -> ensemble

为什么是残差 (论文式 2，全方差定律): var(r) ≤ var(x)。
残差的方差比目标本身小 -> 扩散要建模的分布更容易。论文明说直接学 p(x|y) 会
"收敛慢 + 样本质量差"(大分布偏移下前向过程需要极大噪声)。

★复现的判据不是 RMSE。★ 论文自己的表 1/S3 里 UNet 的 MAE 就优于 CorrDiff，
作者称之为 "expected"(扩散优化 KL，不优化 MAE)。判胜负看:
    CRPS / 功率谱 / rank histogram / SSIM —— 这些 eval_common 全都有。

★我们与论文的规模差距(必须如实报告)★
    论文: 80M 参数, 448x448 整帧, 全局 batch 512, 5.5 万优化步, 21,504 H100-小时
    我们: 1.6M 参数, 192x192 patch, 全局 batch 128-512, 受 debug 2h 墙钟限制
  -> 复现的是★方法★, 不是论文的★数字★。

用法:
  python train_corrdiff.py --smoke                              # 秒级自测(合成数据)
  # 阶段一: 直接复用已训好的 UNet 当回归器(省一次训练)
  python train_corrdiff.py --stage diffusion \
      --regressor-ckpt runs/exp/20260711-unet-b64/ckpt.pt \
      --stats-dir runs/stats/train_dayofyear --out runs/exp/<id>
  # 或从零训回归器
  python train_corrdiff.py --stage both --out runs/exp/<id>
  # 评测(出 ensemble)
  python train_corrdiff.py --eval-only --out runs/exp/<id> --ensemble 16
============================================================================
"""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.paths import PROJECT_ROOT

# 输出目录默认值(相对仓库根锚定, 从任何 cwd 运行都写入正式 runs/); 下方 sentinel 比较也用它, 保持同步
DEFAULT_OUT = str(PROJECT_ROOT / "runs/exp/corrdiff")

torch = TD.torch
if torch is not None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

PRECIP = TD.PRECIP
TARGETS = TD.TARGETS


# ===========================================================================
# 1. EDM (Karras et al. 2022) —— preconditioning + 噪声调度 + 二阶采样
# ===========================================================================
class EDM:
    """Elucidated Diffusion Model。

    ★论文用的不是 EDM 默认值★:
        论文: sigma_max=800, P_mean=0.0      理由: "完全摧毁雷达反射率的巨大数据强度"
        默认: sigma_max=80,  P_mean=-1.2
    我们的目标是★归一化距平★(单位方差), 而且残差方差比目标还小 —— 未必需要那么猛的噪声。
    故默认走 EDM 标准值, 把论文值留作 --paper-noise 的对照实验。
    """

    def __init__(self, sigma_data=1.0, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                 P_mean=-1.2, P_std=1.2):
        self.sd, self.smin, self.smax, self.rho = sigma_data, sigma_min, sigma_max, rho
        self.P_mean, self.P_std = P_mean, P_std

    def _D(self, net, x, sigma, cond):
        """去噪器 D(x;σ,cond) = c_skip·x + c_out·F(c_in·x; c_noise, cond)。"""
        if not torch.is_tensor(sigma) or sigma.ndim == 0:
            sigma = torch.full((x.size(0),), float(sigma), device=x.device)
        s = sigma.view(-1, 1, 1, 1)
        c_skip = self.sd ** 2 / (s ** 2 + self.sd ** 2)
        c_out = s * self.sd / (s ** 2 + self.sd ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + self.sd ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip * x + c_out * net(torch.cat([c_in * x, cond], 1), c_noise)

    def loss(self, net, r, cond, mask):
        """EDM 去噪损失: ln σ ~ N(P_mean, P_std²), 加噪, λ(σ) 加权 ‖D-r‖², 只在陆地上。"""
        sigma = (torch.randn(r.size(0), device=r.device) * self.P_std + self.P_mean).exp()
        n = torch.randn_like(r) * sigma.view(-1, 1, 1, 1)
        D = self._D(net, r + n, sigma, cond)
        w = ((sigma ** 2 + self.sd ** 2) / (sigma * self.sd) ** 2).view(-1, 1, 1, 1)
        se = w * (D - r) ** 2 * mask
        return se.sum() / (mask.sum() * r.size(1) + 1e-6)

    @torch.no_grad()
    def sample(self, net, cond, shape, steps=18, stochastic=True,
               S_churn=40.0, S_min=0.05, S_max=50.0, S_noise=1.003, mask=None):
        """EDM Algorithm 2 二阶采样。

        论文用★随机★采样器(18 步); stochastic=False 退化为确定性 Heun。
        集合的多样性来自不同的初始噪声 (以及随机采样器注入的噪声)。

        ★mask (可选, (H,W) 或 (1,1,H,W), 陆地=1 海洋=0)★:
        我们的域 52% 是海洋, 训练时 loss 只监督陆地 -> 海洋区从未被约束, 整帧采样时它会
        跑飞(诊断: 海洋残差 mean +1.3), 并通过卷积感受野每一步渗进沿海陆地(陆地残差 std
        从 patch 的 0.087 炸到整帧的 0.237)。修法: 每步去噪后把海洋区的残差★钉回 0★
        (残差空间里 0 = "μ 已经说完了, 无附加"), 让海洋始终是干净的已知值, 从根上掐断
        污染源。这是扩散做 inpainting 的标准手法(每步都锁, 不是最后裁一次)。
        """
        dev = cond.device
        if mask is not None:
            m = mask if mask.ndim == 4 else mask.view(1, 1, *mask.shape[-2:])
            m = (m > 0.5).float().to(dev)                       # 陆地=1
        i = torch.arange(steps, dtype=torch.float64, device=dev)
        a, b = self.smax ** (1 / self.rho), self.smin ** (1 / self.rho)
        sig = (a + i / max(steps - 1, 1) * (b - a)) ** self.rho
        sig = torch.cat([sig, torch.zeros(1, dtype=torch.float64, device=dev)]).float()

        x = torch.randn(shape, device=dev) * sig[0]
        if mask is not None:
            x = x * m                                           # 海洋从一开始就锁成 0
        for j in range(steps):
            s_cur = sig[j]
            # --- 随机采样器: 先"回加"一点噪声(churn), 再去噪 ---
            if stochastic and S_min <= s_cur <= S_max:
                gamma = min(S_churn / steps, math.sqrt(2) - 1)
                s_hat = s_cur * (1 + gamma)
                x = x + (s_hat ** 2 - s_cur ** 2).clamp(min=0).sqrt() * S_noise * torch.randn_like(x)
            else:
                s_hat = s_cur
            if mask is not None:
                x = x * m                                       # churn 加的噪声也别留在海洋
            d = (x - self._D(net, x, s_hat, cond)) / s_hat
            x_next = x + (sig[j + 1] - s_hat) * d
            if sig[j + 1] > 0:                                  # 二阶修正
                d2 = (x_next - self._D(net, x_next, sig[j + 1], cond)) / sig[j + 1]
                x_next = x + (sig[j + 1] - s_hat) * 0.5 * (d + d2)
            x = x_next
            if mask is not None:
                x = x * m                                       # ★每步都把海洋钉回 0
        return x


# ===========================================================================
# 2. EMA —— 论文用 EMA(η=0.5); 对扩散的采样质量影响很大, 我们的 UNet 原本没有
# ===========================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for se, pe in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if se.dtype.is_floating_point:
                se.mul_(self.decay).add_(pe.detach(), alpha=1 - self.decay)
            else:
                se.copy_(pe)


# ===========================================================================
# 3. 条件构造: 扩散的条件 = [baseline cond] + [回归出的 μ]
#    论文: "we also add the mean obtained by the regression model in the first stage"
#    -> μ 不只是最后加回来, 还要★作为条件通道喂进扩散★
# ===========================================================================
def diff_cond(cond, mu):
    return torch.cat([cond, mu], 1)


def estimate_sigma_data(regressor, dl, device, k=8):
    """估计残差 r = x - μ 的标准差, 作为 EDM 的 sigma_data。

    ★这一步不能省★: EDM 的 preconditioning (c_skip/c_out/c_in) 全部依赖 sigma_data。
    残差的方差★比目标本身小★(论文式2 的全方差定律), 直接沿用 sigma_data=1.0
    (为单位方差的目标设的) 会让 preconditioning 系统性失配。
    """
    regressor.eval()
    vals = []
    with torch.no_grad():
        for i, (cond, tgt, m) in enumerate(dl):
            if i >= k:
                break
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            r = (tgt - regressor(cond)) * m
            n = m.sum() * tgt.shape[1]
            vals.append(float((r ** 2).sum() / n.clamp(min=1)))
    return float(np.sqrt(np.mean(vals))) if vals else 1.0


# ===========================================================================
# 4. 训练
# ===========================================================================
def train_regressor(model, dl, va_dl, args, device, ddp_info, ckpt):
    """阶段一: UNet 回归 μ = E[x|y]。就是我们已有的确定性 baseline。"""
    rank, world, local, is_dist = ddp_info
    is_main = (rank == 0)
    net = DDP(model, device_ids=[local], static_graph=True) if is_dist else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.reg_lr, weight_decay=args.weight_decay)
    best = float("inf")
    for ep in range(args.reg_epochs):
        net.train(); t0 = time.time()
        for cond, tgt, m in dl:
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            loss = TD.masked_mse(net(cond), tgt, m)
            opt.zero_grad(); loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        model.eval(); vt = vn = 0.0
        with torch.no_grad():
            for cond, tgt, m in va_dl:
                cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                vt += TD.masked_mse(model(cond), tgt, m).item(); vn += 1
        if is_dist:
            tt = torch.tensor([vt, vn], device=device); dist.all_reduce(tt); vt, vn = tt[0].item(), tt[1].item()
        vloss = vt / max(vn, 1)
        improved = vloss < best - 1e-4
        if improved:
            best = vloss
            if is_main:
                torch.save({"model": model.state_dict(), "args": vars(args), "val_best": best}, ckpt)
        if is_main:
            print(f"  [regression] ep {ep+1}/{args.reg_epochs} val={vloss:.4f} best={best:.4f}"
                  f"{' *' if improved else ''} {time.time()-t0:.0f}s", flush=True)
    if is_dist:
        dist.barrier()
    if os.path.exists(ckpt):                                    # ★所有 rank 都载 best
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if is_dist:
        dist.barrier()
    return best


def train_diffusion(gen, regressor, edm, dl, va_dl, args, device, ddp_info, ckpt):
    """阶段二: EDM 学残差 r = x - μ 的分布, 条件 = [cond, μ]。μ 冻结。

    ★支持检查点续训★: Frontier 对 1-91 节点的作业硬性限制 2 小时墙钟(按作业大小分级,
    92+ 节点才给 6 小时 —— 但那要付 276 倍机时, 对我们 1.6M 的小模型是荒谬的)。
    要凑够论文的 ~5.5 万优化步就必须分段跑, 所以 last.pt 里存的是★完整训练态★
    (模型 + 优化器 + EMA + epoch + best), 不是只存权重 —— 只存权重的话优化器动量和
    EMA 影子权重会丢, 续训等于每段都重新热身, 那就不是"接着训"而是"反复重启"。
    """
    rank, world, local, is_dist = ddp_info
    is_main = (rank == 0)
    net = DDP(gen, device_ids=[local], static_graph=True) if is_dist else gen
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))                  # 论文: beta2=0.99
    ema = EMA(gen, args.ema_decay) if args.ema_decay > 0 else None
    best = float("inf"); ep0 = 0

    last = os.path.join(os.path.dirname(ckpt), "last.pt")        # 续训用(每 epoch 覆盖)
    if args.resume and os.path.exists(last):
        st = torch.load(last, map_location=device)
        gen.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if ema and st.get("ema") is not None:
            ema.shadow.load_state_dict(st["ema"])
        ep0 = st["epoch"] + 1; best = st["best"]
        if is_main:
            print(f"  ★续训: 从 epoch {ep0} 接上 (best={best:.4f}, 优化器/EMA 状态已恢复)", flush=True)
        if is_dist:
            dist.barrier()

    for ep in range(ep0, args.epochs):
        net.train(); t0 = time.time(); tot = nb = 0.0
        for cond, tgt, m in dl:
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            with torch.no_grad():
                mu = regressor(cond)                            # 冻结的回归器
            r = tgt - mu                                        # ★残差 = 扩散的目标
            loss = edm.loss(net, r, diff_cond(cond, mu), m)
            opt.zero_grad(); loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
            opt.step()
            if ema:
                ema.update(gen)
            tot += float(loss.item()); nb += 1
        # ---- val: 同一个 EDM 损失, ★固定 RNG★ 否则曲线全是采样噪声 ----
        gen.eval(); vt = vn = 0.0
        st_cpu = torch.get_rng_state()
        st_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(1234)
        with torch.no_grad():
            for cond, tgt, m in va_dl:
                cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                mu = regressor(cond)
                vt += float(edm.loss(gen, tgt - mu, diff_cond(cond, mu), m).item()); vn += 1
        torch.set_rng_state(st_cpu)
        if st_cuda is not None:
            torch.cuda.set_rng_state_all(st_cuda)
        if is_dist:
            tt = torch.tensor([vt, vn, tot, nb], device=device); dist.all_reduce(tt)
            vt, vn, tot, nb = tt[0].item(), tt[1].item(), tt[2].item(), tt[3].item()
        vloss = vt / max(vn, 1)
        improved = vloss < best - 1e-4
        if improved:
            best = vloss
            if is_main:
                sd = (ema.shadow if ema else gen).state_dict()  # ★存 EMA 权重(论文用 EMA 采样)
                torch.save({"model": sd, "raw": gen.state_dict(), "args": vars(args),
                            "val_best": best, "sigma_data": edm.sd, "epoch": ep}, ckpt)
        if is_main:
            # ★每个 epoch 都存完整训练态 -> 作业被 2h 墙钟砍掉时可以无损接上
            torch.save({"model": gen.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.shadow.state_dict() if ema else None,
                        "epoch": ep, "best": best, "sigma_data": edm.sd}, last)
            print(f"  [diffusion] ep {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
                  f"val={vloss:.4f} best={best:.4f}{' *' if improved else ''} {time.time()-t0:.0f}s", flush=True)
    if is_dist:
        dist.barrier()
    if os.path.exists(ckpt):
        gen.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if is_dist:
        dist.barrier()
    return best


# ===========================================================================
# 5. 分块集合采样 —— ★整帧采样会发散★(诊断结论)
#    模型只在 192x192 patch 上训过, 在 720x1440 整帧上采样时扩散轨迹发散
#    (陆地残差 std 从 patch 的 0.087 炸到整帧的 0.234)。修法: 只在训练见过的 tile
#    尺寸上采样, 重叠平均拼回整帧。与 det_predict 的分块框架同构, 但每块采 N 个成员、
#    且跳过纯海洋块(残差=0)。
# ===========================================================================
@torch.no_grad()
def _feather_window_t(tile, ov, dev):
    """羽化窗口 (tile,tile) torch: 块边缘 ov 像素线性降权, 内部为 1, 最小值 1/(ov+1)>0。
    与确定性分块推理 (train_downscale.det_predict) 同一套融合, 消除分块采样的接缝格子。"""
    r = (torch.arange(ov, device=dev, dtype=torch.float32) + 1) / (ov + 1)
    w1 = torch.ones(tile, device=dev)
    w1[:ov] = r; w1[-ov:] = r.flip(0)
    return torch.outer(w1, w1)                                 # (tile,tile)


def tiled_sample(gen, edm, cond, mu, landmask, Cout, tile, ensemble, steps, stochastic):
    """整帧条件 -> (ensemble, Cout, H, W) 残差。分块采样 + ★羽化加权融合★。

    ★两层去缝★:
    1) 每个成员用独立的分块网格随机相位 -> 打散固定接缝、增加集合多样性。
    2) ★羽化加权融合★(而非均匀平均): 每块套一个边缘降权窗口, 重叠带平滑过渡, 消除
       "一格一格"。均匀平均会在覆盖数跳变处 + 块边缘(上下文最少、采样最不稳)留下硬接缝。
    """
    dev = cond.device
    _, _, H, W = cond.shape
    ov = tile // 4                                              # 25% 重叠
    step = tile - ov
    win = _feather_window_t(tile, ov, dev)                     # (tile,tile) 融合权重
    cg_full = diff_cond(cond, mu)                              # (1, Ccond, H, W)
    out = np.zeros((ensemble, Cout, H, W), np.float32)
    for e in range(ensemble):
        acc = torch.zeros((Cout, H, W), device=dev)
        wsum = torch.zeros((1, H, W), device=dev)
        # 随机相位: 网格起点在 [0, step) 内随机, 打散接缝
        oy = int(torch.randint(0, max(step, 1), (1,)).item()) if H > tile else 0
        ox = int(torch.randint(0, max(step, 1), (1,)).item()) if W > tile else 0
        ys = sorted(set(list(range(oy, max(1, H - tile + 1), step)) + [0, max(0, H - tile)]))
        xs = sorted(set(list(range(ox, max(1, W - tile + 1), step)) + [0, max(0, W - tile)]))
        for y0 in ys:
            for x0 in xs:
                lm = landmask[y0:y0 + tile, x0:x0 + tile]
                if lm.sum() < 1:                               # 纯海洋块 -> 残差=0, 跳过
                    continue
                cg = cg_full[:, :, y0:y0 + tile, x0:x0 + tile]
                r = edm.sample(gen, cg, (1, Cout, tile, tile), steps=steps, stochastic=stochastic)[0]
                acc[:, y0:y0 + tile, x0:x0 + tile] += r * win                # ★加权累加
                wsum[:, y0:y0 + tile, x0:x0 + tile] += win
        r_full = acc / wsum.clamp(min=1e-6)
        out[e] = r_full.cpu().numpy()
    return out


# ===========================================================================
# 6. 评测: x = μ + EDM(r), 出 ensemble; 复用 eval_common (CRPS/rank hist/谱/SSIM)
# ===========================================================================
@torch.no_grad()
def evaluate(regressor, gen, edm, stats, args, device):
    from era5_daymet.evaluation import eval_common as EC
    test = TD.DownscaleData(args.era5_dir, args.daymet_dir, [args.test_year],
                            args.in_vars, args.out_vars, stats, use_clim=getattr(args, "use_clim", False))
    regressor.eval(); gen.eval()
    y = args.test_year; Cout = len(args.out_vars)
    days = list(range(0, test.ndays[y], args.eval_stride))
    # ★分片: 把完整天列表切成不相交子段, 多 GCD 并行, 各自 dump_sums, 事后精确合并
    #   (采样很贵, 单卡全年 ~16 GPU-h; 分片才能塞进 debug 2h)。见 submit_fulltest_corrdiff.sh
    if getattr(args, "eval_day_i1", 0) > 0:
        days = days[args.eval_day_i0:args.eval_day_i1]
        print(f"  [shard] 本片评测天索引 [{args.eval_day_i0}:{args.eval_day_i1}] -> {len(days)} 天", flush=True)
    dstd = stats.d_std[:, None, None]; dmean = stats.d_mean[:, None, None]
    ev = EC.MultiMethodEval(["corrdiff"], args.out_vars, test.H, test.W, test.mask[y],
                            precip_scale=stats.precip_scale, precip_log=stats.precip_log)
    t0 = time.time(); clip_frac = []
    for di, t in enumerate(days):
        # Make stochastic ensemble sampling reproducible per calendar day and
        # independent of how the full year is split across GCD shards.
        if getattr(args, "eval_seed", -1) >= 0:
            torch.manual_seed(int(args.eval_seed) + int(t))
        cond, _, m, hr = test.full(y, t)
        cb = torch.from_numpy(cond[None]).float().to(device)
        H, W = cond.shape[1:]
        mu = regressor(cb)                                      # (1,Cout,H,W) 归一化空间
        lm = torch.from_numpy((m[0] if m.ndim == 3 else m) > 0.5).to(device)
        # ★分块采样(整帧会发散); 每成员独立随机相位打散接缝
        r_ens = tiled_sample(gen, edm, cb, mu, lm, Cout, args.eval_tile, args.ensemble,
                             args.edm_steps, stochastic=not args.deterministic_sampler)
        members = (mu.cpu().numpy() + r_ens) * dstd[None] + dmean[None]   # x = μ + r, 反归一化
        if PRECIP in args.out_vars:
            pi = args.out_vars.index(PRECIP)
            if stats.precip_log:
                # ★钳制 + 记录比例: expm1 是指数, 生成式的离群样本会把指标炸成 inf。
                #   clip_frac 训练良好时应≈0; 显著>0 = 模型在吐物理上不可能的降水 = 红旗。
                members[:, pi], cf = TD.precip_inv(members[:, pi], stats.precip_scale,
                                                   return_clipped=True)
                clip_frac.append(cf)
            else:
                members[:, pi] = np.maximum(members[:, pi], 0.0)
        ev.add_day(hr, m, {"corrdiff": members})
        if (di + 1) % 10 == 0:
            print(f"  eval {di+1}/{len(days)} 天 ({time.time()-t0:.0f}s)", flush=True)
    if clip_frac:
        cf = float(np.mean(clip_frac))
        flag = "✓ 正常" if cf < 1e-6 else ("⚠ 偏高" if cf < 1e-3 else "★红旗: 模型在吐物理上不可能的降水")
        print(f"\n  降水被钳制的像素比例 = {cf:.3e}   {flag}", flush=True)
    if getattr(args, "dump_sums", ""):
        ev.dump_sums(args.dump_sums)            # 分片模式: 落盘可加累计量, 不 finalize(合并脚本来 finalize)
    else:
        ev.finalize(args.out, args.test_year, eval_stride=args.eval_stride, tag="corrdiff",
                    n_total_days=test.ndays[y])


# ===========================================================================
# 6. 主流程
# ===========================================================================
def run(args):
    rank, world, local, device, is_dist = TD.setup_ddp()
    is_main = (rank == 0)
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cout = len(args.out_vars)
    Cin = TD.cond_channels(args.in_vars, args.out_vars, args.use_clim)   # 默认20通道; --use-clim=23

    # 回归器: 与 train_unet.py 完全同构 -> 可直接载入已训好的 ckpt
    regressor = TD.UNet(Cin, Cout, base=args.reg_base, temb=0).to(device)
    # 扩散去噪器: 输入 = [噪声(Cout) + cond(Cin) + μ(Cout)], 输出 = 残差(Cout)
    gen = TD.UNet(Cout + Cin + Cout, Cout, base=args.base, temb=128).to(device)

    reg_ckpt = os.path.join(args.out, "regressor.pt")
    gen_ckpt = os.path.join(args.out, "generator.pt")

    if is_main:
        nr = sum(p.numel() for p in regressor.parameters())
        ng = sum(p.numel() for p in gen.parameters())
        print(f"[CorrDiff] 回归器 base={args.reg_base} {nr/1e6:.1f}M | "
              f"扩散器 base={args.base} {ng/1e6:.1f}M | world={world} | "
              f"patch={args.patch} batch={args.batch} (全局 {args.batch*world})", flush=True)

    # ---- 数据 ----
    dl = va_dl = None
    if not args.eval_only:
        tr = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.train_years,
                              args.in_vars, args.out_vars, stats, use_clim=args.use_clim)
        va = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.val_years,
                              args.in_vars, args.out_vars, stats, use_clim=args.use_clim)
        dl = torch.utils.data.DataLoader(
            TD.PatchDS(tr, args.patch, args.steps_per_epoch * args.batch, seed=1234 + rank),
            batch_size=args.batch, num_workers=args.workers, drop_last=True, pin_memory=True,
            worker_init_fn=TD.ds_worker_init)
        # val: 逐 epoch 固定(不含采样噪声), 但按 rank 分片 -> 各卡评不同 patch, 墙钟不变、覆盖 x world
        # (与 train_downscale.fit_deterministic 同一口径; 修正前所有 rank 评的是同一批 patch)
        va_dl = torch.utils.data.DataLoader(
            TD.PatchDS(va, args.patch, args.val_steps * args.batch, seed=987, deterministic=True,
                       index_offset=rank * args.val_steps * args.batch),
            batch_size=args.batch, num_workers=max(1, args.workers // 2), drop_last=True,
            worker_init_fn=TD.ds_worker_init)

    # ---- 阶段一: 回归器 (可复用已训好的 UNet, 省一次训练) ----
    if args.regressor_ckpt:
        sd = torch.load(args.regressor_ckpt, map_location=device)["model"]
        regressor.load_state_dict(sd)
        if is_main:
            print(f"  载入已训好的回归器 <- {args.regressor_ckpt} (跳过阶段一)", flush=True)
    elif args.stage in ("regression", "both"):
        if is_main:
            print("==> 阶段一: 训练回归器 μ = E[x|y]", flush=True)
        b = train_regressor(regressor, dl, va_dl, args, device, (rank, world, local, is_dist), reg_ckpt)
        if is_main:
            print(f"  -> {reg_ckpt} (best val={b:.4f})", flush=True)
    elif os.path.exists(reg_ckpt):
        regressor.load_state_dict(torch.load(reg_ckpt, map_location=device)["model"])

    regressor.eval()
    for p in regressor.parameters():
        p.requires_grad_(False)                                 # ★μ 冻结

    # ---- sigma_data: 从残差估, 不能沿用为"单位方差目标"设的 1.0 ----
    sd_val = args.sigma_data
    if sd_val <= 0 and not args.eval_only:
        sd_val = estimate_sigma_data(regressor, dl, device)
        if is_dist:
            t = torch.tensor([sd_val], device=device); dist.all_reduce(t, op=dist.ReduceOp.AVG)
            sd_val = float(t.item())
        if is_main:
            print(f"  估计 sigma_data(残差 std) = {sd_val:.4f}  "
                  f"(目标本身是单位方差 -> 残差方差更小, 印证论文式2)", flush=True)
    elif sd_val <= 0:
        sd_val = float(torch.load(gen_ckpt, map_location="cpu").get("sigma_data", 1.0))

    edm = EDM(sigma_data=sd_val, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
              rho=args.rho, P_mean=args.p_mean, P_std=args.p_std)

    # ---- 阶段二: 扩散 ----
    if args.stage in ("diffusion", "both") and not args.eval_only:
        if is_main:
            print(f"==> 阶段二: EDM 学残差 r = x - μ  "
                  f"(sigma_max={args.sigma_max} P_mean={args.p_mean} sigma_data={sd_val:.3f})", flush=True)
        b = train_diffusion(gen, regressor, edm, dl, va_dl, args, device,
                            (rank, world, local, is_dist), gen_ckpt)
        if is_main:
            print(f"  -> {gen_ckpt} (best val={b:.4f})", flush=True)

    if args.eval_only:
        ck = torch.load(gen_ckpt, map_location=device)
        gen.load_state_dict(ck["model"])
        if is_main:
            print(f"  载入生成器 <- {gen_ckpt}", flush=True)

    # ---- 评测(仅 rank0) ----
    if is_main and not args.skip_eval:
        print(f"==> 评测: x = μ + EDM(r), {args.ensemble} 成员, {args.edm_steps} 步", flush=True)
        evaluate(regressor, gen, edm, stats, args, device)
    if is_dist:
        dist.barrier(); dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="CorrDiff 复现: 两阶段残差扩散",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", choices=["regression", "diffusion", "both"], default="both")
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default="stats_train")
    p.add_argument("--in-vars", nargs="+", default=TD.DEFAULT_IN)
    p.add_argument("--out-vars", nargs="+", default=TARGETS)
    p.add_argument("--use-clim", action="store_true",
                   help="保留 3 个逐日气候态条件通道 -> 23 通道(旧口径); 默认关=20 通道(指南口径)")
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--regressor-ckpt", default="",
                   help="复用已训好的 UNet 当回归器(如 runs/exp/20260711-unet-b64/ckpt.pt), 跳过阶段一")

    # 模型
    p.add_argument("--reg-base", type=int, default=64, help="回归器 UNet base(与 train_unet 对齐)")
    p.add_argument("--base", type=int, default=64,
                   help="扩散器 UNet base。论文 Note3: 残差是局部/小尺度的 -> 扩散网可以更小")
    p.add_argument("--patch", type=int, default=192, help="必须被 FACTOR=6 整除")
    p.add_argument("--batch", type=int, default=16, help="每 rank; 全局 = batch x world(论文 512)")

    # 训练
    p.add_argument("--lr", type=float, default=2e-4, help="扩散 lr(论文 2e-4)")
    p.add_argument("--reg-lr", type=float, default=1.5e-4, help="回归 lr(与我们的 UNet baseline 对齐)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA(论文用 EMA; 0=关闭)")
    p.add_argument("--epochs", type=int, default=40, help="扩散轮数")
    p.add_argument("--reg-epochs", type=int, default=12, help="回归轮数(本任务 ~ep10 后过拟合)")
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--val-steps", type=int, default=50)
    p.add_argument("--workers", type=int, default=7)

    # EDM —— ★默认走 EDM 标准值, 不是论文值
    # (论文的大噪声是为雷达重尾定的)
    p.add_argument("--sigma-data", type=float, default=-1.0,
                   help="<=0 则从残差自动估计(推荐)。EDM preconditioning 全依赖它")
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0, help="论文用 800(为雷达重尾); EDM 默认 80")
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--p-mean", type=float, default=-1.2, help="论文用 0.0; EDM 默认 -1.2")
    p.add_argument("--p-std", type=float, default=1.2)
    p.add_argument("--paper-noise", action="store_true",
                   help="一键切到论文的噪声设置(sigma_max=800, P_mean=0.0) 做对照")

    # 采样 / 评测
    p.add_argument("--edm-steps", type=int, default=18, help="论文 18 步")
    p.add_argument("--eval-tile", type=int, default=192,
                   help="★分块采样的窗口尺寸(=训练 patch)。整帧采样会发散, 必须分块")
    p.add_argument("--ensemble", type=int, default=16, help="集合成员数(论文 32)")
    p.add_argument("--deterministic-sampler", action="store_true",
                   help="用确定性 Heun 代替论文的随机采样器")
    p.add_argument("--eval-stride", type=int, default=1,
                   help="1=完整 test 年(默认, 汇报必须用此)。>1=子采样, 只做快检, 结果会标 __SUBSAMPLED")
    p.add_argument("--eval-seed", type=int, default=20260723,
                   help="随机集合评测的逐日起始 seed; 实际 seed=eval_seed+day_idx。<0=不固定")
    p.add_argument("--eval-only", action="store_true")
    # 分片评测(多 GCD 并行, 事后 _merge_corrdiff_sums.py 精确合并)
    p.add_argument("--eval-day-i0", type=int, default=0, help="本片起始天索引(进 days 列表)")
    p.add_argument("--eval-day-i1", type=int, default=0, help="本片结束天索引(不含); 0=不分片, 跑全部")
    p.add_argument("--dump-sums", default="", help="分片模式: 把可加累计量落到此 .pkl, 不 finalize")
    p.add_argument("--skip-eval", action="store_true", help="训练与评测解耦(集合采样很贵)")
    p.add_argument("--resume", action="store_true",
                   help="从 last.pt 续训(模型+优化器+EMA+epoch 全恢复)。Frontier 对 <92 节点的作业\n"
                        "硬性限制 2h 墙钟, 凑够论文的 ~5.5 万优化步必须分段跑")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if torch is None:
        sys.exit("需要 PyTorch。")
    if args.paper_noise:
        args.sigma_max, args.p_mean = 800.0, 0.0

    args.model = "corrdiff"
    if args.smoke:
        # ★先记下命令行显式给的值 —— apply_smoke 会把 out/epochs 覆写掉
        user_out = args.out if args.out != DEFAULT_OUT else None
        user_epochs = args.epochs if args.epochs != 40 else None
        TD.apply_smoke(args)
        if user_out:
            args.out = user_out; os.makedirs(user_out, exist_ok=True)
        args.patch = 24; args.reg_epochs = 1; args.eval_tile = 24
        args.epochs = user_epochs if user_epochs else 1         # 显式指定则尊重(为了能测续训)
        args.ensemble = 2; args.edm_steps = 4; args.eval_stride = 10
        args.reg_base = 16; args.base = 16; args.batch = 2; args.workers = 0
        args.steps_per_epoch = 4; args.val_steps = 2; args.regressor_ckpt = ""
        args.stage = "both"
    TD.check_patch(args)                                        # --patch 必须被 FACTOR=6 整除
    run(args)


if __name__ == "__main__":
    main()
