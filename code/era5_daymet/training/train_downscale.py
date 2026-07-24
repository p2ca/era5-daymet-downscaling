#!/usr/bin/env python
# Packaged implementation; code/train_downscale.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
ERA5(120x240) -> Daymet(720x1440) 6x 降尺度: CNN均值 / 条件diffusion, 多节点 DDP
============================================================================
训练 val(2018,2019) -> 测试 test(2020), 在 Daymet 有效域上 masked。

* 归一化/气候态都来自"训练集预计算"(compute_norm_stats.py 的输出), train/val/test 共用:
    输入 ERA5 用 {stats}/era5/normalize_{mean,std}.npz
    目标 Daymet 用 {stats}/daymet/normalize_{mean,std}.npz + climatology.npz
* ERA5 输入变量 与 Daymet 目标变量是两套独立 list(两边变量不同)。
* 目标 = (Daymet - daymet_mean)/daymet_std (~单位方差); 推理 pred = out*daymet_std + daymet_mean。
  条件 = [bilinear(ERA5 输入)归一化, Δz=HR_oro-up(LR_oro), landcover, land_sea_mask, 气候态(条件通道)]。
* 模型: --model cnn(回归距平) | diffusion(条件 DDPM 生成距平, 出 ensemble)。
* 评测: masked RMSE/MAE/bias/corr(集合均值) + CRPS + 功率谱 + rank histogram。

DDP(Frontier/SLURM): srun 每 GPU 一个 rank, NCCL; 见文件底部 sbatch 示例。
自测: python train_downscale.py --smoke --model diffusion (自动造合成数据+stats, 单/多卡都行)

依赖: torch。
============================================================================
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import (
    _load_raw_static,
    _squeeze_2d,
    fill_nan_daymean,
    make_bilinear,
    npz_memmap,
)
from era5_daymet.data.compute_norm_stats import slot_index

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:
    torch = None

FACTOR = 6
TARGETS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]


def precip_fwd(p, clip, scale):
    """降水建模空间: ×scale 变毫米 -> drizzle clip -> log1p(与 compute_norm_stats 一致)。"""
    p = np.maximum(p, 0.0) * scale
    p = np.where(p < clip, 0.0, p)
    return np.log1p(p)


# log1p(mm) 的物理上界: 世界日降水纪录约 1825 mm -> log1p ≈ 7.51。取 8.0 (≈2980 mm) 已极宽松。
PRECIP_LOG_MAX = 8.0


def precip_inv(x, scale, max_log=PRECIP_LOG_MAX, return_clipped=False):
    """log1p(mm) -> m/day。

    ★必须钳制★: 这是 expm1, 是指数函数。生成式模型(扩散)的样本有重尾, 只要有★一个★
    离群像素(比如 log 空间值 50), expm1(50)≈5e21 就会把整幅图的 RMSE 炸成 inf。
    确定性方法从不触发这个 —— L2 损失把它们的输出压得很平, 永远到不了那个量级 ——
    所以这个坑直到接入 CorrDiff 才暴露出来。

    钳到 8.0 对任何真实降水都是无操作(远超世界纪录), 因此不改变任何已记录的确定性结果;
    它只在生成式模型吐出物理上不可能的值时兜底。return_clipped=True 会一并返回被钳的
    像素比例 —— 这个数字本身是个诊断: 训练良好的模型应该≈0, 若显著>0 就是红旗。
    """
    n_clip = float(np.mean(x > max_log)) if return_clipped else 0.0
    p = np.maximum(np.expm1(np.minimum(x, max_log)), 0.0) / scale
    return (p, n_clip) if return_clipped else p

# ★输入合同(2026-07-17, 对齐学长规范): 17 ERA5 动态变量, 严格按此顺序。
#   加 3 静态(dz/lc/lsm) + 3 气候态(逐日, ★保留★) => 条件通道 = 17+3+3 = 23。
#   注: 学长规范删气候态(20 通道), 我们按用户决定保留气候态 -> 23 通道。
DEFAULT_IN = ["2m_temperature", "2m_temperature_max", "2m_temperature_min",
              "total_precipitation_24hr", "10m_u_component_of_wind", "10m_v_component_of_wind",
              "volumetric_soil_water_layer_1", "geopotential_500", "geopotential_850",
              "specific_humidity_500", "specific_humidity_850", "temperature_500", "temperature_850",
              "u_component_of_wind_500", "u_component_of_wind_850",
              "v_component_of_wind_500", "v_component_of_wind_850"]
PRECIP = "total_precipitation_24hr"


# ===========================================================================
# 1. 训练集统计 + 气候态(只加载, 不重算)
# ===========================================================================
class Stats:
    def __init__(self, stats_dir, in_vars, out_vars):
        e_m = np.load(os.path.join(stats_dir, "era5", "normalize_mean.npz"))
        e_s = np.load(os.path.join(stats_dir, "era5", "normalize_std.npz"))
        d_m = np.load(os.path.join(stats_dir, "daymet", "normalize_mean.npz"))
        d_s = np.load(os.path.join(stats_dir, "daymet", "normalize_std.npz"))
        clim = np.load(os.path.join(stats_dir, "daymet", "climatology.npz"))
        meta = json.load(open(os.path.join(stats_dir, "daymet", "meta.json")))
        self.e_mean = np.array([float(e_m[v]) for v in in_vars], np.float32)
        self.e_std = np.array([max(float(e_s[v]), 1e-6) for v in in_vars], np.float32)
        self.d_mean = np.array([float(d_m[v]) for v in out_vars], np.float32)
        self.d_std = np.array([max(float(d_s[v]), 1e-6) for v in out_vars], np.float32)
        self.clim = {v: clim[v].astype(np.float32) for v in out_vars}     # (Nslot,720,1440)
        self.clim_mode = meta["clim"]
        self.precip_log = bool(meta.get("precip_log", False))             # 与 stats 一致的降水变换
        self.precip_clip = float(meta.get("precip_clip", 0.1))
        self.precip_scale = float(meta.get("precip_scale", 1000.0))
        self.oro_std = float(d_s["orography"]) if "orography" in d_s.files else 1000.0
        self.lc_mean = float(d_m["landcover"]) if "landcover" in d_m.files else 0.0
        self.lc_std = float(d_s["landcover"]) if "landcover" in d_s.files else 1.0


# ===========================================================================
# 2. 数据(numpy, 不依赖 torch)
# ===========================================================================
class DownscaleData:
    def __init__(self, era5_dir, daymet_dir, years, in_vars, out_vars, stats, factor=FACTOR):
        self.in_vars, self.out_vars, self.s, self.f = in_vars, out_vars, stats, factor
        self.lr, self.hr, self.dz, self.lc, self.lsm, self.mask, self.slots, self.ndays = ({} for _ in range(8))
        upref = None
        for y in years:
            ef = M.find_year_files(era5_dir, y); df = M.find_year_files(daymet_dir, y)
            if not ef or not df:
                raise FileNotFoundError(f"{y}: 缺 ERA5/Daymet")
            self.lr[y] = {v: fill_nan_daymean(M.load_var_stack(ef, v)) for v in in_vars}
            self.hr[y] = {v: npz_memmap(df[0], v) for v in out_vars}
            T, Hl, Wl = next(iter(self.lr[y].values())).shape
            self.ndays[y] = T
            if upref is None:
                upref = make_bilinear(Hl, Wl, factor); self.Hl, self.Wl = Hl, Wl
                self.H, self.W = Hl * factor, Wl * factor
            lro = M.load_var_stack(ef, "orography")
            lro = fill_nan_daymean(lro)[0] if lro is not None else np.zeros((Hl, Wl), np.float32)
            hro = _squeeze_2d(_load_raw_static(df, "orography"))
            hro = hro if hro is not None else np.zeros((self.H, self.W), np.float32)
            self.dz[y] = (hro.astype(np.float32) - upref(lro.astype(np.float32)))
            lc = _squeeze_2d(_load_raw_static(df, "landcover"))
            self.lc[y] = (lc if lc is not None else np.zeros((self.H, self.W), np.float32)).astype(np.float32)
            lsm = _squeeze_2d(_load_raw_static(df, M.LAND_MASK_VAR))
            self.lsm[y] = (lsm > 0.5).astype(np.float32)
            self.mask[y] = (lsm > 0.5)
            self.slots[y] = slot_index(stats.clim_mode, y, T)
        self.years = list(years)

    def _hr(self, y, v, t):
        d = np.asarray(self.hr[y][v][t]); return d[0] if d.ndim == 3 else d

    def get_patch(self, y, t, y0, x0, Ph, Pw=None):
        Pw = Pw or Ph; f = self.f; s = self.s
        ly0, lx0, lph, lpw = y0 // f, x0 // f, Ph // f, Pw // f
        up = make_bilinear(lph, lpw, f)
        cin = []
        for i, v in enumerate(self.in_vars):
            x = up(self.lr[y][v][t, ly0:ly0 + lph, lx0:lx0 + lpw].astype(np.float32))
            if v == PRECIP and s.precip_log:                  # ERA5 输入降水: ×scale+clip+log1p(去 drizzle)
                x = precip_fwd(x, s.precip_clip, s.precip_scale)
            cin.append((x - s.e_mean[i]) / s.e_std[i])
        cin = np.stack(cin, 0)                                                         # (Cin,Ph,Pw)
        dz = (self.dz[y][y0:y0 + Ph, x0:x0 + Pw] / s.oro_std)[None]
        lc = ((self.lc[y][y0:y0 + Ph, x0:x0 + Pw] - s.lc_mean) / s.lc_std)[None]
        lsm = self.lsm[y][y0:y0 + Ph, x0:x0 + Pw][None]
        slot = int(self.slots[y][t])
        # 气候态作为条件通道(已与 stats 同变换/单位, 直接按 daymet mean/std 归一化)
        climc = np.stack([(s.clim[v][slot, y0:y0 + Ph, x0:x0 + Pw] - s.d_mean[i]) / s.d_std[i]
                          for i, v in enumerate(self.out_vars)], 0)                    # (Cout,Ph,Pw)
        cond = np.concatenate([cin, dz, lc, lsm, climc], 0).astype(np.float32)         # (Cin+3+Cout,Ph,Pw)
        hr = np.stack([self._hr(y, v, t)[y0:y0 + Ph, x0:x0 + Pw] for v in self.out_vars], 0).astype(np.float32)
        ht = hr.copy()
        for i, v in enumerate(self.out_vars):
            if v == PRECIP and s.precip_log:                  # Daymet 目标降水: 同样 ×scale+clip+log1p
                ht[i] = precip_fwd(ht[i], s.precip_clip, s.precip_scale)
        target = ((ht - s.d_mean[:, None, None]) / s.d_std[:, None, None]).astype(np.float32)
        m = self.mask[y][y0:y0 + Ph, x0:x0 + Pw].astype(np.float32)
        return cond, target, m[None], hr                      # hr 保持原值, 给评测当真值

    def random_patch(self, rng, P, min_land=0.3, tries=20):
        for _ in range(tries):
            y = self.years[rng.integers(len(self.years))]
            t = int(rng.integers(self.ndays[y]))
            y0 = int(rng.integers(0, (self.H - P) // self.f + 1)) * self.f
            x0 = int(rng.integers(0, (self.W - P) // self.f + 1)) * self.f
            if self.mask[y][y0:y0 + P, x0:x0 + P].mean() >= min_land:
                return self.get_patch(y, t, y0, x0, P)
        return self.get_patch(y, t, y0, x0, P)

    def full(self, y, t):
        return self.get_patch(y, t, 0, 0, self.H, self.W)


# ===========================================================================
# 3. torch: Dataset / UNet / Diffusion / 指标
# ===========================================================================
if torch is not None:
    class PatchDS(torch.utils.data.Dataset):
        def __init__(self, data, P, length, seed=0):
            self.d, self.P, self.len = data, P, length
            self.rng = np.random.default_rng(seed)
        def __len__(self): return self.len
        def __getitem__(self, i):
            cond, tgt, m, _ = self.d.random_patch(self.rng, self.P)
            return torch.from_numpy(cond), torch.from_numpy(tgt), torch.from_numpy(m)

    class FullFrameDS(torch.utils.data.Dataset):
        """整幅(720x1440)训练用: 每个样本 = 随机一帧完整场(不切窗)。接口同 PatchDS。
        唯一帧数 = Σ_year ndays(~13,870), 故随机有放回采样(length 由训练循环按 epoch 步数定)。"""
        def __init__(self, data, length, seed=0):
            self.d, self.len = data, length
            self.rng = np.random.default_rng(seed)
            self.frames = [(y, t) for y in data.years for t in range(data.ndays[y])]  # (year,day) 平表
        def __len__(self): return self.len
        def __getitem__(self, i):
            y, t = self.frames[int(self.rng.integers(len(self.frames)))]
            cond, tgt, m, _ = self.d.full(y, t)
            return torch.from_numpy(cond), torch.from_numpy(tgt), torch.from_numpy(m)

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

    class UNet(nn.Module):
        def __init__(self, in_ch, out_ch, base=64, temb=0):
            super().__init__()
            self.temb = temb
            if temb:
                self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
            self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
            self.d1 = ResBlock(base, base, temb); self.p1 = nn.Conv2d(base, base, 4, 2, 1)
            self.d2 = ResBlock(base, base * 2, temb); self.p2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
            self.mid = ResBlock(base * 2, base * 2, temb)
            self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1); self.r2 = ResBlock(base * 4, base, temb)
            self.u1 = nn.ConvTranspose2d(base, base, 4, 2, 1); self.r1 = ResBlock(base * 2, base, temb)
            self.outc = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, out_ch, 3, padding=1))
        def forward(self, x, t=None):
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


def crps_ensemble(members, truth, mask):
    N = members.shape[0]; mb = mask > 0.5; out = []
    for c in range(truth.shape[0]):
        mem = members[:, c][:, mb]; y = truth[c][mb]
        t1 = np.abs(mem - y[None]).mean(0)
        t2 = np.abs(mem[:, None] - mem[None, :]).mean((0, 1)) if N > 1 else 0.0
        out.append(float((t1 - 0.5 * t2).mean()))
    return out


def rank_hist(members, truth, mask):
    N = members.shape[0]; mb = mask > 0.5
    mem = members[:, 0][:, mb]; y = truth[0][mb]
    ranks = (mem < y[None]).sum(0) + np.random.randint(0, 2, y.shape) * (mem == y[None]).sum(0)
    h, _ = np.histogram(ranks, bins=np.arange(N + 2))
    return (h / h.sum())


def radial_psd(img, sz):
    img = img - img.mean(); w = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * w))) ** 2
    c0, c1 = np.array(P.shape) // 2; Y, X = np.indices(P.shape); r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def pick_land_box(mask, sz):
    ys, xs = np.where(mask)
    if len(ys) == 0: return (0, 0, sz)
    for y0 in range(ys.min(), max(ys.min() + 1, ys.max() - sz), 40):
        for x0 in range(xs.min(), max(xs.min() + 1, xs.max() - sz), 40):
            if mask[y0:y0 + sz, x0:x0 + sz].all(): return (y0, x0, sz)
    return (int(ys.min()), int(xs.min()), sz)


# ===========================================================================
# 4. DDP
# ===========================================================================
def setup_ddp():
    """SLURM 多节点 DDP。无 SLURM 环境 -> 单进程。返回 (rank, world, local, device, is_dist)。"""
    if torch is None:
        return 0, 1, 0, "cpu", False
    if "SLURM_PROCID" not in os.environ or int(os.environ.get("SLURM_NTASKS", "1")) == 1:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, 0, dev, False
    world = int(os.environ["SLURM_NTASKS"]); rank = int(os.environ["SLURM_PROCID"])
    local = int(os.environ["SLURM_LOCALID"])
    # MASTER_ADDR: 优先用外部导出的; 否则从 SLURM_NODELIST 取第一个节点
    if "MASTER_ADDR" not in os.environ:
        nl = os.environ.get("SLURM_NODELIST", "")
        try:
            import subprocess
            host = subprocess.check_output(["scontrol", "show", "hostnames", nl]).decode().split()[0]
        except Exception:
            host = os.environ.get("HOSTNAME", "127.0.0.1")
        os.environ["MASTER_ADDR"] = host
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(minutes=30), rank=rank, world_size=world)
    if rank == 0:
        print(f"[DDP] world_size={world}  master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", flush=True)
    return rank, world, local, torch.device(f"cuda:{local}"), True


# ===========================================================================
# 5. 训练 + 评测
# ===========================================================================
def run(args):
    rank, world, local, device, is_dist = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"device={device} model={args.model} in={len(args.in_vars)} out={len(args.out_vars)} "
              f"patch={args.patch} world={world}", flush=True)

    stats = Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cout = len(args.out_vars)
    Cin = len(args.in_vars) + 3 + Cout            # ERA5输入 + [Δz,landcover,lsm] + 气候态条件
    in_ch = Cin + (Cout if args.model == "diffusion" else 0)
    temb = 128 if args.model == "diffusion" else 0
    model = UNet(in_ch, Cout, base=args.base, temb=temb).to(device)
    diff = Diffusion(args.diff_steps, device) if args.model == "diffusion" else None

    if not args.eval_only:
        data = DownscaleData(args.era5_dir, args.daymet_dir, args.train_years, args.in_vars, args.out_vars, stats)
        ds = PatchDS(data, args.patch, args.steps_per_epoch * args.batch, seed=1234 + rank)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                         drop_last=True, pin_memory=True)
        ddp = DDP(model, device_ids=[local], static_graph=True) if is_dist else model
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            ddp.train(); t0 = time.time(); tot = 0.0; nb = 0
            for cond, tgt, m in dl:
                cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                loss = diff.loss(ddp, tgt, cond, m) if args.model == "diffusion" else masked_mse(ddp(cond), tgt, m)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            if is_dist:
                lt = torch.tensor([tot, nb], device=device); dist.all_reduce(lt); tot, nb = lt[0].item(), lt[1].item()
            if is_main:
                print(f"  epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} {time.time()-t0:.0f}s", flush=True)
        if is_main:
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
            print("  -> ckpt.pt", flush=True)
        if is_dist:
            dist.barrier()
    else:
        if is_main:
            model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    if is_main:
        if args.eval_only:
            model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
        evaluate(model, diff, stats, args, device)
    if is_dist:
        dist.barrier(); dist.destroy_process_group()


@torch.no_grad() if torch is not None else (lambda f: f)
def evaluate(model, diff, stats, args, device, is_main=True):
    """is_main=False(仅序列并行整幅评测时): 本 rank 只参与每天的前向集合通信(det_predict 里
    模型 all-gather K/V), 不累计指标/不写文件 -> 让整幅评测也 ~1/sp 快, 而非 rank0 单卡 136s/帧。"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    test = DownscaleData(args.era5_dir, args.daymet_dir, [args.test_year], args.in_vars, args.out_vars, stats)
    model.eval(); Cout = len(args.out_vars); y = args.test_year
    days = list(range(0, test.ndays[y], args.eval_stride))
    agg = {v: dict(se=0., ae=0., be=0., n=0., sp=0., st=0., spp=0., stt=0., spt=0., crps=0., nd=0) for v in args.out_vars}
    rh = []; psd_keep = None
    dstd = stats.d_std[:, None, None]; dmean = stats.d_mean[:, None, None]
    for t in days:
        cond, _, m, hr = test.full(y, t)
        cb = torch.from_numpy(cond[None]).float().to(device)
        H, W = cond.shape[1:]
        if args.model == "diffusion":
            mem = [diff.ddim(model, cb, (1, Cout, H, W), steps=args.ddim_steps)[0].cpu().numpy()
                   for _ in range(args.ensemble)]
            members = np.stack(mem, 0)
        else:
            members = det_predict(model, cb, getattr(args, "eval_tile", 0), device)[None]  # SP 下所有 rank 参与
        if not is_main:                    # 非主 rank: 只为集合通信参与前向, 不算指标
            continue
        # 反归一化 -> 物理量; 降水若是 log 空间则 expm1 还原, 并截断到 >=0
        mem_phys = members * dstd[None] + dmean[None]
        if PRECIP in args.out_vars:
            pi = args.out_vars.index(PRECIP)
            mem_phys[:, pi] = (precip_inv(mem_phys[:, pi], stats.precip_scale) if stats.precip_log
                               else np.maximum(mem_phys[:, pi], 0.0))
        ens = mem_phys.mean(0); mb = m[0] > 0.5
        for i, v in enumerate(args.out_vars):
            p = ens[i][mb]; tr = hr[i][mb]; d = p - tr; a = agg[v]
            a["se"] += float((d * d).sum()); a["ae"] += float(np.abs(d).sum()); a["be"] += float(d.sum()); a["n"] += d.size
            a["sp"] += p.sum(); a["st"] += tr.sum(); a["spp"] += (p * p).sum(); a["stt"] += (tr * tr).sum(); a["spt"] += (p * tr).sum()
            a["crps"] += crps_ensemble(mem_phys[:, i:i + 1], hr[i:i + 1], m[0])[0]; a["nd"] += 1
        if members.shape[0] > 1:
            rh.append(rank_hist(mem_phys, hr, m[0]))
        if psd_keep is None:
            box = pick_land_box(mb, min(384, H, W)); psd_keep = (t, hr[0], ens[0], mem_phys[0, 0], box)
    if not is_main:                        # 非主 rank 前向已参与完毕, 指标/写文件仅主 rank
        return
    res = {}
    for v in args.out_vars:
        a = agg[v]; n = max(a["n"], 1)
        cov = a["spt"]/n - (a["sp"]/n)*(a["st"]/n); vp = a["spp"]/n-(a["sp"]/n)**2; vt = a["stt"]/n-(a["st"]/n)**2
        res[v] = dict(rmse=round((a["se"]/n)**.5, 4), mae=round(a["ae"]/n, 4), bias=round(a["be"]/n, 4),
                      corr=round(cov/((vp*vt)**.5), 4) if vp > 0 and vt > 0 else None,
                      crps=round(a["crps"]/max(a["nd"], 1), 4))
    res["_ensemble"] = args.ensemble if args.model == "diffusion" else 1; res["_n_test_days"] = len(days)
    if rh: res["_rank_hist_tmax"] = [float(x) for x in np.mean(rh, 0)]
    json.dump(res, open(os.path.join(args.out, "eval_metrics.json"), "w"), indent=2, default=float)
    print("\n变量 | RMSE | MAE | bias | corr | CRPS")
    for v in args.out_vars:
        r = res[v]; print(f"  {v:26s} {r['rmse']} {r['mae']} {r['bias']} {r['corr']} {r['crps']}")
    if psd_keep:
        t, tr, en, mem, (by, bx, bs) = psd_keep; crop = lambda a: a[by:by+bs, bx:bx+bs]; k = np.arange(1, bs//2)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        im = ax[0].imshow(crop(en), origin="lower", cmap="viridis"); ax[0].set_title(f"{args.model} pred tmax day{t}"); fig.colorbar(im, ax=ax[0], fraction=.046)
        for img, lab in [(tr, "truth"), (en, "ens mean"), (mem, "member"), ]:
            ax[1].loglog(k, radial_psd(crop(img), bs)[1:bs//2], label=lab)
        ax[1].legend(); ax[1].set_title("power spectrum"); ax[1].grid(True, which="both", alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "eval_spectrum.png"), dpi=130, bbox_inches="tight")
    if rh:
        fig, ax = plt.subplots(figsize=(6, 3.5)); n = len(res["_rank_hist_tmax"])
        ax.bar(range(n), res["_rank_hist_tmax"]); ax.axhline(1/n, color="r", ls="--")
        ax.set_title("rank histogram tmax (flat=calibrated)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "eval_rank_hist.png"), dpi=130, bbox_inches="tight")
    print(f"  -> {args.out}/eval_metrics.json (+ spectrum/rank_hist png)")


# ===========================================================================
# 5b. 确定性模型(UNet/ViT)共用: 通用参数 + 早停训练引擎 + 分块推理
# ===========================================================================
def add_common_args(p):
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default="stats_train")
    p.add_argument("--in-vars", nargs="+", default=DEFAULT_IN)
    p.add_argument("--out-vars", nargs="+", default=TARGETS)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])   # 1980-2017
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])       # 2018-2019(早停监控)
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])              # 2020(最终评测)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--val-steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=10, help="val 连续 N 个 epoch 不提升就早停")
    p.add_argument("--warmup", type=int, default=0, help="LR 线性 warmup 的 epoch 数(ViT 建议 5-10)")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="梯度裁剪 max-norm; 0=关闭(UNet baseline 口径)。Transformer 建议 1.0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true",
                   help="开启 bf16 autocast 混合精度(MI250X 上 1.5-2x 提速; bf16 range 够大, 无需 GradScaler)")
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--ddim-steps", type=int, default=50)
    return p


def check_patch(args):
    """--patch 必须被 FACTOR 整除, 否则 get_patch 里 lph=Ph//f 的整除截断会让 cond 通道尺寸打架。

    例: Ph=64, f=6 -> lph=10 -> ERA5 通道上采样回 60x60, 而 dz/lc/lsm/climc 按 Ph 裁成 64x64,
    np.concatenate 报 "size 60 vs 64"。只有在真数据上才会触发(合成自测的 patch 恰好合法),
    所以这里提前拦, 别让它烧掉一次排队 + 启动。
    """
    if args.patch % FACTOR:
        ok = [p for p in range(args.patch - FACTOR, args.patch + FACTOR + 1) if p > 0 and p % FACTOR == 0]
        sys.exit(f"--patch({args.patch}) 必须能被降尺度因子 FACTOR({FACTOR}) 整除。"
                 f"否则 cond 的 ERA5 通道会是 {args.patch // FACTOR * FACTOR}, 而静态/气候态通道是 {args.patch}。"
                 f"就近的合法值: {ok}")


def fit_deterministic(model, stats, args, device, ddp_info, sp_ctx=None):
    """train(train_years) + 早停(val_years, masked MSE) + warmup/余弦 LR; 存 best, 评测时载入。
    sp_ctx!=None: 序列并行(整幅)——SP 组内多 GCD 协同算一帧, 不走 DDP, 手动分类梯度同步。"""
    rank, world, local, is_dist = ddp_info
    is_main = (rank == 0)
    sp = sp_ctx is not None
    # SP 下"数据并行度"= dp_size(帧并行组数); SP 组内各 rank 喂同一帧(种子按 dp_rank), 只在模型里切 token
    dp_size = sp_ctx["dp_size"] if sp else world
    seed_off = sp_ctx["dp_rank"] if sp else rank
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    tr_data = DownscaleData(args.era5_dir, args.daymet_dir, args.train_years, args.in_vars, args.out_vars, stats)
    va_data = DownscaleData(args.era5_dir, args.daymet_dir, args.val_years, args.in_vars, args.out_vars, stats)
    full_frame = getattr(args, "full_frame", False)
    # DDP steps 修复(4.4): 固定"每 epoch 见 epoch_frames 帧", steps 随数据并行度收缩 -> 加节点真省墙钟。
    # (SP 用 dp_size 而非 world: SP 组内是同一帧的 token 切分, 不增加帧吞吐。)
    steps = args.steps_per_epoch
    if full_frame and getattr(args, "epoch_frames", 0) > 0:
        steps = max(1, math.ceil(args.epoch_frames / (dp_size * args.batch)))
        if is_main:
            sp_note = f" | 序列并行 sp_size={dist.get_world_size(sp_ctx['sp_group'])} 单步~1/sp" if sp else ""
            print(f"[full-frame] epoch_frames={args.epoch_frames} dp_size={dp_size} batch/rank={args.batch} "
                  f"-> steps/epoch={steps}, 全局 batch={dp_size*args.batch}{sp_note}", flush=True)
    if full_frame:
        tr_ds = FullFrameDS(tr_data, steps * args.batch, seed=1234 + seed_off)
        va_ds = FullFrameDS(va_data, args.val_steps * args.batch, seed=987)
    else:
        tr_ds = PatchDS(tr_data, args.patch, steps * args.batch, seed=1234 + seed_off)
        va_ds = PatchDS(va_data, args.patch, args.val_steps * args.batch, seed=987)
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, num_workers=args.workers,
                                     drop_last=True, pin_memory=True)
    va = torch.utils.data.DataLoader(va_ds, batch_size=args.batch, num_workers=max(1, args.workers // 2),
                                     drop_last=True)
    if sp:                                                 # 序列并行: 不走 DDP, 手动分类梯度同步
        net = model
        from era5_daymet.models import seq_parallel_attn as SP
        sp_sharded, sp_redundant = model.sp_param_groups()
        # ★初始权重广播: DDP 构造时会自动做, SP 不走 DDP 必须手动 —— 否则各 rank 用各自 RNG
        #   初始化 -> 权重不同 -> SP 组内 all-gather 的 K/V 来自不同权重 -> 前向不连贯 -> 训不动。
        if is_dist:
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            for b in model.buffers():
                dist.broadcast(b.data, src=0)
    else:
        net = DDP(model, device_ids=[local], static_graph=True) if is_dist else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_scale(ep):
        if args.warmup and ep < args.warmup:
            return (ep + 1) / args.warmup
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    use_amp = getattr(args, "amp", False)                 # bf16 autocast(--amp); 默认 fp32 口径不变
    best, bad, ckpt = float("inf"), 0, os.path.join(args.out, "ckpt.pt")
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_scale(ep)
        net.train(); t0 = time.time(); trt = trn = 0.0
        for cond, tgt, m in tr:
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = masked_mse(net(cond), tgt, m)
            opt.zero_grad(); loss.backward()
            if sp:                                        # 序列并行: 手动分类梯度同步(sharded SP-SUM / redundant 不SUM, 都 DP 平均)
                SP.sp_sync_grads(sp_sharded, sp_redundant, sp_ctx["sp_group"], sp_ctx["dp_group"], dp_size)
            if args.grad_clip:                            # Transformer 无裁剪会在 lr>1e-4 附近发散
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            trt += loss.item(); trn += 1                  # ★记 train loss(诊断 train-vs-val 过拟合/欠训)
        model.eval(); vt = vn = 0.0
        with torch.no_grad():
            for cond, tgt, m in va:
                cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    vt += masked_mse(model(cond), tgt, m).item(); vn += 1
        if is_dist:
            tt = torch.tensor([vt, vn, trt, trn], device=device); dist.all_reduce(tt)
            vt, vn, trt, trn = tt[0].item(), tt[1].item(), tt[2].item(), tt[3].item()
        vloss = vt / max(vn, 1); trloss = trt / max(trn, 1)
        improved = vloss < best - 1e-4
        if improved:
            best, bad = vloss, 0
        else:
            bad += 1
        if is_main:
            print(f"  ep {ep+1}/{args.epochs}  train={trloss:.4f}  val={vloss:.4f}  best={best:.4f}  bad={bad}/{args.patience}  "
                  f"lr={args.lr*lr_scale(ep):.2e}  {time.time()-t0:.0f}s", flush=True)
            if improved:
                torch.save({"model": model.state_dict(), "args": vars(args), "val_best": best, "epoch": ep}, ckpt)
        if bad >= args.patience:                      # ★ 早停: 连续 patience 个 epoch 不提升
            if is_main:
                print(f"  早停: val 连续 {args.patience} 个 epoch 没提升 (best={best:.4f})", flush=True)
            break
    if is_dist:
        dist.barrier()
    if is_main and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])   # 评测用 best
    if sp and is_dist:                          # SP 整幅评测要各 rank 权重一致 -> 从 rank0 广播 best
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    return ckpt


def _feather_window(tile, ov):
    """羽化窗口 (tile,tile): 块边缘的 ov 像素线性降权, 内部为 1。

    ★为什么: 分块推理时若用★均匀平均★(每块权重相同), 重叠区覆盖数从 1 跳到 2 不连续,
    且每块在自己边缘处上下文最少、预测最差 -> 拼出规则的"一格一格"接缝。羽化窗口让
    每块只在重叠带里平滑过渡, 边缘降权、内部满权, 接缝被抹平。
    ★最小权重 = 1/(ov+1) > 0 (不到 0): 保证域边界(只被单块覆盖处)除法不为 0、且恢复原值。
    """
    r = (np.arange(ov) + 1) / (ov + 1)                     # (0,1) 之间, 不含端点
    w1 = np.ones(tile, np.float32)
    w1[:ov] = r; w1[-ov:] = r[::-1]
    return np.outer(w1, w1).astype(np.float32)             # (tile,tile)


def det_predict(model, cond, tile, device, tile_batch=32):
    """确定性预测 -> (Cout,H,W) numpy。tile=0: 整帧(全卷积 UNet); tile>0: 分块+★羽化加权融合★(ViT 必须, 固定 token 数)。

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
            out[:, y0:y0 + tile, x0:x0 + tile] += p * win[None]      # ★加权累加
            wsum[y0:y0 + tile, x0:x0 + tile] += win
    return out / np.maximum(wsum, 1e-6)[None]


# ===========================================================================
# 6. 合成自测
# ===========================================================================
def make_synth(root):
    import compute_norm_stats as CN
    Hl, Wl, f, N = 24, 24, FACTOR, 40
    for y in (2018, 2019, 2020):
        sp = "val" if y < 2020 else "test"
        ed = f"{root}/era5/{sp}"; dd = f"{root}/daymet/{sp}"; os.makedirs(ed, exist_ok=True); os.makedirs(dd, exist_ok=True)
        rng = np.random.default_rng(y); oro = (rng.random((Hl, Wl)) * 1000).astype(np.float32)
        lr = {v: (rng.random((N, Hl, Wl)).astype(np.float32) * (5 if "precip" in v else 30) + (0 if "precip" in v else 270)) for v in TARGETS}
        np.savez(f"{ed}/{y}.npz", **lr, orography=np.repeat(oro[None], N, 0), valid_mask=np.ones((Hl, Wl), bool))
        up = make_bilinear(Hl, Wl, f); hro = up(oro) + rng.random((Hl*f, Wl*f)).astype(np.float32) * 200
        d = {v: (np.stack([up(lr[v][t]) for t in range(N)]) + rng.standard_normal((N, Hl*f, Wl*f)).astype(np.float32) * (2 if "precip" in v else 1))[:, None] for v in TARGETS}
        np.savez(f"{dd}/{y}_0.npz", **d, orography=hro, landcover=rng.integers(0, 10, (Hl*f, Wl*f)).astype(np.float32),
                 land_sea_mask=np.ones((Hl*f, Wl*f), np.float32))
    CN.process_side("era5", f"{root}/era5", [2018, 2019], "monthly", 1, 15, f"{root}/stats", False)
    CN.process_side("daymet", f"{root}/daymet", [2018, 2019], "monthly", 1, 15, f"{root}/stats", True)
    return f"{root}/era5", f"{root}/daymet", f"{root}/stats"


def apply_smoke(args):
    """合成数据 + stats, 并把 args 缩成秒级自测(train_unet/train_vit 共用)。"""
    import tempfile
    root = tempfile.mkdtemp()
    args.era5_dir, args.daymet_dir, args.stats_dir = make_synth(root)
    args.in_vars = TARGETS; args.out_vars = TARGETS
    args.train_years = [2018]; args.val_years = [2019]; args.test_year = 2020
    args.patch = 48; args.batch = 4; args.epochs = 3; args.steps_per_epoch = 6; args.val_steps = 3
    args.patience = 2; args.warmup = 1; args.workers = 0; args.eval_stride = 10; args.out = root + "/out"
    return root


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=["cnn", "diffusion"], default="cnn")
    p.add_argument("--era5-dir", default=M.ERA5_DIR); p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--stats-dir", default="stats_train")
    p.add_argument("--in-vars", nargs="+", default=DEFAULT_IN)
    p.add_argument("--out-vars", nargs="+", default=TARGETS)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])
    p.add_argument("--out", default="runs/exp"); p.add_argument("--ckpt", default="runs/exp/ckpt.pt")
    p.add_argument("--patch", type=int, default=192); p.add_argument("--base", type=int, default=64)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--diff-steps", type=int, default=1000); p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument("--eval-stride", type=int, default=1,
                   help="1=完整 test 年(默认, 汇报必须用此)。>1=子采样, 结果会被 eval_common 标 __SUBSAMPLED")
    p.add_argument("--bucket-cap-mb", type=int, default=25)
    p.add_argument("--eval-only", action="store_true"); p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if torch is None:
        sys.exit("需要 torch(Frontier GPU)。本机仅可做数据/语法检查。")
    if args.smoke:
        import tempfile
        root = tempfile.mkdtemp()
        args.era5_dir, args.daymet_dir, args.stats_dir = make_synth(root)
        args.in_vars = TARGETS; args.out_vars = TARGETS
        args.train_years = [2018, 2019]; args.test_year = 2020
        args.patch = 48; args.base = 16; args.batch = 4; args.epochs = 1; args.steps_per_epoch = 8
        args.workers = 0; args.ensemble = 4; args.eval_stride = 10; args.diff_steps = 50; args.ddim_steps = 8
        args.out = root + "/out"
        print("[smoke] 合成数据+stats 自测 ...", flush=True)
    run(args)


if __name__ == "__main__":
    main()
