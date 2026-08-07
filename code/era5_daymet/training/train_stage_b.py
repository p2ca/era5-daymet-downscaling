#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
train_stage_b.py — CorrDiff 阶段 B: patched residual diffusion
============================================================================
冻结的阶段 A 均值 μ 从缓存读入(与逐步重算数值等价, 见 data/mu_cache.py), 在全域上形成
残差 y - μ 后交给官方 RandomPatching2D 切块, 由官方 ResidualLoss + EDM 预条件训练。

★ 切块发生在残差形成之后, 因此 μ 与 patch 位置无关 —— 这是缓存成立的前提, 也是官方
  实现的固有顺序; 反过来先切块再算 μ 会因 GroupNorm 统计量与卷积边界而严重失真。

★ patch 位置必须每步重掷: 官方靠 set_patch_num() 的副作用完成, 直接改 patch_indices
  不会触发。此处显式调用并断言位置确有变化。

损失只在陆地上归一化: 海洋从未被监督, 计入会稀释梯度并抬高表观分数。
============================================================================
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.mu_cache import MuCache
from era5_daymet.models.corrdiff_loss import ResidualLoss
from era5_daymet.models.patching import RandomPatching2D
from era5_daymet.models.preconditioning import EDMPrecondSuperResolution
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.stage_b_mean import CachedRegressionMean, pin_ocean


class FrameStream(torch.utils.data.Dataset):
    """洗牌无放回的整幅流, 每个样本给出 (cond, target, mu, land)。

    取帧口径与阶段 A 的 FullFrameDS 一致: 全局序号决定第几遍数据与帧号, 跨 rank 分片
    互不重叠, 无放回。
    """

    def __init__(self, data, cache, target, length, seed, index_offset, epoch_span,
                 deterministic=False):
        self.d, self.cache, self.target = data, cache, target
        self.ti = TD.TARGETS.index(target)
        self.len, self.base_seed = length, int(seed)
        self.index_offset, self.epoch_span = int(index_offset), int(epoch_span)
        self.deterministic, self.epoch = deterministic, 0
        self.frames = [(y, t) for y in data.years for t in range(data.ndays[y])]
        self.perm = (np.random.default_rng(self.base_seed).permutation(len(self.frames))
                     if deterministic else None)
        self._pid, self._pperm = -1, None

    def __len__(self):
        return self.len

    def _perm_for(self, k):
        if self._pid != k:
            self._pid = k
            self._pperm = np.random.default_rng(
                np.random.SeedSequence([self.base_seed, int(k)])).permutation(len(self.frames))
        return self._pperm

    def __getitem__(self, i):
        n = len(self.frames)
        if self.deterministic:
            y, t = self.frames[int(self.perm[(self.index_offset + int(i)) % len(self.perm)])]
        else:
            g = self.epoch * self.epoch_span + self.index_offset + int(i)
            y, t = self.frames[int(self._perm_for(g // n)[g % n])]
        cond, tgt, mask, _ = self.d.full(y, t)
        mu = self.cache.get(self.target, y, t)[None]
        land = (mask[0] > 0.5).astype(np.float32)[None]
        return (torch.from_numpy(cond), torch.from_numpy(tgt[self.ti:self.ti + 1]),
                torch.from_numpy(mu), torch.from_numpy(land))


def reseat_patches(patching, land, patch, min_land, rng, max_try=200):
    """重掷 patch 位置, 并要求每块的陆地占比不低于阈值。

    官方以 set_patch_num() 的副作用重掷; 这里在其之上加陆地约束 —— 全海的块不含任何
    被监督的像素, 会浪费一次优化更新。
    """
    H, W = land.shape[-2:]
    picked = []
    for _ in range(max_try):
        if len(picked) >= patching.patch_num:
            break
        i = int(rng.integers(0, H - patch + 1)); j = int(rng.integers(0, W - patch + 1))
        if float(land[..., i:i + patch, j:j + patch].mean()) >= min_land:
            picked.append((i, j))
    while len(picked) < patching.patch_num:                     # 兜底: 放宽约束
        picked.append((int(rng.integers(0, H - patch + 1)), int(rng.integers(0, W - patch + 1))))
    patching.patch_indices = picked


def _rng_capture(rank, rng):
    """本 rank 的随机流状态: torch CPU/CUDA 噪声流 + numpy patch 位置流。"""
    st = {"rank": rank, "torch_cpu": torch.get_rng_state(),
          "np_patch": rng.bit_generator.state}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def _rng_gather(is_dist, world, st):
    """收齐全 rank 的 RNG 状态供 rank0 入盘; 所有 rank 必须同时调用。"""
    if not is_dist:
        return [st]
    lst = [None] * world
    dist.all_gather_object(lst, st)
    return lst


def main():
    p = argparse.ArgumentParser(description="CorrDiff 阶段 B 训练")
    p.add_argument("--cache", required=True)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])
    # 结构
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 2])
    p.add_argument("--n-grid-channels", type=int, default=100)
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--patch-num", type=int, default=1)
    p.add_argument("--min-land", type=float, default=0.10)
    # EDM
    p.add_argument("--p-mean", type=float, default=-1.2)
    p.add_argument("--p-std", type=float, default=1.2)
    p.add_argument("--sigma-data", type=float, required=True)
    # 训练
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1e6)
    p.add_argument("--duration", type=int, default=2_000_000, help="processed samples 上限")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="训练循环超过该秒数后保存断点并干净退出; 0 表示只按 duration 停。"
                        "分段训练用: duration 固定不变(取帧流的版图), 每段由本参数收尾")
    p.add_argument("--ckpt-every", type=int, default=2048, help="每多少 samples 存一次")
    p.add_argument("--snap-every", type=int, default=524288,
                   help="每多少 samples 存一个不覆盖的 model-only 快照 snap_*.pt(0 关闭); "
                        "供时长消融与尾窗平均, 独立于 best/last 断点")
    p.add_argument("--save-rng", type=int, default=0,
                   help="断点附带全 rank 的 RNG 状态, 续训时按 rank 恢复以消除随机流不连续; "
                        "涉及跨 rank 收集, 默认关闭")
    p.add_argument("--val-every", type=int, default=8192)
    p.add_argument("--val-steps", type=int, default=4)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    args = p.parse_args()

    rank, world, local, device, is_dist = TD.setup_ddp()
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    cache = MuCache(args.cache, [args.target]); cache.verify({args.target: args.stage_a_ckpt})
    stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    tr = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.train_years,
                          TD.DEFAULT_IN, TD.TARGETS, stats)
    va = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.val_years,
                          TD.DEFAULT_IN, TD.TARGETS, stats)
    H, W = tr.H, tr.W

    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_in_channels=41 + args.n_grid_channels, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=args.model_channels,
        channel_mult=list(args.channel_mult), attn_resolutions=[16],
        N_grid_channels=args.n_grid_channels, gridtype="learnable",
        sigma_data=args.sigma_data, amp_mode=True).to(device)
    reg = CachedRegressionMean(out_channels=1).to(device)
    loss_fn = ResidualLoss(regression_net=reg, P_mean=args.p_mean, P_std=args.p_std,
                           sigma_data=args.sigma_data, hr_mean_conditioning=True)
    model = DDP(net, device_ids=[local]) if is_dist else net
    if is_main:
        gp = net.model.pos_embd.numel(); tot = sum(q.numel() for q in net.parameters())
        print(f"[stageB] target={args.target} 参数 {tot:,} (位置网格 {gp:,} / 网络 {tot-gp:,}) "
              f"world={world} patch={args.patch} patch_num={args.patch_num}", flush=True)
        print(f"[stageB] sigma_data={args.sigma_data} P_mean={args.p_mean} lr={args.lr} "
              f"duration={args.duration:,} samples", flush=True)

    per_step = world * args.patch_num
    steps_total = args.duration // per_step
    span = world                                        # 每步全体 rank 消耗的帧数
    # ★index_offset 必须按"每 rank 取样数"分块, 而不是按 rank 号偏移 1:
    #   后者会让相邻 rank 的取帧序列几乎完全重合(实测 rank0 与 rank1 共享 100% 的帧),
    #   全体 rank 只覆盖训练集的一小部分。
    # 续训: 8M samples 在单个 24h 墙钟内未必跑得完, 因此断点必须能真正接上。
    # 取帧序列由全局序号唯一决定, 故只需把本 rank 的起点后移 done_steps, 长度相应缩短 ——
    # 既不重复已训过的帧, 也不跳过未训的帧。
    ck = None
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    done_steps = (ck["samples"] // per_step) if ck else 0
    if done_steps >= steps_total:
        raise SystemExit(f"断点已达 {ck['samples']:,} samples >= duration {args.duration:,}")

    ds = FrameStream(tr, cache, args.target, steps_total - done_steps, 1234,
                     rank * steps_total + done_steps, world * steps_total)
    vs = FrameStream(va, cache, args.target, args.val_steps, 987, rank * args.val_steps,
                     0, deterministic=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.workers,
                                     pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(vs, batch_size=1, num_workers=max(1, args.workers // 2))

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0)
    if ck is not None:
        net.load_state_dict(ck["model"])                 # 各 rank 读同一文件, 参数天然一致
        opt.load_state_dict(ck["opt"])
        if is_main:
            print(f"[stageB] 从 {args.resume} 续训: 已完成 {ck['samples']:,} samples "
                  f"({done_steps:,} 步), 剩余 {steps_total - done_steps:,} 步", flush=True)

    patching = RandomPatching2D(img_shape=(H, W), patch_shape=(args.patch, args.patch),
                                patch_num=args.patch_num)
    rng = np.random.default_rng(args.seed + rank)
    if args.save_rng and ck is not None:
        saved = ck.get("rng")
        if saved and len(saved) == world:
            st = saved[rank]
            torch.set_rng_state(st["torch_cpu"])
            if "torch_cuda" in st and torch.cuda.is_available():
                torch.cuda.set_rng_state(st["torch_cuda"])
            rng.bit_generator.state = st["np_patch"]
            if is_main:
                print("[stageB] RNG 状态已按 rank 恢复, 随机流与断点前连续", flush=True)
        elif is_main:
            print("[stageB] 断点无匹配的 RNG 状态(缺失或 world 不同), 使用新随机流", flush=True)

    # 一次性机制检查: 官方靠 set_patch_num() 的副作用重掷位置, 忘了调就会全程用同一批
    # 位置且不报错。这里连抽若干次确认位置确实在变 —— 不能放到每步做, 因为单个 patch 的
    # 位置空间有限, 相邻两次偶然重合是合法事件, 拿它中断训练会白白拖垮整个作业。
    _seen = set()
    for _ in range(8):
        patching.set_patch_num(args.patch_num)
        _seen.add(tuple(patching.patch_indices))
    if len(_seen) < 2:
        raise RuntimeError("set_patch_num() 未重掷 patch 位置; 官方 patching 行为异常")
    if is_main:
        print(f"[stageB] patch 重掷机制自检通过 ({len(_seen)}/8 次取到不同位置)", flush=True)

    def run_batch(cond, tgt, mu, land, train=True, prng=None):
        cond = cond.to(device, non_blocking=True); tgt = tgt.to(device, non_blocking=True)
        mu = pin_ocean(mu.to(device, non_blocking=True), land.to(device, non_blocking=True))
        land = land.to(device, non_blocking=True)
        patching.set_patch_num(args.patch_num)           # ★官方靠此副作用重掷位置
        reseat_patches(patching, land, args.patch, args.min_land, prng if prng is not None else rng)
        reg.set(mu)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device != "cpu")):
            l = loss_fn(net=model.module if is_dist else net, img_clean=tgt, img_lr=cond,
                        patching=patching, use_patch_grad_acc=False)
        lp = patching.apply(land)                        # 与损失同一批位置
        return (l.float() * lp).sum() / lp.sum().clamp_min(1.0)

    seen, t0, hist = done_steps * per_step, time.time(), []
    best = float("inf")
    hf = out / "loss_history.json"
    if ck is not None and hf.exists():                   # 曲线要能跨作业拼成一条
        hist = [h for h in json.loads(hf.read_text()) if h["samples"] <= seen]
        if hist:
            best = min(h["val"] for h in hist)
    run_sum = run_n = 0.0                    # 区间内训练损失的累计, 供求平均
    gn_sum, gn_max = 0.0, 0.0                # 区间内裁剪前梯度范数(均值与峰值)
    net.train()
    # 不再逐步推进 ds.epoch: 全局序号已由 index_offset + i 覆盖整个训练预算, 第几遍数据
    # 由 g // n 自动决定。况且 DataLoader 的 worker 是 fork 出去的副本, 在主进程改 epoch
    # 也传不到 worker。
    for step, (cond, tgt, mu, land) in enumerate(dl, 1):
        loss = run_batch(cond, tgt, mu, land, train=True)
        opt.zero_grad(); loss.backward()
        # DDP 已同步梯度, 各 rank 的范数相同; 裁剪前范数入曲线供稳定性监控
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
        opt.step()
        seen += per_step
        run_sum += float(loss.detach()); run_n += 1.0
        gn_sum += gn; gn_max = max(gn_max, gn)
        if seen % args.val_every < per_step:
            # ★验证必须去掉采样噪声, 否则曲线读不出趋势: 扩散损失的 sigma 与噪声逐次随机,
            #   patch 位置也逐次随机, 两重随机叠加会淹没真实变化。这里固定 patch 位置的
            #   随机源与 torch 的噪声种子, 并在结束后还原训练侧状态, 不扰动训练。
            net.eval(); vt = vn = 0.0
            st_cpu = torch.get_rng_state()
            st_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            vrng = np.random.default_rng(20260803 + rank)     # 每次验证都从同一状态开始
            with torch.no_grad():
                for k, (c2, t2, m2, l2) in enumerate(vl):
                    torch.manual_seed(4242 + k)                # 固定该批的 sigma 与噪声
                    vt += float(run_batch(c2, t2, m2, l2, train=False, prng=vrng)); vn += 1
            torch.set_rng_state(st_cpu)
            if st_cuda is not None:
                torch.cuda.set_rng_state_all(st_cuda)
            net.train()
            v = vt / max(vn, 1)
            tr_avg = run_sum / max(run_n, 1)                   # 区间平均, 而非末步单点
            gn_avg, gn_pk = gn_sum / max(run_n, 1), gn_max
            run_sum = run_n = 0.0
            gn_sum, gn_max = 0.0, 0.0
            if is_dist:
                q = torch.tensor([v, tr_avg, 1.0], device=device); dist.all_reduce(q)
                v = float(q[0] / q[2]); tr_avg = float(q[1] / q[2])
            hist.append({"samples": seen, "train": tr_avg, "val": v,
                         "gnorm_mean": round(gn_avg, 6), "gnorm_max": round(gn_pk, 6),
                         "seconds": round(time.time() - t0, 1)})
            if is_main:
                print(f"[stageB] {seen:>9,} samples  train={tr_avg:.5f}  "
                      f"val={v:.5f}  {time.time()-t0:.0f}s", flush=True)
                if v < best:
                    best = v
                    TD._atomic_torch_save({"model": net.state_dict(), "opt": opt.state_dict(),
                                           "samples": seen, "val": v, "args": vars(args)},
                                          out / "ckpt.pt")
                tmp = out / f"loss_history.json.tmp.{os.getpid()}"
                tmp.write_text(json.dumps(hist, indent=1))
                os.replace(tmp, out / "loss_history.json")
        do_ckpt = seen % args.ckpt_every < per_step          # seen 全 rank 一致, 判定同步
        rng_states = None
        if args.save_rng and do_ckpt:
            rng_states = _rng_gather(is_dist, world, _rng_capture(rank, rng))
        if is_main and do_ckpt:
            payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                       "samples": seen, "args": vars(args)}
            if rng_states is not None:
                payload["rng"] = rng_states
            TD._atomic_torch_save(payload, out / "last.pt")
        if is_main and args.snap_every > 0 and seen % args.snap_every < per_step:
            # model-only 快照, 不覆盖: 供时长消融与尾窗平均
            TD._atomic_torch_save({"model": net.state_dict(), "samples": seen},
                                  out / f"snap_{seen:09d}.pt")
        if seen >= args.duration:
            break
        # 分段训练的收尾开关。停止判定必须全 rank 一致, 否则先退出的 rank 会让其余 rank
        # 卡死在下一次集合通信上; 以 rank0 的时钟为准广播。
        if args.max_seconds > 0:
            timeup = torch.tensor(
                [1.0 if (rank == 0 and time.time() - t0 >= args.max_seconds) else 0.0],
                device=device)
            if is_dist:
                dist.broadcast(timeup, src=0)
            if timeup.item() > 0:
                if is_main:
                    print(f"[stageB] 达到 --max-seconds={args.max_seconds:.0f}s, "
                          f"保存断点后退出 ({seen:,} samples)", flush=True)
                break

    rng_states = None
    if args.save_rng:
        rng_states = _rng_gather(is_dist, world, _rng_capture(rank, rng))
    if is_main:
        # 退出前无条件补写断点: 周期写盘停在 ckpt_every 的整数倍上, 不含最后的零头
        payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                   "samples": seen, "args": vars(args)}
        if rng_states is not None:
            payload["rng"] = rng_states
        TD._atomic_torch_save(payload, out / "last.pt")
        print(f"[stageB] 完成 {seen:,} samples, best val={best:.5f}, "
              f"{time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
