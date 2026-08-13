#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
train_jit.py — 整幅像素空间条件扩散 (JiT / JiTMoE)
============================================================================
单目标条件生成 p(y | 20 通道条件场): 不做残差分解、不依赖回归均值模型, 整幅
720x1440 直接训练。范式与超参遵循 JiT (arXiv:2511.13720):

  z = t*y + (1-t)*noise_scale*e,  t ~ sigmoid(N(P_mean, P_std^2)),  t=1 为数据端
  网络输出 x 预测, 损失在 v 空间: || (y - x_hat) / max(1-t, t_eps) ||^2

口径约定:
  - 目标场在海洋置 0 后参与加噪(z 的统计处处良定), 损失默认只在陆地归一化;
    采样侧配合把每步 x 预测的海洋区钳 0(见 models/jit_sampler.py)。
  - lr = blr * 全局批 / 256(线性缩放), 逐步线性 warmup 后恒定; AdamW(0.9, 0.95),
    无权重衰减, 无梯度裁剪(--grad-clip 默认 1e6 仅作范数监控)。
  - 每步维护两份参数 EMA(采样默认用 ema1)。EMA 只覆盖参数; MoE 路由偏置是 buffer,
    随 model state 保存, 导出 EMA 权重采样时由加载方从 model state 取 buffer。
  - 取帧为全局序号决定的无放回洗牌流(与其余整幅训练一致), 断点续训逐帧精确;
    扩散的 (t, 噪声) 走逐 rank 专属 Generator, --save-rng 时随断点入盘 -> 续训后
    随机流逐位连续, 且不受任何库消耗全局随机流的影响。
  - --moe 时偶数序块(索引 1,3,...)的 FFN 换成 DeepSeek 风格稀疏层; 逐层专家负载
    全程记录进损失曲线; --bias-gamma > 0 启用免辅助损失负载均衡。
============================================================================
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.models.jit_backbone import JiT
from era5_daymet.training import train_downscale as TD


class JitFrameStream(torch.utils.data.Dataset):
    """整幅帧流: 每个样本 = (cond, 单目标 target, land)。

    全局序号 g = index_offset + i 唯一决定帧: 第 g//n 遍数据用只依赖 (seed, 遍数) 的
    置换取第 g%n 帧。跨 rank 分片互不重叠、无放回; deterministic=True 时用固定置换,
    供逐次完全一致的验证。
    """

    def __init__(self, data, target, length, seed, index_offset, deterministic=False):
        self.d = data
        self.ti = TD.TARGETS.index(target)
        self.len, self.base_seed = int(length), int(seed)
        self.index_offset, self.deterministic = int(index_offset), deterministic
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

    def frame_of(self, i):
        """样本序号 -> (年, 日)。索引数学集中于此, 供分片覆盖检验直接调用。"""
        n = len(self.frames)
        if self.deterministic:
            return self.frames[int(self.perm[(self.index_offset + int(i)) % n])]
        g = self.index_offset + int(i)
        return self.frames[int(self._perm_for(g // n)[g % n])]

    def __getitem__(self, i):
        y, t = self.frame_of(i)
        cond, tgt, mask, _ = self.d.full(y, t)
        land = (mask[0] > 0.5).astype(np.float32)[None]
        return (torch.from_numpy(cond), torch.from_numpy(tgt[self.ti:self.ti + 1]),
                torch.from_numpy(land))


def jit_vloss(net, tgt, cond, weight, noise_scale, p_mean, p_std, t_eps, generator=None):
    """JiT 训练损失。恒等式 (v - v_pred) = (y - x_hat)/(1-t) 使 v 差可由 x 差直接算出。
    t 与噪声取自 generator(训练用专属流并随断点入盘 -> 续训逐位连续, 且不受其他库
    消耗全局随机流的影响; 验证用逐批固定种子的临时流)。"""
    B = tgt.shape[0]
    t = torch.sigmoid(torch.randn(B, device=tgt.device, generator=generator)
                      * p_std + p_mean)
    tb = t.view(B, 1, 1, 1)
    e = torch.randn(tgt.shape, device=tgt.device, dtype=tgt.dtype,
                    generator=generator) * noise_scale
    z = tb * tgt + (1.0 - tb) * e
    x_hat = net(z, t, cond)
    diff = (tgt - x_hat.float()) / (1.0 - tb).clamp_min(t_eps)
    return (diff.square() * weight).sum() / weight.sum().clamp_min(1.0)


def _rng_capture(rank, gen):
    st = {"rank": rank, "gen": gen.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def _rng_gather(is_dist, world, st):
    if not is_dist:
        return [st]
    lst = [None] * world
    dist.all_gather_object(lst, st)
    return lst


def _fingerprint(net, grads=False):
    """逐参数(或其梯度)与 buffer 的 (sum, |sum|, sq-sum) 指纹, 供跨 rank 一致性自检。
    梯度缺席记为 NaN 三元组 —— 缺席模式本身也必须全 rank 一致, 否则优化器步进会分叉。"""
    vals = []
    for _, p in net.named_parameters():
        t = p.grad if grads else p
        if t is None:
            vals += [float("nan"), float("nan"), float("nan")]
        else:
            td = t.detach().double()
            vals += [float(td.sum()), float(td.abs().sum()), float(td.square().sum())]
    if not grads:
        for _, b in net.named_buffers():
            if b.dtype.is_floating_point:
                td = b.detach().double()
                vals += [float(td.sum()), float(td.abs().sum()),
                         float(td.square().sum())]
    return torch.tensor(vals, dtype=torch.float64)


def _assert_ranks_synced(vec, device, world, what):
    """all_gather 指纹并逐 rank 比对; 不一致说明 DDP 同步失效, 必须立即中止而不是
    带着单卡血统的权重跑完全程。"""
    v = vec.to(device)
    outs = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(outs, v)
    for k in range(1, world):
        if not torch.allclose(outs[0], outs[k], rtol=0, atol=1e-10, equal_nan=True):
            bad = int((~torch.isclose(outs[0], outs[k], rtol=0, atol=1e-10,
                                      equal_nan=True)).sum())
            raise RuntimeError(f"[jit] rank0 与 rank{k} 的{what}不一致 "
                               f"({bad} 项越界) —— DDP 同步失效, 中止训练")


def _ema_init(net):
    return {n: p.detach().clone().float() for n, p in net.named_parameters()}


@torch.no_grad()
def _ema_update(ema, net, decay):
    for n, p in net.named_parameters():
        ema[n].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def _drain_moe_load(layers, is_dist):
    """取出并清零各 MoE 层的专家命中计数, 分布式下聚成全局计数。返回 (L, E) 或 None。"""
    if not layers:
        return None
    c = torch.stack([m.pop_load() for m in layers])
    if is_dist:
        dist.all_reduce(c)
    return c


def build_model(a, hw):
    """按参数字典构建 JiT(a 取 vars(args) 或 checkpoint 里保存的 args)。
    训练与离线采样/评测共用, 保证从 checkpoint 重建的结构与训练时逐项一致。"""
    moe_config = None
    if a.get("moe"):
        moe_config = {
            "num_experts": a["experts"],
            "moe_intermediate_size": a["moe_intermediate"] or 2 * a["hidden"],
            "num_experts_per_tok": a["experts_per_tok"],
            "n_group": 2, "topk_group": 2,
            "routed_scaling_factor": a["routed_scaling"],
            "interleave": not a["moe_all_layers"],
            "use_shared_expert": not a["moe_no_shared"],
            "proj_drop": a["proj_dropout"],
        }
    return JiT(hw=hw, patch=a["patch"], cond_ch=a["cond_ch"], out_ch=1,
               hidden=a["hidden"], depth=a["depth"], num_heads=a["heads"],
               mlp_ratio=a["mlp_ratio"], bottleneck=a["bottleneck"],
               attn_drop=a["attn_dropout"], proj_drop=a["proj_dropout"],
               moe_config=moe_config)


def main(argv=None, data=None):
    p = argparse.ArgumentParser(description="JiT / JiTMoE 整幅条件扩散训练")
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stats-dir", default="")
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--cond-ch", type=int, default=20)
    # 结构
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--proj-dropout", type=float, default=0.0)
    # MoE
    p.add_argument("--moe", action="store_true", help="偶数序块 FFN 换成 DeepSeek 稀疏层")
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--experts-per-tok", type=int, default=2)
    p.add_argument("--moe-intermediate", type=int, default=0, help="0 -> 2*hidden")
    p.add_argument("--routed-scaling", type=float, default=2.5)
    p.add_argument("--moe-all-layers", action="store_true", help="全部块用 MoE(消融)")
    p.add_argument("--moe-no-shared", action="store_true", help="去共享专家(消融)")
    p.add_argument("--bias-gamma", type=float, default=0.0,
                   help="免辅助损失负载均衡的偏置步长; 0 = 偏置恒零(只记录负载)")
    # 扩散
    p.add_argument("--p-mean", type=float, default=-0.8)
    p.add_argument("--p-std", type=float, default=0.8)
    p.add_argument("--noise-scale", type=float, default=4.0,
                   help="噪声幅度; 参考口径为等效边长/256")
    p.add_argument("--t-eps", type=float, default=0.05)
    # 训练
    p.add_argument("--lr", type=float, default=0.0, help="绝对学习率; 0 -> 用 --blr 缩放")
    p.add_argument("--blr", type=float, default=5e-5, help="lr = blr * 全局批 / 256")
    p.add_argument("--warmup-samples", type=int, default=70_000)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--ema1", type=float, default=0.9999)
    p.add_argument("--ema2", type=float, default=0.9996)
    p.add_argument("--batch", type=int, default=1, help="每 rank 每步帧数")
    p.add_argument("--grad-clip", type=float, default=1e6)
    p.add_argument("--duration", type=int, default=8_000_000, help="processed samples 上限")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="超时保存断点并干净退出; 0 表示只按 duration 停")
    p.add_argument("--ckpt-every", type=int, default=8192)
    p.add_argument("--snap-every", type=int, default=524_288,
                   help="model+ema1 快照 snap_*.pt 的间隔(0 关闭)")
    p.add_argument("--save-rng", type=int, default=1)
    p.add_argument("--val-every", type=int, default=8192)
    p.add_argument("--val-steps", type=int, default=4, help="每 rank 的验证批数")
    p.add_argument("--loss-scope", choices=["land", "all"], default="land")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    args = p.parse_args(argv)

    rank, world, local, device, is_dist = TD.setup_ddp()
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)          # 全局流只用于权重初始化(DDP 构造时统一)
    gen = torch.Generator(device=device)         # 扩散 (t, 噪声) 的专属流, 逐 rank 独立
    gen.manual_seed(args.seed * 100_003 + rank)

    if data is None:
        stats = TD.Stats(args.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
        tr = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.train_years,
                              TD.DEFAULT_IN, TD.TARGETS, stats)
        va = TD.DownscaleData(args.era5_dir, args.daymet_dir, args.val_years,
                              TD.DEFAULT_IN, TD.TARGETS, stats)
    else:
        tr, va = data
    H, W = tr.H, tr.W

    net = build_model(vars(args), (H, W)).to(device)
    # broadcast_buffers=False: 本模型的 buffer 要么恒定(位置编码), 要么由构造保证全
    # rank 一致(路由偏置从全局 all-reduce 计数更新); 默认的逐前向广播只会掩盖潜在的
    # 不一致而非修复它。find_unused_parameters 仅 MoE 需要(专家非每步全命中)。
    model = (DDP(net, device_ids=[local], find_unused_parameters=args.moe,
                 broadcast_buffers=False) if is_dist else net)
    moe_layers = net.moe_layers()

    global_batch = world * args.batch
    lr = args.lr if args.lr > 0 else args.blr * global_batch / 256
    if is_main:
        pc = net.param_counts()
        gh, gw = net.x_embedder.gh, net.x_embedder.gw
        print(f"[jit] target={args.target} 参数 {pc['total']:,} "
              f"(激活 {pc['activated']:,} / 路由专家 {pc['routed_experts']:,}) "
              f"token {gh}x{gw}={gh*gw} world={world} batch/rank={args.batch}", flush=True)
        print(f"[jit] lr={lr:.2e} (blr={args.blr:.1e} x {global_batch}/256) "
              f"noise_scale={args.noise_scale} P_mean={args.p_mean} moe={args.moe} "
              f"duration={args.duration:,} samples", flush=True)

    per_step = global_batch
    steps_total = args.duration // per_step
    ck = None
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    done_steps = (ck["samples"] // per_step) if ck else 0
    if done_steps >= steps_total:
        raise SystemExit(f"断点已达 {ck['samples']:,} samples >= duration {args.duration:,}")

    ds = JitFrameStream(tr, args.target, (steps_total - done_steps) * args.batch, 1234,
                        (rank * steps_total + done_steps) * args.batch)
    vs = JitFrameStream(va, args.target, args.val_steps * args.batch, 987,
                        rank * args.val_steps * args.batch, deterministic=True)
    # prefetch_factor=1: 7 个 worker 仍有 7 个整批在飞, 流水不断; 默认值 2 会让在飞
    # 缓冲翻倍(整幅批很大), 无谓抬高本就紧张的任务内存水位
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     pin_memory=True, drop_last=True,
                                     prefetch_factor=(1 if args.workers > 0 else None))
    vl = torch.utils.data.DataLoader(vs, batch_size=args.batch,
                                     num_workers=max(1, args.workers // 2),
                                     prefetch_factor=1)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)
    ema1, ema2 = _ema_init(net), _ema_init(net)
    if ck is not None:
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema1 = {n: v.to(device) for n, v in ck["ema1"].items()}
        ema2 = {n: v.to(device) for n, v in ck["ema2"].items()}
        if args.save_rng:
            saved = ck.get("rng")
            if saved and len(saved) == world:
                st = saved[rank]
                gen.set_state(st["gen"])
                torch.set_rng_state(st["torch_cpu"])
                if "torch_cuda" in st and torch.cuda.is_available():
                    torch.cuda.set_rng_state(st["torch_cuda"])
                if is_main:
                    print("[jit] RNG 状态已按 rank 恢复, 随机流与断点前连续", flush=True)
            elif is_main:
                print("[jit] 断点无匹配 RNG 状态(缺失或 world 不同), 使用新随机流", flush=True)
        if is_main:
            print(f"[jit] 从 {args.resume} 续训: 已完成 {ck['samples']:,} samples "
                  f"({done_steps:,} 步), 剩余 {steps_total - done_steps:,} 步", flush=True)

    def run_batch(cond, tgt, land, g):
        cond = cond.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        land = land.to(device, non_blocking=True)
        tgt = tgt * land                          # 海洋无监督, 目标定义为 0
        w = land if args.loss_scope == "land" else torch.ones_like(land)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device != "cpu")):
            return jit_vloss(model, tgt, cond, w, args.noise_scale,
                             args.p_mean, args.p_std, args.t_eps, generator=g)

    seen, t0, hist = done_steps * per_step, time.time(), []
    best = float("inf")
    hf = out / "loss_history.json"
    if ck is not None and hf.exists():
        hist = [h for h in json.loads(hf.read_text()) if h["samples"] <= seen]
        if hist:
            best = min(h["val"] for h in hist)
    # 断点字典用完必须释放: 它每 rank 常驻约 payload 大小的宿主内存, 且会被随后 fork 的
    # DataLoader worker 整体继承; 任务级内存本就贴近 cgroup 上限, 多驻留会零星触发内核
    # OOM 击杀单个 rank(表现为无任何应用报错的整作业 Force Terminated)
    del ck
    gc.collect()
    run_sum = run_n = 0.0
    gn_sum, gn_max = 0.0, 0.0
    load_mon = None
    net.train()
    for step, (cond, tgt, land) in enumerate(dl, 1):
        cur_lr = lr * min(1.0, (seen + per_step) / max(1, args.warmup_samples))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        loss = run_batch(cond, tgt, land, gen)
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise SystemExit(f"[jit] rank{rank} 损失非有限 ({lv}) @ {seen:,} samples")
        opt.zero_grad()
        loss.backward()
        if step == 1 and is_dist:
            _assert_ranks_synced(_fingerprint(net, grads=True), device, world,
                                 "首步梯度")
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
        opt.step()
        _ema_update(ema1, net, args.ema1)
        _ema_update(ema2, net, args.ema2)
        if step == 1 and is_dist:
            _assert_ranks_synced(_fingerprint(net), device, world, "首步后参数/buffer")
            if is_main:
                print("[jit] 首步自检通过: 梯度与参数全 rank 一致", flush=True)
        if moe_layers and args.bias_gamma > 0:
            counts = _drain_moe_load(moe_layers, is_dist)
            for m, c in zip(moe_layers, counts):
                m.update_bias(args.bias_gamma, c)
            load_mon = counts if load_mon is None else load_mon + counts
        seen += per_step
        run_sum += lv; run_n += 1.0
        gn_sum += gn; gn_max = max(gn_max, gn)

        if seen % args.val_every < per_step:
            # 验证去掉两重采样噪声: 帧确定, (t, 噪声)用逐批固定种子的临时流,
            # 训练专属流不被触碰
            net.eval(); vt = vn = 0.0
            with torch.no_grad():
                for k, (c2, t2, l2) in enumerate(vl):
                    vg = torch.Generator(device=device); vg.manual_seed(4242 + k)
                    vt += float(run_batch(c2, t2, l2, vg)); vn += 1
            net.train()
            v, tr_avg = vt / max(vn, 1), run_sum / max(run_n, 1)
            if is_dist:
                q = torch.tensor([v, tr_avg, 1.0], device=device); dist.all_reduce(q)
                v, tr_avg = float(q[0] / q[2]), float(q[1] / q[2])
            rec = {"samples": seen, "train": round(tr_avg, 6), "val": round(v, 6),
                   "lr": cur_lr, "gnorm_mean": round(gn_sum / max(run_n, 1), 4),
                   "gnorm_max": round(gn_max, 4), "seconds": round(time.time() - t0, 1)}
            if moe_layers:
                if args.bias_gamma <= 0:
                    c = _drain_moe_load(moe_layers, is_dist)
                    load_mon = c if load_mon is None else load_mon + c
                shares = load_mon / load_mon.sum(dim=1, keepdim=True).clamp_min(1.0)
                rec["moe_load"] = [[round(float(x), 4) for x in row] for row in shares]
                load_mon = None
            hist.append(rec)
            run_sum = run_n = 0.0
            gn_sum, gn_max = 0.0, 0.0
            if is_main:
                print(f"[jit] {seen:>9,} samples  train={tr_avg:.5f}  val={v:.5f}  "
                      f"lr={cur_lr:.2e}  {time.time()-t0:.0f}s", flush=True)
                if v < best:
                    best = v
                    TD._atomic_torch_save(
                        {"model": net.state_dict(), "opt": opt.state_dict(),
                         "ema1": ema1, "ema2": ema2, "samples": seen, "val": v,
                         "args": vars(args)}, out / "ckpt.pt")
                tmp = out / f"loss_history.json.tmp.{os.getpid()}"
                tmp.write_text(json.dumps(hist, indent=1))
                os.replace(tmp, out / "loss_history.json")

        do_ckpt = seen % args.ckpt_every < per_step
        rng_states = None
        if args.save_rng and do_ckpt:
            rng_states = _rng_gather(is_dist, world, _rng_capture(rank, gen))
        if is_main and do_ckpt:
            payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                       "ema1": ema1, "ema2": ema2, "samples": seen, "args": vars(args)}
            if rng_states is not None:
                payload["rng"] = rng_states
            TD._atomic_torch_save(payload, out / "last.pt")
        if is_main and args.snap_every > 0 and seen % args.snap_every < per_step:
            TD._atomic_torch_save({"model": net.state_dict(), "ema1": ema1,
                                   "samples": seen}, out / f"snap_{seen:09d}.pt")
        if seen >= args.duration:
            break
        if args.max_seconds > 0:
            timeup = torch.tensor(
                [1.0 if (rank == 0 and time.time() - t0 >= args.max_seconds) else 0.0],
                device=device)
            if is_dist:
                dist.broadcast(timeup, src=0)
            if timeup.item() > 0:
                if is_main:
                    print(f"[jit] 达到 --max-seconds={args.max_seconds:.0f}s, "
                          f"保存断点后退出 ({seen:,} samples)", flush=True)
                break

    rng_states = (_rng_gather(is_dist, world, _rng_capture(rank, gen))
                  if args.save_rng else None)
    if is_main:
        payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                   "ema1": ema1, "ema2": ema2, "samples": seen, "args": vars(args)}
        if rng_states is not None:
            payload["rng"] = rng_states
        TD._atomic_torch_save(payload, out / "last.pt")
        print(f"[jit] 完成 {seen:,} samples, best val={best:.5f}, "
              f"{time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
