#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stageb_dump.py — CorrDiff 阶段 B 采样落盘(逐日逐像素场), 不做指标与绘图
============================================================================
对给定年份的每一天, 用冻结的阶段 A 均值 μ(缓存)与阶段 B 残差扩散网采 N 个成员, 只把
"可由成员归约出的逐日逐像素场"落盘, 布局与 jit_dump 完全一致, dump_metrics 与 render 通吃:

  ens_mean/<年>_d<日>.npy   成员均值          float32  物理单位   (有效域外 NaN)
  crps/<年>_d<日>.npy       逐像素 CRPS       float32  物理单位   (有效域外 NaN)
  crps_log/<年>_d<日>.npy   逐像素 CRPS       float32  log1p(mm)  (仅降水)
  spread/<年>_d<日>.npy     成员标准差        float32  物理单位   (有效域外 NaN)
  rank/<年>_d<日>.npy       真值在成员中的名次 int8    0..members(有效域外 -1)

采样是 patched 的 EDM 二阶采样(GridPatching2D 网格切块 + 羽化融合), 条件拼接与训练损失
同一套实现: [μ, 本 patch 的 51 通道条件, 全域上下文按 cond_mode 的下标插值副本, 位置嵌入]。
overlap/boundary 是推理期融合几何, 训练时不存在, 必须显式给定并记进 meta。

三处输入身份核对, 不一致即拒绝: 阶段 B checkpoint 记录的输入目录、训练时绑定的 μ 缓存
与阶段 A checkpoint, 都与本次给定的一致; μ 缓存本身再按阶段 A 的 SHA 校验。这些错配都
不改变任何形状, 采样照样出场, 只是全部错标。

种子 = seed*100003 + (年*1000+日)*131 + 成员, 只依赖 (seed, 年, 日, 成员); 场文件与分片
无关。多卡: 按天分给 SLURM rank, rank0 写 meta.json; --finalize 校验文件数并置 status。
============================================================================
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.data.mu_cache import MuCache
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation.jit_dump import (fields_for, file_sig, finalize, resolve_days,
                                                save_field)
from downscaling_4x.models.patching import GridPatching2D
from downscaling_4x.models.preconditioning import EDMPrecondSuperResolution
from downscaling_4x.models.stochastic_sampler import stochastic_sampler
from downscaling_4x.training.stage_b_mean import pin_ocean, stage_b_cond_channels

# 推理期融合几何的缺省: 与 6x 线锁定的配置相同(patch 192 下 overlap 96 / boundary 8)。
# 它不是训练参数, 换个值照样出场; 因此一律写进 meta, 跨 run 比较前先核对。
DEFAULT_OVERLAP, DEFAULT_BOUNDARY = 96, 8


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    args = args if isinstance(args, dict) else vars(args)
    return ck, args


def build_net(args, H, W, device, amp=False):
    """按阶段 B checkpoint 自述重建预条件网; 结构参数对不上会在 load_state_dict 报错,
    sigma_data 不改变形状, 所以必须从 checkpoint 取而不是命令行。amp 打开时网络内部
    允许 autocast(SongUNet 会校验 amp_mode 与 autocast 状态一致)。"""
    mode = args.get("mode", C.DEFAULT_MODE)
    n_grid = int(args.get("n_grid_channels", 100))
    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_out_channels=1,
        img_in_channels=stage_b_cond_channels(mode, n_grid=n_grid),
        model_type="SongUNetPosEmbd",
        model_channels=int(args.get("model_channels", 64)),
        channel_mult=list(args.get("channel_mult", [1, 2, 2])), attn_resolutions=[16],
        N_grid_channels=n_grid, gridtype="learnable",
        sigma_data=float(args["sigma_data"]), amp_mode=bool(amp))
    return net.to(device).eval()


def write_meta(out, a, args, ckpt_path, ck, days, patching):
    target = args["target"]
    meta = {
        "id": Path(out).name,
        "date": datetime.date.today().isoformat(),
        "kind": "stageb_sample_dump",
        "target": target,
        "unit": "mm/day" if target == C.PRECIP else "K",
        "mode": args.get("mode", C.DEFAULT_MODE),
        "diffusion_ckpt": file_sig(ckpt_path),
        "run": str(a.run),
        "which": a.which,
        "trained_samples": int(ck.get("samples", -1)),
        "stage_a_ckpt": file_sig(a.stage_a_ckpt),
        "mu_cache": str(a.cache),
        "input": II.describe(a.era5_dir),
        "members": a.members,
        "steps": a.steps,
        "sigma": {"min": a.sigma_min, "max": a.sigma_max, "rho": a.rho,
                  "data": float(args["sigma_data"])},
        "config": {"patch": a.patch, "train_patch": int(args.get("patch", 0)) or None,
                   "overlap": a.overlap, "boundary": a.boundary,
                   "patches_per_frame": int(patching.patch_num),
                   "model_channels": args.get("model_channels"), "channel_mult": args.get("channel_mult"),
                   "n_grid_channels": args.get("n_grid_channels"), "amp": bool(a.amp)},
        "recon": "member = pin_ocean(μ) + r̂ (归一化空间); 降水 expm1 后 <clip 置零",
        "seed": a.seed,
        "years": a.years,
        "all_days": bool(a.all_days),
        "n_days_expected": len(days),
        "fields": fields_for(target),
        "crps_log_transform": f"precip_fwd: <{C.PRECIP_CLIP_MM:g} mm 置零后 log1p (成员与真值同)",
        "mask_note": "有效域 = daymet_land AND era5_valid; 场按 (H,W) 落盘, 域外 NaN",
        "seed_scheme": "seed*100003+(year*1000+day)*131+member",
        "status": "running",
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(Path(out) / "meta.json", "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="CorrDiff 阶段 B 采样落盘(逐日逐像素场, 不出图)")
    ap.add_argument("--run", required=True, help="阶段 B 训练 run 目录, 取其 checkpoint")
    ap.add_argument("--which", choices=["ckpt", "last"], default="ckpt",
                    help="ckpt=best-val(默认) / last=末端")
    ap.add_argument("--cache", default=None, help="μ 缓存目录; 缺省取 checkpoint 记录的那份")
    ap.add_argument("--stage-a-ckpt", default=None, help="阶段 A checkpoint; 缺省取 checkpoint 记录的那份")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--patch", type=int, default=None,
                    help="推理切块边长; 缺省取 checkpoint 记录的训练 patch, 显式指定即视为有意偏离并记进 meta")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="切块重叠像素(推理期融合几何)")
    ap.add_argument("--boundary", type=int, default=DEFAULT_BOUNDARY, help="切块边界像素(推理期融合几何)")
    ap.add_argument("--amp", action="store_true", help="采样前向用 bf16 autocast(缺省 fp32)")
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--all-days", action="store_true",
                    help="取 years 内全部日(365 日历/年); 缺省用每月 5/15/25 抽样")
    ap.add_argument("--days", nargs="+", default=None,
                    help='指定 "年-日序" 列表(如 2020-0 2020-100), 覆盖 years/all-days')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help=">0 时按已跑天的最长耗时预估, 剩余预算装不下一整天就停止开新天(接力段补缺)")
    ap.add_argument("--overwrite", action="store_true",
                    help="重采已落盘的天; 缺省跳过各场文件齐全的天, 使分段接力只补缺")
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程校验各场文件数并标注 status; 不采样")
    ap.add_argument("--out", required=True)
    II.add_arg(ap)
    a = ap.parse_args()

    out = Path(a.out)
    ckpt_path = Path(a.run) / f"{a.which}.pt"
    days = resolve_days(a)
    allow = getattr(a, II.ALLOW_DEST, False)

    if a.finalize:
        mp = out / "meta.json"
        target = json.load(open(mp))["target"] if mp.exists() else load_ckpt(ckpt_path)[1]["target"]
        finalize(out, target, days)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    ck, args = load_ckpt(ckpt_path)
    target = args["target"]
    ti = C.TARGETS.index(target)
    is_precip = target == C.PRECIP
    mode = args.get("mode", C.DEFAULT_MODE)

    # ---- 输入身份: checkpoint 记录的输入目录、μ 缓存与阶段 A 都必须与本次一致 ----
    II.check(args.get("era5_dir"), a.era5_dir, f"阶段 B checkpoint {ckpt_path}", allow=allow)
    a.cache = a.cache or args.get("cache")
    a.stage_a_ckpt = a.stage_a_ckpt or args.get("stage_a_ckpt")
    if not a.cache or not a.stage_a_ckpt:
        raise SystemExit("checkpoint 未记录 μ 缓存/阶段 A, 必须显式给 --cache 与 --stage-a-ckpt")
    II.check(args.get("cache"), a.cache, f"阶段 B 训练时绑定的 μ 缓存 (checkpoint {ckpt_path})", allow=allow)
    II.check(args.get("stage_a_ckpt"), a.stage_a_ckpt, f"阶段 B 训练时绑定的阶段 A (checkpoint {ckpt_path})", allow=allow)
    cache = MuCache(a.cache, [target])
    cache.verify({target: a.stage_a_ckpt})
    II.check(cache.manifest.get("era5_dir"), a.era5_dir, f"μ 缓存 {a.cache}", allow=allow)
    if (cache.manifest.get("mode") or C.DEFAULT_MODE) != mode:
        raise SystemExit(f"μ 缓存按 --mode {cache.manifest.get('mode')} 建, 与 checkpoint 的 {mode} 不符")

    # ---- 推理切块几何: patch 缺省取训练值; 偏离是合法的几何扫描, 但要回显并记进 meta ----
    train_patch = int(args.get("patch", 0)) or None
    if a.patch is None:
        if train_patch is None:
            raise SystemExit(f"{ckpt_path} 未记录训练 patch, 必须显式给 --patch")
        a.patch = train_patch
    elif train_patch is not None and a.patch != train_patch and rank == 0:
        print(f"[stageb_dump] ★推理 patch {a.patch} != 训练 patch {train_patch}: 有意偏离训练几何", flush=True)

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    years_all = sorted({y for y, _ in days})
    avail = sorted({y - 1 for y in years_all} | set(years_all)) if need else years_all
    fi = FrameIndex(years_all, avail, need, lags, split="dump")
    frame_of = {f: k for k, f in enumerate(fi.frames)}
    skipped = [d for d in days if d not in frame_of]
    if skipped:
        print(f"[stageb_dump] {len(skipped)} 天因缺完整历史被跳过: {skipped[:5]}...", flush=True)
        days = [d for d in days if d in frame_of]
    ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
    dd = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode,
                       era5_cache_years=len(ds_years))
    H, W = dd.H, dd.W
    land = dd.mask

    net = build_net(args, H, W, device, amp=a.amp)
    net.load_state_dict(ck["model"])
    for p in net.parameters():
        p.requires_grad_(False)
    patching = GridPatching2D(img_shape=(H, W), patch_shape=(a.patch, a.patch),
                              overlap_pix=a.overlap, boundary_pix=a.boundary)

    out.mkdir(parents=True, exist_ok=True)
    mine = days[rank::ntasks]
    if not a.overwrite:
        # 分段接力: 各场文件齐全的天视为已完成; 种子只依赖 (seed, 年, 日, 成员), 重采与
        # 跳过在结果上等价, 跳过只是省时间
        done = [d for d in mine if all((out / f / f"{d[0]}_d{d[1]}.npy").exists() for f in fields_for(target))]
        if done:
            print(f"[stageb_dump] rank {rank}: {len(done)} 天已落盘, 跳过", flush=True)
            mine = [d for d in mine if d not in set(done)]
    if rank == 0 and not (out / "meta.json").exists():
        write_meta(out, a, args, ckpt_path, ck, days, patching)
    print(f"[stageb_dump] rank {rank}/{ntasks} device={device} target={target} members={a.members} "
          f"steps={a.steps} patch={a.patch} ov={a.overlap} bd={a.boundary} "
          f"({patching.patch_num} 块/帧) amp={a.amp} 分到 {len(mine)} 天; 场={fields_for(target)}", flush=True)

    land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
    lg = lambda x: MT.precip_log_mm(x, stats.precip_clip)
    t_start, day_secs = time.time(), 0.0
    for y, day in mine:
        if a.max_seconds > 0 and time.time() - t_start + day_secs > a.max_seconds:
            print(f"  [rank{rank}] 预算 {a.max_seconds:.0f}s 装不下再一天(单天 {day_secs:.0f}s), "
                  f"余 {len(mine) - mine.index((y, day))} 天留给接力段", flush=True)
            break
        t0 = time.time()
        cond, _tgt, _mask, hr = dd.full(y, day, fi.history_of(frame_of[(y, day)]))
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        mu_np = cache.get(target, y, day)
        if not np.isfinite(mu_np[land]).all():
            raise RuntimeError(f"μ 缓存 {y}-d{day} 在有效域内含非有限值")
        mu_t = pin_ocean(torch.from_numpy(mu_np[None, None]).float().to(device), land_t)
        mu_pinned = mu_t[0, 0].cpu().numpy()

        members_norm = []
        for m in range(a.members):
            torch.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
            lat = torch.randn(1, 1, H, W, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(a.amp and device != "cpu")):
                r = stochastic_sampler(
                    net=net, latents=lat, img_lr=cond_t, patching=patching, mean_hr=mu_t,
                    num_steps=a.steps, sigma_min=a.sigma_min, sigma_max=a.sigma_max,
                    rho=a.rho, S_churn=0, S_noise=1, cond_mode=mode)
            members_norm.append(mu_pinned + r[0, 0].float().cpu().numpy())
        members_norm = np.stack(members_norm, 0)                 # (M,H,W) 归一化空间

        if is_precip and stats.precip_log:
            mem_phys = C.precip_inv(members_norm * stats.d_std[ti] + stats.d_mean[ti],
                                    stats.precip_scale) * stats.precip_scale
            mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)   # 与预处理同口径
        else:
            mem_phys = members_norm * stats.d_std[ti] + stats.d_mean[ti]

        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)
        _, crps_px = MT.crps_ensemble(mem_phys[:, None], truth[None], land, per_pixel=True)
        crps_field = np.where(land, crps_px[0], np.nan).astype(np.float32)

        mem_l = mem_phys[:, land]
        tr_l = truth[land]
        ties = (mem_l == tr_l[None]).sum(0)
        tie_rng = np.random.default_rng(a.seed * 100003 + (y * 1000 + day) * 131 + 7)
        ranks_l = (mem_l < tr_l[None]).sum(0) + tie_rng.integers(0, ties + 1)
        rank_field = np.full((H, W), -1, np.int8)
        rank_field[land] = np.clip(ranks_l, 0, a.members).astype(np.int8)

        save_field(out, "ens_mean", y, day, np.where(land, ens_mean, np.nan).astype(np.float32))
        save_field(out, "spread", y, day, np.where(land, spread, np.nan).astype(np.float32))
        save_field(out, "crps", y, day, crps_field)
        save_field(out, "rank", y, day, rank_field)
        if is_precip:
            _, crps_log_px = MT.crps_ensemble(lg(mem_phys)[:, None], lg(truth)[None], land, per_pixel=True)
            save_field(out, "crps_log", y, day, np.where(land, crps_log_px[0], np.nan).astype(np.float32))
        day_secs = max(day_secs, time.time() - t0)
        print(f"  [rank{rank}] {y}-d{day} 落场完成 {time.time() - t0:.0f}s", flush=True)

    print(f"[stageb_dump] rank {rank} 完成 {len(mine)} 天", flush=True)
    if ntasks == 1:
        finalize(out, target, days)


if __name__ == "__main__":
    main()
