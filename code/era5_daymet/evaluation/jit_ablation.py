#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
jit_ablation.py — JiT 整幅扩散的通道敏感性实验, 逐像素逐月落盘
============================================================================
按机理分组置换条件通道后重新采样, 看逐像素 CRPS 怎么变。ΔCRPS = crps_month(组)
− crps_month(none), 逐像素逐月, 任何按分区/季节的聚合都是它的函数, 与本脚本解耦。

JiT 是单阶段模型, 没有确定性的均值分支可借, 所以每个分组每天都要重采一遍完整集合。
让这件事仍然可行的是**配对种子**: 成员种子只依赖 (seed, 年, 日, 成员), 与是否置换无关,
因此同一天同一成员在各组之间的噪声逐位相同, 差值里几乎不含采样噪声。none 组走与消融
组完全相同的采样路径但不改任何通道, 是严格配对的基线 —— 不能拿正式 dump 当基线。

一天只读一次数据、复用同一份条件张量跑完所有分组: 数据层每天读一次的开销被分组数摊薄。

落盘 <out>/crps_<组>.npz:
    crps_month      (12,H,W) float32  该组逐月的逐像素 CRPS 均值(陆地外 NaN)
    crps_log_month  (12,H,W) float32  同上, log1p(mm) 空间(仅降水)
    n_days          (12,)    int32    各月天数
基线组固定叫 none。多卡时各 rank 先写 parts/ 分片(存原始和), 由 --finalize 合并。

采样权重取 raw(--ema 0), 与正式 dump 同一档: EMA 窗口对这批短训练过长, 既掉质量又带
16px 网格。
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

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.evaluation import ablation_groups as AB
from era5_daymet.evaluation.stageb_dump import apply_ablation, days_in_months
from era5_daymet.models.jit_sampler import generate
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.train_jit import build_model


def main():
    ap = argparse.ArgumentParser(description="JiT 通道敏感性(逐像素逐月 ΔCRPS)")
    ap.add_argument("--run", required=True, help="JiT 训练 run 目录, 取其 checkpoint")
    ap.add_argument("--which", choices=["ckpt", "last"], default="ckpt")
    ap.add_argument("--ema", type=int, choices=[0, 1, 2], default=0)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--months", type=int, nargs="+", default=None,
                    help="只算这些月份; 缺省全年。采样始终全域整幅, 不裁区域")
    ap.add_argument("--groups", nargs="+", default=None,
                    help=f"缺省 = none + 全部八组 + 三个复合; 可选 {sorted(AB.GROUPS)} "
                         f"{sorted(AB.COMPOSITES)}")
    ap.add_argument("--ablate-mode", choices=["zero", "doy"], default="zero")
    ap.add_argument("--doy-year", type=int, default=2019)
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程把各 rank 的分片合并成 crps_<组>.npz; 不采样")
    ap.add_argument("--max-days", type=int, default=0, help=">0 时只取前 N 天, 冒烟用")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups = a.groups or ([AB.NONE] + list(AB.GROUPS) + list(AB.COMPOSITES))
    if AB.NONE not in groups:
        groups = [AB.NONE] + groups          # 基线必须有, 否则 ΔCRPS 无从谈起
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(a.run) / f"{a.which}.pt"

    if a.finalize:
        finalize(out, groups)
        return

    # 设备要在载入模型之前按 rank 定好, 否则权重会留在 cuda:0
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local % torch.cuda.device_count())
        device = f"cuda:{local % torch.cuda.device_count()}"
    else:
        device = "cpu"

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    target = args["target"]
    ti = TD.TARGETS.index(target)
    is_precip = target == TD.PRECIP

    stats = TD.Stats(args["stats_dir"], TD.DEFAULT_IN, TD.TARGETS)
    dd = TD.DownscaleData(args["era5_dir"], args["daymet_dir"], list(a.years),
                          TD.DEFAULT_IN, TD.TARGETS, stats)
    H, W = dd.H, dd.W
    net = build_model(args, (H, W))
    sd = dict(ck["model"])
    if a.ema:
        for n, v in ck[f"ema{a.ema}"].items():
            sd[n] = v
    net.load_state_dict(sd)
    net = net.to(device).eval()

    donor = None
    if a.ablate_mode == "doy":
        if a.doy_year in a.years:
            raise ValueError("--doy-year 不得与 --years 重合")
        donor = TD.DownscaleData(args["era5_dir"], args["daymet_dir"], [a.doy_year],
                                 TD.DEFAULT_IN, TD.TARGETS, stats)

    days = days_in_months(a.years, a.months) if a.months else \
        [(y, t) for y in a.years for t in range(365)]
    if a.max_days:
        days = days[:a.max_days]
    all_days = days
    mine = days[rank::ntasks]
    slots = {g: AB.channel_slots(AB.resolve(g), TD.DEFAULT_IN) for g in groups}
    month_of = {y: np.array([x.month for x in M.daymet_dates(y)], int) for y in a.years}

    acc = {g: np.zeros((12, H, W), np.float32) for g in groups}
    acc_log = {g: np.zeros((12, H, W), np.float32) for g in groups} if is_precip else None
    cnt = np.zeros(12, np.int32)
    # rank 数可以多于天数(每 rank 一天时最后几个 rank 分不到), 海陆掩膜与天的分配无关,
    # 因此统一取全局第一天的, 空分片也能落出形状正确的零和
    land = dd.full(*all_days[0])[2][0] > 0.5
    print(f"[jitAbl] rank {rank}/{ntasks} target={target} 分到 {len(mine)} 天 "
          f"分组={len(groups)} 成员={a.members} 步数={a.steps} device={device}", flush=True)
    t0 = time.time()

    for k, (y, day) in enumerate(mine):
        cond, _tgt, mask, hr = dd.full(y, day)
        land = mask[0] > 0.5
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        truth_log = np.log1p(np.maximum(truth, 0.0)) if is_precip else None
        dn = donor.full(a.doy_year, day)[0] if donor is not None else None
        mo = int(month_of[y][day]) - 1
        cnt[mo] += 1

        for g in groups:
            cb = apply_ablation(cond, slots[g], a.ablate_mode, dn) if slots[g] else cond
            cond_t = torch.from_numpy(cb[None]).float().to(device)
            mem = []
            for m in range(a.members):
                gen = torch.Generator(device=device)
                # 种子与分组无关 -> 各组噪声逐位相同, 差值里不含采样噪声
                gen.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=(device != "cpu")):
                    s = generate(net, cond_t, steps=a.steps, noise_scale=args["noise_scale"],
                                 t_eps=args["t_eps"], land=land_t, generator=gen)
                mem.append((s[0, 0].float() * land_t[0, 0]).cpu().numpy())
            mem = np.stack(mem, 0)

            if is_precip and stats.precip_log:
                mem_phys = TD.precip_inv(mem * stats.d_std[ti] + stats.d_mean[ti],
                                         stats.precip_scale) * stats.precip_scale
                mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)
            else:
                mem_phys = mem * stats.d_std[ti] + stats.d_mean[ti]

            _, crps_px = TD.crps_ensemble(mem_phys[:, None], truth[None], land, per_pixel=True)
            acc[g][mo] += np.where(land, crps_px[0], 0.0).astype(np.float32)
            if is_precip:
                mem_log = np.log1p(np.maximum(mem_phys, 0.0))
                _, crps_lp = TD.crps_ensemble(mem_log[:, None], truth_log[None], land,
                                              per_pixel=True)
                acc_log[g][mo] += np.where(land, crps_lp[0], 0.0).astype(np.float32)

        el = time.time() - t0
        print(f"[jitAbl] {k+1}/{len(mine)} 天完成 {el:.0f}s "
              f"预计总计 {el/(k+1)*len(mine):.0f}s", flush=True)

    nanmask = ~land
    if ntasks > 1:                            # 分片: 存原始和, 由 --finalize 合并
        (out / "parts").mkdir(exist_ok=True)
        for g in groups:
            d = {"sum_month": acc[g], "n_days": cnt.astype(np.int32)}
            if is_precip:
                d["sum_log_month"] = acc_log[g]
            np.savez_compressed(out / "parts" / f"crps_{g}_rank{rank}.npz", **d)
        if rank == 0:
            np.save(out / "sea_mask.npy", nanmask)
            write_meta(out, a, args, ckpt_path, ck.get("samples"), groups, all_days, target)
    else:
        for g in groups:
            d = {"crps_month": reduce_month(acc[g], cnt, nanmask),
                 "n_days": cnt.astype(np.int32)}
            if is_precip:
                d["crps_log_month"] = reduce_month(acc_log[g], cnt, nanmask)
            np.savez_compressed(out / f"crps_{g}.npz", **d)
        write_meta(out, a, args, ckpt_path, ck.get("samples"), groups, all_days, target)
    print(f"[jitAbl] rank {rank} 完成 {len(mine)} 天 {time.time()-t0:.0f}s", flush=True)


def reduce_month(sum_month, cnt, nanmask):
    """逐月求和 -> 逐月均值; 陆地外置 NaN。"""
    m = sum_month / np.maximum(cnt, 1)[:, None, None]
    m[:, nanmask] = np.nan
    return m.astype(np.float32)


def finalize(out, groups):
    """把各 rank 的分片按组累加, 落成与单机路径一致的 crps_<组>.npz。"""
    parts = sorted((out / "parts").glob("crps_*_rank*.npz"))
    assert parts, f"没有分片可合并: {out/'parts'}"
    acc, acc_log, cnt = {}, {}, None
    for f in parts:
        g = f.stem[len("crps_"):].rsplit("_rank", 1)[0]
        z = np.load(f)
        acc[g] = acc.get(g, 0.0) + z["sum_month"].astype(np.float64)
        if "sum_log_month" in z.files:
            acc_log[g] = acc_log.get(g, 0.0) + z["sum_log_month"].astype(np.float64)
        if g == AB.NONE:
            cnt = (cnt if cnt is not None else 0) + z["n_days"].astype(np.int64)
    sea = np.load(out / "sea_mask.npy") if (out / "sea_mask.npy").exists() else None
    nanmask = sea if sea is not None else np.zeros(next(iter(acc.values())).shape[1:], bool)
    for g, v in acc.items():
        d = {"crps_month": reduce_month(v, cnt, nanmask), "n_days": cnt.astype(np.int32)}
        if g in acc_log:
            d["crps_log_month"] = reduce_month(acc_log[g], cnt, nanmask)
        np.savez_compressed(out / f"crps_{g}.npz", **d)
    print(f"[jitAbl] finalize: {len(acc)} 组, 天数/月 {cnt.tolist()} -> {out}")


def write_meta(out, a, args, ckpt_path, samples, groups, days, target):
    p = Path(ckpt_path)
    meta = {"id": out.name, "date": datetime.date.today().isoformat(),
            "kind": "jit_channel_ablation", "target": target,
            "unit": "mm/day" if target == TD.PRECIP else "K",
            "spaces": ["phys", "log1p(mm)"] if target == TD.PRECIP else ["phys"],
            "ckpt": {"path": str(p), "bytes": p.stat().st_size, "trained_samples": samples},
            "weight": "raw" if a.ema == 0 else f"ema{a.ema}",
            "members": a.members, "steps": a.steps, "seed": a.seed,
            "noise_scale": args["noise_scale"], "t_eps": args["t_eps"],
            "years": list(a.years), "months": list(a.months) if a.months else None,
            "n_days": len(days),
            "groups": {g: AB.describe(g) for g in groups},
            "ablate_mode": a.ablate_mode,
            "doy_year": a.doy_year if a.ablate_mode == "doy" else None,
            "baseline": AB.NONE,
            "seed_scheme": "seed*100003+(year*1000+day)*131+member; 与分组无关, 各组配对",
            "note": "ΔCRPS = crps_month(组) − crps_month(none); 配对种子, 差值几乎不含采样噪声"}
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
