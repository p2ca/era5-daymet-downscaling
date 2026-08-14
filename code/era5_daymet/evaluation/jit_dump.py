#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jit_dump.py — JiT 整幅采样落盘(逐日逐像素场, 不做任何指标计算或绘图)

对给定年份的每一天, 用整幅 ODE 采样器采 N 个成员, 只把"可由成员归约出的逐日逐像素
场"落盘, 全部下游聚合与出图交给 evaluation.render 离线完成。落盘内容与布局与阶段 B 的
patched 扩散(stageb_dump)完全一致, 因此同一 render 管线直接通吃:

  ens_mean/<年>_d<日>.npy   成员均值          float32  物理单位   (陆地外 NaN)
  crps/<年>_d<日>.npy       逐像素 CRPS       float32  物理单位   (陆地外 NaN)
  crps_log/<年>_d<日>.npy   逐像素 CRPS       float32  log1p(mm)  (仅降水; 陆地外 NaN)
  spread/<年>_d<日>.npy     成员标准差        float32  物理单位   (陆地外 NaN)
  rank/<年>_d<日>.npy       真值在成员中的名次 int8    0..members(陆地外 -1)

CRPS 与 rank 依赖全体成员, 采样结束即无法重算, 因此必须在此处算好; ens_mean/spread 亦为
成员归约。其余按分区/月份/分层的聚合与全部图都是这些场的函数, 归 render 管线, 与本脚本解耦。

与 stageb_dump 的差别只在采样一侧: JiT 是整幅像素空间条件扩散, 无 Stage-A 均值、无 μ 缓存、
无 patch 融合; 网络结构按 checkpoint 内保存的 args 重建, 采样权重默认取 raw(--ema 0)。
反变换、降水 drizzle 截断、CRPS/rank 口径与 stageb_dump 逐项相同。

种子 = seed*100003 + (年*1000+日)*131 + 成员, 只依赖 (seed, 年, 日, 成员); 平局名次的随机
劈分种子只依赖 (seed, 年, 日)。故场文件与分片无关, 重跑/改分片逐位一致。

多卡: 按天分给 SLURM rank, 各写各的天文件(天互不相交, 无需合并); rank0 写 meta.json。
--finalize: srun 结束后单进程校验各场文件数并把 meta.status 置 done。
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from era5_daymet.models.jit_sampler import generate
from era5_daymet import contract as C
from era5_daymet.data import dataset as DS
from era5_daymet.evaluation import metrics as MT
from era5_daymet.training.train_jit import build_model

# 采样时始终落盘的场(降水追加 crps_log)
BASE_FIELDS = ("ens_mean", "crps", "spread", "rank")


def default_days(years, doms=(5, 15, 25)):
    """每月固定日 x 12 月 x 各年 -> [(year, 0基日序), ...]; 365 日历下均匀覆盖季节。"""
    out = []
    for y in years:
        for m in range(1, 13):
            for dom in doms:
                doy = (datetime.date(y, m, dom) - datetime.date(y, 1, 1)).days
                out.append((y, doy))
    return out


def planned_days(years, all_days):
    return [(y, t) for y in years for t in range(365)] if all_days else default_days(years)


def resolve_days(a):
    """显式 --days(年-日序 列表)优先于 years/all-days; 用于冒烟或补采个别日。"""
    if a.days:
        return [(int(s.split("-")[0]), int(s.split("-")[1])) for s in a.days]
    return planned_days(a.years, a.all_days)


def fields_for(target):
    return list(BASE_FIELDS) + (["crps_log"] if target == C.PRECIP else [])


def file_sig(path):
    """轻量溯源: 路径/大小/mtime; 不做 sha 以免拖慢作业启动。"""
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def write_meta(out, a, args, ckpt_path, samples, days):
    target = args["target"]
    meta = {
        "id": Path(out).name,
        "date": datetime.date.today().isoformat(),
        "kind": "jit_sample_dump",
        "target": target,
        "unit": "mm/day" if target == C.PRECIP else "K",
        "diffusion_ckpt": file_sig(ckpt_path),
        "run": str(a.run),
        "weight": "raw" if a.ema == 0 else f"ema{a.ema}",
        "trained_samples": int(samples) if samples is not None else -1,
        "members": a.members,
        "steps": a.steps,
        "noise_scale": args["noise_scale"],
        "t_eps": args["t_eps"],
        "patch": args["patch"],
        "seed": a.seed,
        "years": a.years,
        "all_days": bool(a.all_days),
        "n_days_expected": len(days),
        "fields": fields_for(target),
        "seed_scheme": "seed*100003+(year*1000+day)*131+member",
        "status": "running",
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(Path(out) / "meta.json", "w"), indent=1, ensure_ascii=False)


def finalize(out, target, days):
    """校验各场文件数, 标注缺失日, 把 status 置 done。纯文件系统, 无需 GPU/数据。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    want_names = {f"{y}_d{t}" for y, t in days}
    report, missing = {}, {}
    for fld in fields_for(target):
        have = {fp.stem for fp in (out / fld).glob("*.npy")}  # "<年>_d<日>"
        report[fld] = len(have)
        miss = sorted(want_names - have)
        if miss:
            missing[fld] = miss
    meta["written"] = report
    meta["missing"] = missing
    meta["status"] = "done" if not missing else "incomplete"
    json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
    tag = "OK" if not missing else f"缺 {sum(len(v) for v in missing.values())} 个场-日"
    print(f"[jit_dump] finalize {out.name}: {report} -> status={meta['status']} ({tag})")


def save_field(out, sub, y, day, arr):
    d = Path(out) / sub
    d.mkdir(exist_ok=True)
    np.save(d / f"{y}_d{day}.npy", arr)


def main():
    ap = argparse.ArgumentParser(description="JiT 整幅采样落盘(逐日逐像素场, 不出图)")
    ap.add_argument("--run", required=True, help="JiT 训练 run 目录, 取其 checkpoint")
    ap.add_argument("--which", choices=["ckpt", "last"], default="ckpt",
                    help="ckpt=best-val(默认) / last=末端")
    ap.add_argument("--ema", type=int, choices=[0, 1, 2], default=0,
                    help="采样权重: 0=raw(默认, 已定档) / 1,2=EMA")
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--all-days", action="store_true",
                    help="取 years 内全部日(365 日历/年); 缺省用每月 5/15/25 抽样")
    ap.add_argument("--days", nargs="+", default=None,
                    help='指定 "年-日序" 列表(如 2020-0 2020-100), 覆盖 years/all-days')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程校验各场文件数并标注 status; 不采样")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    ckpt_path = Path(a.run) / f"{a.which}.pt"
    days = resolve_days(a)

    if a.finalize:
        mp = out / "meta.json"
        if mp.exists():
            target = json.load(open(mp))["target"]
        else:
            target = torch.load(ckpt_path, map_location="cpu",
                                weights_only=False)["args"]["target"]
        finalize(out, target, days)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    target = args["target"]
    ti = C.TARGETS.index(target)
    is_precip = target == C.PRECIP

    stats = DS.Stats(args["stats_dir"], C.DEFAULT_IN, C.TARGETS)
    dd = DS.DownscaleData(args["era5_dir"], args["daymet_dir"], a.years,
                          C.DEFAULT_IN, C.TARGETS, stats)
    H, W = dd.H, dd.W

    net = build_model(args, (H, W))
    sd = dict(ck["model"])
    if a.ema:
        for n, v in ck[f"ema{a.ema}"].items():
            sd[n] = v                      # 参数用 EMA, buffer(含路由偏置)保持 model 值
    net.load_state_dict(sd)
    net = net.to(device).eval()

    out.mkdir(parents=True, exist_ok=True)
    mine = days[rank::ntasks]
    if rank == 0:
        write_meta(out, a, args, ckpt_path, ck.get("samples"), days)

    weight = "raw" if a.ema == 0 else f"ema{a.ema}"
    print(f"[jit_dump] rank {rank}/{ntasks} device={device} weight={weight} target={target} "
          f"members={a.members} steps={a.steps} 分到 {len(mine)} 天; 场={fields_for(target)}",
          flush=True)

    for y, day in mine:
        t0 = time.time()
        cond, _tgt, mask, hr = dd.full(y, day)
        land = mask[0] > 0.5
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)

        members_norm = []
        for m in range(a.members):
            g = torch.Generator(device=device)
            g.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device != "cpu")):
                s = generate(net, cond_t, steps=a.steps, noise_scale=args["noise_scale"],
                             t_eps=args["t_eps"], land=land_t, generator=g)
            members_norm.append((s[0, 0].float() * land_t[0, 0]).cpu().numpy())
        members_norm = np.stack(members_norm, 0)                    # (M,H,W) 归一化(log1p)空间

        if is_precip and stats.precip_log:
            mem_phys = C.precip_inv(members_norm * stats.d_std[ti] + stats.d_mean[ti],
                                     stats.precip_scale) * stats.precip_scale
            # 融合后 drizzle 截断, 与预处理同口径, 恢复零质量
            mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)
        else:
            mem_phys = members_norm * stats.d_std[ti] + stats.d_mean[ti]

        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)

        # CRPS(物理): 逐像素, 陆地外置 NaN
        _, crps_px = MT.crps_ensemble(mem_phys[:, None], truth[None], land, per_pixel=True)
        crps_field = np.where(land, crps_px[0], np.nan).astype(np.float32)

        # rank: 真值在成员中的名次, 平局按标准做法均匀劈分(种子只依赖 seed/年/日)
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
            # log1p(mm) 空间 CRPS: 与功率谱一致取物理量的 log1p(max(.,0))
            mem_log = np.log1p(np.maximum(mem_phys, 0.0))
            truth_log = np.log1p(np.maximum(truth, 0.0))
            _, crps_log_px = MT.crps_ensemble(mem_log[:, None], truth_log[None], land, per_pixel=True)
            save_field(out, "crps_log", y, day,
                       np.where(land, crps_log_px[0], np.nan).astype(np.float32))

        print(f"  [rank{rank}] {y}-d{day} 落场完成 {time.time() - t0:.0f}s", flush=True)

    print(f"[jit_dump] rank {rank} 完成 {len(mine)} 天", flush=True)
    if ntasks == 1:
        finalize(out, target, days)


if __name__ == "__main__":
    main()
