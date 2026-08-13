#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stageb_dump.py — 阶段 B 采样落盘(逐日逐像素场), 不做任何指标计算或绘图

对给定年份的每一天, 以锁定的融合配置采样 N 个成员, 只把"可由成员归约出的逐日逐像素
场"落盘, 全部下游聚合与出图交给 evaluation.render 离线完成。落盘内容:

  ens_mean/<年>_d<日>.npy   成员均值          float32  物理单位   (陆地外 NaN)
  crps/<年>_d<日>.npy       逐像素 CRPS       float32  物理单位   (陆地外 NaN)
  crps_log/<年>_d<日>.npy   逐像素 CRPS       float32  log1p(mm)  (仅降水; 陆地外 NaN)
  spread/<年>_d<日>.npy     成员标准差        float32  物理单位   (陆地外 NaN)
  rank/<年>_d<日>.npy       真值在成员中的名次 int8    0..members(陆地外 -1)

成员本体是采样瞬态, 不落盘。CRPS 与 rank 依赖全体成员, 采样结束即无法重算, 因此必须在
此处算好; ens_mean/spread 亦为成员归约。其余按分区/月份/分层的聚合与全部图都是这些场的
函数, 归 render 管线, 与本脚本解耦。

降水的 CRPS 同时存物理(mm/day)与 log1p(mm)两套: 两个单位空间的 CRPS 排名不可换算,
且采样后不可重算; rank 对单调变换不变, 降水两空间同一张, 只存一次。

种子 = seed*100003 + (年*1000+日)*131 + 成员, 只依赖 (seed, 年, 日, 成员), 与分片和
配置无关, 重跑/改分片逐位一致。平局名次的随机劈分种子只依赖 (seed, 年, 日), 故名次场
也与分片无关。

多卡: 按天分给 SLURM rank, 各写各的天文件(天互不相交, 无需合并); rank0 写 meta.json。
--finalize: srun 结束后单进程校验各场文件数并把 meta.status 置 done。

--ablate 打开通道敏感性实验: 按 ablation_groups 的机理分组置换条件通道后重采样, 落盘
内容与正式 dump 完全一致, 因此下游聚合与出图无需任何改动, ΔCRPS 直接由两次 dump 相减。
两点保证配对可比:
  * 种子只依赖 (seed, 年, 日, 成员), 与是否置换无关 -> 成员噪声在两次运行间逐位相同,
    差值里几乎不含采样噪声, 这是能用小成员数做对比的前提。
  * 置换会同时改变阶段 A 的 μ, 而 μ 缓存是按未置换的输入算的; 因此一旦 --ablate 生效,
    μ 一律由阶段 A 网络在置换后的条件上现算, 不再读缓存。--ablate none 是对照组: 走
    完全相同的现算路径但不改任何通道, 与消融组构成严格配对。
采样始终是全域整幅, 与选定的月份无关; 区域性结论由下游按分区聚合得出, 不在采样时裁剪。
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
from era5_daymet.data.mu_cache import MuCache
from era5_daymet.evaluation import ablation_groups as AB
from era5_daymet.models.patching import GridPatching2D
from era5_daymet.models.preconditioning import EDMPrecondSuperResolution
from era5_daymet.models.stochastic_sampler import stochastic_sampler
from era5_daymet.training import train_downscale as TD
from era5_daymet.training.stage_b_mean import pin_ocean

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


def days_in_months(years, months):
    """years 内落在指定月份的全部日; 月份取自数据层的 365 日历, 与场文件的日索引一致。"""
    want = set(int(m) for m in months)
    return [(y, t) for y in years
            for t, d in enumerate(M.daymet_dates(y)) if d.month in want]


def resolve_days(a):
    """显式 --days 优先于 --months, --months 优先于 years/all-days。"""
    if a.days:
        return [(int(s.split("-")[0]), int(s.split("-")[1])) for s in a.days]
    if getattr(a, "months", None):
        return days_in_months(a.years, a.months)
    return planned_days(a.years, a.all_days)


def apply_ablation(cond, slots, mode, donor=None):
    """按 slots 置换条件通道, 返回新数组(不改入参)。

    slots 的每一项是 (通道下标, 置换方式), 由 ablation_groups.channel_slots 给出。
    mode="doy" 且该通道允许时取 donor(另一年同一 day-of-year 的条件)对应通道, 否则填 0;
    land_sea_mask 固定填 1。
    """
    out = cond.copy()
    for idx, fill in slots:
        if fill == AB.FILL_ONE:
            out[idx] = 1.0
        elif fill == AB.FILL_DOY and mode == "doy" and donor is not None:
            out[idx] = donor[idx]
        else:
            out[idx] = 0.0
    return out


def load_stage_a(ckpt_path, device):
    """载入单目标阶段 A 回归网络(冻结、推理态), 用于在置换后的条件上现算 μ。"""
    from era5_daymet.tools.plotting import plot_unet_annual_maps as PU
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck = payload.get("args", {})
    cin = TD.cond_channels(list(ck.get("in_vars", TD.DEFAULT_IN)), TD.TARGETS, False)
    net = PU._build_model(ck, cin, 1, device)
    net.load_state_dict(payload["model"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def fields_for(target, save_mu=False):
    return (list(BASE_FIELDS) + (["crps_log"] if target == TD.PRECIP else [])
            + (["mu"] if save_mu else []))


def file_sig(path):
    """轻量溯源: 路径/大小/mtime; 不做 1GB 级 sha 以免拖慢作业启动。"""
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def write_meta(out, a, days):
    meta = {
        "id": Path(out).name,
        "date": datetime.date.today().isoformat(),
        "kind": "stageb_sample_dump",
        "target": a.target,
        "unit": "mm/day" if a.target == TD.PRECIP else "K",
        "diffusion_ckpt": file_sig(a.diffusion_ckpt),
        "stage_a_ckpt": file_sig(a.stage_a_ckpt),
        "mu_cache": a.cache,
        "config": {"patch": a.patch, "overlap": a.overlap, "boundary": a.boundary},
        "members": a.members,
        "steps": a.steps,
        "sigma": {"min": a.sigma_min, "max": a.sigma_max, "data": a.sigma_data},
        "seed": a.seed,
        "years": a.years,
        "all_days": bool(a.all_days),
        "months": list(a.months) if getattr(a, "months", None) else None,
        "ablation": (dict(AB.describe(a.ablate), mode=a.ablate_mode,
                          donor_year=(a.doy_year if a.ablate_mode == "doy" else None))
                     if a.ablate else None),
        "mu_source": ("阶段 A 在置换后的条件上现算" if a.ablate else "μ 缓存"),
        "n_days_expected": len(days),
        "fields": fields_for(a.target, a.save_mu),
        "grid_note": "H,W 由数据层给定; 场按 (H,W) 落盘",
        "seed_scheme": "seed*100003+(year*1000+day)*131+member",
        "status": "running",
    }
    json.dump(meta, open(Path(out) / "meta.json", "w"), indent=1, ensure_ascii=False)


def finalize(out, a, days):
    """校验各场文件数, 标注缺失日, 把 status 置 done。纯文件系统, 无需 GPU/数据。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    want = {(y, t) for y, t in days}
    report, missing = {}, {}
    for fld in fields_for(a.target, a.save_mu):
        have = {fp.stem for fp in (out / fld).glob("*.npy")}  # "<年>_d<日>"
        want_names = {f"{y}_d{t}" for y, t in want}
        report[fld] = len(have)
        miss = sorted(want_names - have)
        if miss:
            missing[fld] = miss
    meta["written"] = report
    meta["missing"] = missing
    meta["status"] = "done" if not missing else "incomplete"
    json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
    tag = "OK" if not missing else f"缺 {sum(len(v) for v in missing.values())} 个场-日"
    print(f"[dump] finalize {out.name}: {report} -> status={meta['status']} ({tag})")


def save_field(out, sub, y, day, arr):
    d = Path(out) / sub
    d.mkdir(exist_ok=True)
    np.save(d / f"{y}_d{day}.npy", arr)


def main():
    ap = argparse.ArgumentParser(description="阶段 B 采样落盘(逐日逐像素场, 不出图)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stage-a-ckpt", required=True, help="校验 μ 缓存绑定")
    ap.add_argument("--diffusion-ckpt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--all-days", action="store_true",
                    help="取 years 内全部日(365 日历/年); 缺省用每月 5/15/25 抽样")
    ap.add_argument("--days", nargs="+", default=None,
                    help='指定 "年-日序" 列表(如 2020-0 2020-100), 覆盖 years/all-days; '
                         "用于冒烟或补采个别缺失日")
    ap.add_argument("--months", type=int, nargs="+", default=None,
                    help="只采这些月份(1-12)的全部日; 采样仍是全域整幅, 不裁区域")
    ap.add_argument("--ablate", default=None,
                    help=f"通道置换实验: 组名 {sorted(AB.GROUPS)} / 复合 {sorted(AB.COMPOSITES)} / "
                         f"'{AB.NONE}'(对照, 现算 μ 但不置换) / 逗号分隔的通道名")
    ap.add_argument("--ablate-mode", choices=["zero", "doy"], default="zero",
                    help="动态通道的置换值: zero=归一化空间填 0(抹掉天气+季节+空间梯度, "
                         "分布外); doy=换成 --doy-year 同一 day-of-year 的实测场(只抹天气, 分布内)")
    ap.add_argument("--doy-year", type=int, default=2019,
                    help="--ablate-mode doy 的供体年份; 不得与 --years 重合")
    ap.add_argument("--save-mu", action="store_true",
                    help="额外落盘阶段 A 的 μ(物理单位), 供离线做阶段 A/B 误差分解")
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--overlap", type=int, required=True)
    ap.add_argument("--boundary", type=int, required=True)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-data", type=float, required=True)
    ap.add_argument("--model-channels", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程校验各场文件数并标注 status; 不采样")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ti = TD.TARGETS.index(a.target)
    out = Path(a.out)
    days = resolve_days(a)

    if a.finalize:
        finalize(out, a, days)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    out.mkdir(parents=True, exist_ok=True)
    mine = days[rank::ntasks]
    if rank == 0:
        write_meta(out, a, days)

    stats = TD.Stats(a.stats_dir, TD.DEFAULT_IN, TD.TARGETS)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, a.years, TD.DEFAULT_IN, TD.TARGETS, stats)
    H, W = d.H, d.W

    # 置换会改变 μ, 缓存随之失效 -> 消融(含对照)一律现算 μ; 常规 dump 仍走缓存
    slots, net_a, donor, cache = [], None, None, None
    if a.ablate:
        slots = AB.channel_slots(AB.resolve(a.ablate), TD.DEFAULT_IN)
        net_a = load_stage_a(a.stage_a_ckpt, device)
        if a.ablate_mode == "doy":
            if a.doy_year in a.years:
                raise ValueError(f"--doy-year {a.doy_year} 与 --years 重合, 供体必须是另一年")
            donor = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [a.doy_year],
                                     TD.DEFAULT_IN, TD.TARGETS, stats)
    else:
        cache = MuCache(a.cache, [a.target])
        cache.verify({a.target: a.stage_a_ckpt})
    print(f"[dump] rank {rank}/{ntasks} device={device} 分到 {len(mine)} 天; ablate={a.ablate or '-'}; 场={fields_for(a.target, a.save_mu)}",
          flush=True)

    net = EDMPrecondSuperResolution(
        img_resolution=[H, W], img_in_channels=41 + 100, img_out_channels=1,
        model_type="SongUNetPosEmbd", model_channels=a.model_channels,
        channel_mult=[1, 2, 2], attn_resolutions=[16],
        N_grid_channels=100, gridtype="learnable",
        sigma_data=a.sigma_data).to(device).eval()
    net.load_state_dict(torch.load(a.diffusion_ckpt, map_location=device)["model"])
    patching = GridPatching2D(img_shape=(H, W), patch_shape=(a.patch, a.patch),
                              overlap_pix=a.overlap, boundary_pix=a.boundary)
    is_precip = a.target == TD.PRECIP

    for y, day in mine:
        t0 = time.time()
        cond, _tgt, mask, hr = d.full(y, day)
        land = mask[0] > 0.5
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        if slots:
            dn = donor.full(a.doy_year, day)[0] if donor is not None else None
            cond = apply_ablation(cond, slots, a.ablate_mode, dn)
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
        if net_a is not None:                       # 在(可能已置换的)条件上现算 μ
            with torch.no_grad():
                mu_n = TD.det_predict(net_a, cond_t, tile=0, device=device)[0]
        else:
            mu_n = cache.get(a.target, y, day)
        mu_t = pin_ocean(torch.from_numpy(mu_n[None, None]).float().to(device), land_t)

        members_norm = []
        for m in range(a.members):
            torch.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
            lat = torch.randn(1, 1, H, W, device=device)
            with torch.no_grad():
                r = stochastic_sampler(
                    net=net, latents=lat, img_lr=cond_t, patching=patching,
                    mean_hr=mu_t, num_steps=a.steps, sigma_min=a.sigma_min,
                    sigma_max=a.sigma_max, rho=7, S_churn=0, S_noise=1)
            members_norm.append(mu_t[0, 0].cpu().numpy() + r[0, 0].float().cpu().numpy())
        members_norm = np.stack(members_norm, 0)                    # (M,H,W) 归一化(log1p)空间

        if is_precip and stats.precip_log:
            mem_phys = TD.precip_inv(members_norm * stats.d_std[ti] + stats.d_mean[ti],
                                     stats.precip_scale) * stats.precip_scale
            # 融合后 drizzle 截断, 与预处理同口径, 恢复零质量
            mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)
        else:
            mem_phys = members_norm * stats.d_std[ti] + stats.d_mean[ti]

        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)

        # CRPS(物理): 逐像素, 陆地外置 NaN
        _, crps_px = TD.crps_ensemble(mem_phys[:, None], truth[None], land, per_pixel=True)
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

        if a.save_mu:                               # 阶段 A 的 μ(物理单位), 供 A/B 误差分解
            mu_phys = mu_n * stats.d_std[ti] + stats.d_mean[ti]
            if is_precip and stats.precip_log:
                mu_phys = TD.precip_inv(mu_phys, stats.precip_scale) * stats.precip_scale
                mu_phys = np.where(mu_phys < stats.precip_clip, 0.0, mu_phys)
            save_field(out, "mu", y, day, np.where(land, mu_phys, np.nan).astype(np.float32))

        if is_precip:
            # log1p(mm) 空间 CRPS: 与功率谱一致取物理量的 log1p(max(.,0))
            mem_log = np.log1p(np.maximum(mem_phys, 0.0))
            truth_log = np.log1p(np.maximum(truth, 0.0))
            _, crps_log_px = TD.crps_ensemble(mem_log[:, None], truth_log[None], land, per_pixel=True)
            save_field(out, "crps_log", y, day,
                       np.where(land, crps_log_px[0], np.nan).astype(np.float32))

        print(f"  [rank{rank}] {y}-d{day} 落场完成 {time.time() - t0:.0f}s", flush=True)

    print(f"[dump] rank {rank} 完成 {len(mine)} 天", flush=True)
    if ntasks == 1:
        finalize(out, a, days)


if __name__ == "__main__":
    main()
