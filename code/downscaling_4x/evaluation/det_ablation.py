#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
det_ablation.py — 确定性模型的通道敏感性实验, 逐像素逐月落盘
============================================================================
置换条件通道(缺省 = 逐通道消融, history_51 下共 51 通道 + none 基线), 重算确定性预测,
看 |pred − truth| 怎么变。确定性方法没有采样噪声, ΔMAE 是精确值, 不需要噪声地板、
不需要配对种子; 一天只读一次数据、复用同一份条件张量跑完所有分组。

落盘 <out>/mae_<组>.npz(组名里的冒号在文件名中写成双下划线):
    mae_month  (12,H,W) float32  该组逐月的 |pred−truth| 均值(有效域外 NaN)
    n_days     (12,)    int32    各月天数
基线组固定叫 none(不置换任何通道)。ΔMAE = mae_month(组) − mae_month(none), 逐像素、
逐月, 任何按分区/季节的聚合都是它的函数, 与本脚本解耦 —— 任选区域出结果不必重跑实验。

对照自检: none 组与正式落场(det_dump 的 ens_mean)是同一个确定性前向, 应逐位一致;
每次运行都会对第一天做这项比对(给了 --check-dump 时)。
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
from downscaling_4x.evaluation import ablation_groups as AB
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.det_dump import METHOD_ARCH, METHOD_MOE, NN_METHODS
from downscaling_4x.evaluation.metrics import precip_log_mm
from downscaling_4x.training import train_downscale as TD


def _load_net(method, ckpt_path, device, era5_dir=M.ERA5_DIR, allow_input_mismatch=False):
    """载入确定性主体(冻结、推理态); method↔结构的硬校验与 det_dump 同一套。"""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck = payload.get("args", {})
    II.check(ck.get("era5_dir"), era5_dir, f"checkpoint {ckpt_path}", allow=allow_input_mismatch)
    arch = ck.get("arch", "unet") or "unet"
    if arch != METHOD_ARCH[method]:
        raise SystemExit(f"--method {method} 要求 arch={METHOD_ARCH[method]}, "
                         f"checkpoint 是 {arch!r} —— 拿错了 checkpoint")
    if method in METHOD_MOE and bool(ck.get("moe")) != METHOD_MOE[method]:
        raise SystemExit(f"--method {method} 要求 moe={METHOD_MOE[method]}, "
                         f"checkpoint moe={bool(ck.get('moe'))} —— jda/jma 拿反了")
    if ck.get("crop"):
        raise SystemExit(f"checkpoint 是按 crop={ck['crop']} 训的(仅冒烟用), 不能整幅评测")
    mode = ck.get("mode", C.DEFAULT_MODE)
    tgt = ck.get("target", C.TARGETS[0])
    out_vars = list(C.TARGETS) if tgt == "all" else [tgt]
    net = TD.build_regressor(C.cond_channels(mode), len(out_vars), ck, hw=C.HR_SHAPE)
    net.load_state_dict(payload["model"])
    net.to(device).eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net, mode, out_vars


def main():
    ap = argparse.ArgumentParser(description="确定性模型通道敏感性(逐像素逐月 ΔMAE)")
    ap.add_argument("--method", required=True, choices=NN_METHODS)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target", default=None, choices=C.TARGETS,
                    help="单目标 checkpoint 缺省取其自述目标; 联合(all)模型必须指定")
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--groups", nargs="+", default=None,
                    help=f"只跑给定的组(机理组/复合/通道名); 缺省 = none + 逐通道; "
                         f"可选 {sorted(AB.GROUPS)} {sorted(AB.COMPOSITES)}")
    ap.add_argument("--ablate-mode", choices=["zero", "doy"], default="zero")
    ap.add_argument("--doy-year", type=int, default=2019)
    ap.add_argument("--space", choices=["phys", "log"], default="phys",
                    help="log 仅对降水有效: MAE 在 log1p(mm) 空间上算")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--check-dump", default="",
                    help="正式落场的目标目录(含 ens_mean/); 首日对 none 组做逐位一致自检")
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程把各 rank 的分片合并成 mae_<组>.npz; 不算前向")
    ap.add_argument("--max-days", type=int, default=0, help=">0 时只取前 N 天, 冒烟用")
    ap.add_argument("--out", required=True)
    II.add_arg(ap)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.finalize:                                       # 合并各 rank 的分片
        parts = sorted((out / "parts").glob("mae_*_rank*.npz"))
        assert parts, f"没有分片可合并: {out/'parts'}"
        acc, cnt = {}, None
        for f in parts:
            g = f.stem[len("mae_"):].rsplit("_rank", 1)[0]
            z = np.load(f)
            acc[g] = acc.get(g, 0.0) + z["sum_month"].astype(np.float64)
            if g == AB.fname(AB.NONE):
                cnt = (cnt if cnt is not None else 0) + z["n_days"].astype(np.int64)
        sea = np.load(out / "sea_mask.npy") if (out / "sea_mask.npy").exists() else None
        for g, v in acc.items():
            m = v / np.maximum(cnt, 1)[:, None, None]
            if sea is not None:
                m[:, sea] = np.nan                       # 有效域外统一置 NaN, 与单机路径一致
            np.savez_compressed(out / f"mae_{g}.npz", mae_month=m.astype(np.float32),
                                n_days=cnt.astype(np.int32))
        print(f"[det-abl] finalize: {len(acc)} 组, 天数/月 {cnt.tolist()} -> {out}")
        return

    # ★ 设备要在载入模型★之前★按 rank 定好: 先建到 cuda:0 再改 device, 权重会留在 cuda:0
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local % torch.cuda.device_count())
        device = f"cuda:{local % torch.cuda.device_count()}"
    else:
        device = "cpu"

    net, mode, out_vars = _load_net(a.method, a.checkpoint, device, a.era5_dir,
                                    getattr(a, II.ALLOW_DEST, False))
    got = str(next(net.parameters()).device)
    assert got == device, f"模型在 {got} 而输入在 {device}: 设备初始化必须早于载模型"
    target = a.target or (out_vars[0] if len(out_vars) == 1 else None)
    if target is None:
        raise SystemExit("联合(all)模型必须给 --target 指定看哪个输出通道")
    if target not in out_vars:
        raise SystemExit(f"checkpoint 只输出 {out_vars}, 不含 --target {target}")
    oi = out_vars.index(target)                          # 模型输出通道
    ti = C.TARGETS.index(target)                         # 合同/统计通道
    is_precip = (target == C.PRECIP)
    log_space = (a.space == "log")
    if log_space and not is_precip:
        raise ValueError("--space log 只对降水有意义")

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)

    def make_data(years, split):
        avail = sorted({y - 1 for y in years} | set(years)) if need else sorted(years)
        fi = FrameIndex(list(years), avail, need, lags, split=split)
        ys = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
        d = DownscaleData(a.era5_dir, a.daymet_dir, ys, stats, mode=mode,
                          era5_cache_years=len(ys))
        return fi, d

    fi, d = make_data(a.years, "abl")
    H, W = d.H, d.W
    donor_fi = donor_d = None
    if a.ablate_mode == "doy":
        if a.doy_year in a.years:
            raise ValueError("--doy-year 不得与 --years 重合")
        donor_fi, donor_d = make_data([a.doy_year], "abl-donor")
        donor_idx = {f: k for k, f in enumerate(donor_fi.frames)}

    if a.groups:
        groups = list(a.groups)
    else:
        groups = [AB.NONE] + AB.per_channel(mode)        # 缺省 = 逐通道消融
    if AB.NONE not in groups:
        groups = [AB.NONE] + groups                      # 基线必须有, 否则 ΔMAE 无从谈起
    slots = {g: AB.channel_slots(AB.resolve(g, mode), mode) for g in groups}
    month_of = {y: np.array([x.month for x in M.daymet_dates(y)], int) for y in a.years}

    frames = list(range(len(fi)))
    if a.max_days:
        frames = frames[: a.max_days]
    my = frames[rank::ntasks]
    acc = {g: np.zeros((12, H, W), np.float32) for g in groups}
    cnt = np.zeros(12, np.int32)
    # float32 标量: 反归一化的算术精度路径与 det_dump 完全相同, none 组才能与正式落场逐位一致
    dstd, dmean = stats.d_std[ti], stats.d_mean[ti]
    print(f"[det-abl] method={a.method} target={target} mode={mode} 天数={len(my)}/{len(frames)} "
          f"分组={len(groups)} ablate={a.ablate_mode} device={device}", flush=True)
    t0 = time.time()

    def forward(cond):
        """置换后的条件 -> 物理场(float32, 温度 K / 降水 mm/day), 算术与 det_dump 相同。"""
        with torch.no_grad():
            mu = net(torch.from_numpy(cond[None]).float().to(device))[0, oi]
        mu = mu.float().cpu().numpy() * dstd + dmean
        if is_precip and stats.precip_log:
            mu = C.precip_inv(mu, stats.precip_scale) * stats.precip_scale
        return mu

    land = d.mask
    for k, idx in enumerate(my):
        y, t = fi.frames[idx]
        cond, _tg, _mask, hr = d.full(y, t, fi.history_of(idx))
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        dn = None
        if donor_d is not None:
            j = donor_idx.get((a.doy_year, t))
            if j is None:
                raise RuntimeError(f"doy 供体缺帧 ({a.doy_year}, d{t}): 无完整历史")
            dn = donor_d.full(a.doy_year, t, donor_fi.history_of(j))[0]
        mo = int(month_of[y][t]) - 1
        cnt[mo] += 1
        for g in groups:
            cb = AB.apply_ablation(cond, slots[g], a.ablate_mode, dn) if slots[g] else cond
            mu = forward(cb)
            if k == 0 and g == AB.NONE and a.check_dump:
                ref = np.load(Path(a.check_dump) / "ens_mean" / f"{y}_d{t}.npy")
                mine = np.where(land, mu, np.nan).astype(np.float32)
                if not np.array_equal(np.nan_to_num(mine), np.nan_to_num(ref.astype(np.float32))):
                    dmax = float(np.nanmax(np.abs(mine - ref)))
                    raise RuntimeError(f"none 组与正式落场不一致 (最大差 {dmax:.3e}): "
                                       "权重/条件/反变换有一处对不上, 全部结果不可信")
                print(f"[det-abl] none 组与正式落场逐位一致 ({y}_d{t})", flush=True)
            if log_space:
                err = np.abs(precip_log_mm(mu, stats.precip_clip) - precip_log_mm(truth, stats.precip_clip))
            else:
                err = np.abs(mu - truth)
            acc[g][mo] += np.where(land, err, 0.0).astype(np.float32)
        if (k + 1) % 10 == 0 or k + 1 == len(my):
            el = time.time() - t0
            print(f"[det-abl] {k+1}/{len(my)}  {el:.0f}s  预计总计 {el/(k+1)*len(my):.0f}s", flush=True)

    nanmask = ~land
    if ntasks > 1:                                       # 分片: 存原始和, 由 --finalize 合并
        (out / "parts").mkdir(exist_ok=True)
        for g in groups:
            np.savez_compressed(out / "parts" / f"mae_{AB.fname(g)}_rank{rank}.npz",
                                sum_month=acc[g], n_days=cnt.astype(np.int32))
        if rank == 0:
            np.save(out / "sea_mask.npy", nanmask)
    else:
        for g in groups:
            m = acc[g] / np.maximum(cnt, 1)[:, None, None]
            m[:, nanmask] = np.nan
            np.savez_compressed(out / f"mae_{AB.fname(g)}.npz", mae_month=m.astype(np.float32),
                                n_days=cnt.astype(np.int32))
    if rank != 0:
        print(f"[det-abl] rank {rank} 完成 {len(my)} 天 {time.time()-t0:.0f}s", flush=True)
        return
    meta = {"id": out.name, "date": datetime.date.today().isoformat(),
            "kind": "det_channel_ablation", "method": a.method, "target": target,
            "unit": ("log1p(mm)" if log_space else ("mm/day" if is_precip else "K")),
            "space": a.space, "mode": mode,
            "checkpoint": str(a.checkpoint), "years": list(a.years),
            "input": II.describe(a.era5_dir),
            "n_days": len(frames),
            "groups": {g: AB.describe(g, mode) for g in groups},
            "file_of_group": {g: f"mae_{AB.fname(g)}.npz" for g in groups},
            "ablate_mode": a.ablate_mode,
            "doy_year": a.doy_year if a.ablate_mode == "doy" else None,
            "baseline": AB.NONE,
            "note": "ΔMAE = mae_month(组) − mae_month(none); 确定性, 无采样噪声; "
                    "逐像素逐月落盘, 任选区域的聚合都无需重跑",
            "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(f"[det-abl] 完成 {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
