#!/usr/bin/env python
# Packaged implementation; code/compute_norm_stats.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
计算"训练集"的逐变量 归一化均值/方差 + 气候态, 分别为 ERA5 / Daymet 保存
============================================================================
铁律(防泄漏): 只在 TRAIN 上算; val/test 一律加载同一份, 不重算。

输出目录结构:
  {out}/era5/normalize_mean.npz     {var: 标量 mean}
  {out}/era5/normalize_std.npz      {var: 标量 std}
  {out}/era5/climatology.npz        {var: (Nslot,120,240)}  (仅 dynamic 变量)
  {out}/daymet/normalize_mean.npz   {var: 标量 mean}
  {out}/daymet/normalize_std.npz    {var: 标量 std}
  {out}/daymet/climatology.npz      {var: (Nslot,720,1440)} (仅 dynamic 变量)
  (各目录另存 meta.json: 变量清单/训练年/clim模式/slot 数)

处理哪些变量(从 npz 自动发现):
  - 跳过: 标量(hrs_each_step/num_steps_per_shard/extra_steps)、退化列(days_of_year/
          time_of_day 是 (365,120,0))、掩膜(land_sea_mask/valid_mask/hr_valid_mask)、
          重复拼写(lattitude)。
  - 静态(landcover/latitude/orography): 算 mean/std, 不算气候态。
  - 其余空间场: 算 mean/std + 气候态。
统计域: ERA5 用"有限值"像素(对齐域外的 NaN 自动排除; 全 NaN 的变量如 sea_surface_temperature
        在陆地域无值 -> 跳过并提示)。Daymet 用 land_sea_mask>0.5 且有限(排除海洋 226.9 填充)。
值空间: 原始物理单位(不做 log)。降水若想 log, 在模型侧处理。

用法:
  python compute_norm_stats.py --train-years 2018 2019 --out stats_train
  # 全量: python compute_norm_stats.py --train-years $(seq 1980 2017) --clim dayofyear --out stats_train
============================================================================
"""
import argparse
import json
import os
import sys

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import npz_memmap, _squeeze_2d, _load_raw_static

SKIP = {"hrs_each_step", "num_steps_per_shard", "extra_steps", "days_of_year", "time_of_day",
        "land_sea_mask", "valid_mask", "hr_valid_mask", "lattitude"}
STATIC = {"landcover", "latitude", "orography"}
PRECIP = "total_precipitation_24hr"


def precip_fwd(p, clip, scale=1000.0):
    """降水建模空间: 原始单位(米)先 ×scale 变毫米, 再把过小的(drizzle, ERA5 尤其严重)clip 成 0, 再 log1p。"""
    p = np.maximum(p, 0.0) * scale          # 米 -> 毫米
    p = np.where(p < clip, 0.0, p)          # drizzle clip(毫米)
    return np.log1p(p)


def n_slots(mode):
    # dayofyear 用 day_idx(0..364) 当 slot: 你的日历是 dec31-drop, 每年恒 365 天,
    # 所以是 365 个 slot(不是天文 day-of-year 的 366; 那样第 366 槽永远空)。
    return {"annual": 1, "monthly": 12, "dayofyear": 365}[mode]


def slot_index(mode, year, ndays):
    if mode == "annual":
        return np.zeros(ndays, int)
    if mode == "monthly":
        dates = M.era5_dates(year)[:ndays]
        return np.array([d.month - 1 for d in dates], int)
    # dayofyear: 直接用"年内第几天"(day_idx), 跨年对齐稳, 无空槽
    return np.minimum(np.arange(ndays), 364)


def smooth_doy(clim, win):
    if win <= 1 or clim.shape[0] < 3:
        return clim
    k = win // 2
    pad = np.concatenate([clim[-k:], clim, clim[:k]], 0)
    cs = np.cumsum(pad, 0)
    return ((cs[2 * k:] - cs[:-2 * k]) / (2 * k))[:clim.shape[0]].astype(np.float32)


def list_vars(files):
    with np.load(files[0], allow_pickle=True) as z:
        return list(z.files)


def accumulate(get_day, years_ndays, HW, dynamic, mode, stride, land_mask, tf=None):
    """逐天累计: 标量 mean/std(有限&land masked) + 逐像素气候态(Nslot,H,W)。无有效像素返回 None。
    tf: 可选逐天变换(如降水 log1p+clip), 在统计/气候态前应用。"""
    H, W = HW; Ns = n_slots(mode) if dynamic else 1
    csum = np.zeros((Ns, H, W)); cpix = np.zeros((Ns, H, W))
    s = ss = cnt = 0.0
    for (year, nd) in years_ndays:
        sl = slot_index(mode, year, nd) if dynamic else np.zeros(nd, int)
        for t in range(0, nd, stride):
            x = np.asarray(get_day(year, t), np.float64)
            if tf is not None:
                x = tf(x)
            fin = np.isfinite(x)
            if land_mask is not None:
                fin &= land_mask
            if not fin.any():
                continue
            sli = sl[t]
            csum[sli] += np.where(fin, x, 0.0); cpix[sli] += fin
            xv = x[fin]; s += xv.sum(); ss += (xv * xv).sum(); cnt += xv.size
    if cnt == 0:
        return None
    mean = s / cnt; std = float(np.sqrt(max(ss / cnt - mean ** 2, 1e-8)))
    clim = (csum / np.maximum(cpix, 1)).astype(np.float32)
    return float(mean), std, clim


def process_side(name, base_dir, train_years, mode, stride, smooth, out_dir, is_daymet,
                 precip_log=True, precip_clip=0.1, precip_scale=1000.0):
    files0 = M.find_year_files(base_dir, train_years[0])
    if not files0:
        print(f"[{name}] 找不到 {train_years[0]} 的文件"); return
    allv = list_vars(files0)
    means, stds, clims = {}, {}, {}
    # Daymet 评测域 mask
    land = None
    if is_daymet:
        land = _squeeze_2d(_load_raw_static(files0, M.LAND_MASK_VAR)) > 0.5
    years_files = {y: M.find_year_files(base_dir, y) for y in train_years}

    for v in allv:
        if v in SKIP:
            continue
        dynamic = v not in STATIC
        tf = (lambda a: precip_fwd(a, precip_clip, precip_scale)) if (precip_log and v == PRECIP) else None
        try:
            if is_daymet:
                # 静态 2D -> 直接读; dynamic -> memmap 逐天
                if not dynamic:
                    arr = _squeeze_2d(_load_raw_static(years_files[train_years[0]], v))
                    if arr is None or arr.ndim != 2 or min(arr.shape) < 2:
                        continue
                    fin = np.isfinite(arr) & (land if land is not None else True)
                    if not np.asarray(fin).any():
                        continue
                    vals = arr[fin]; means[v] = float(vals.mean()); stds[v] = float(max(vals.std(), 1e-6))
                    print(f"  [{name}] {v:30s}(static) mean/std={means[v]:.3f}/{stds[v]:.3f}")
                    continue
                mm = {y: npz_memmap(years_files[y][0], v) for y in train_years}
                if any(m is None for m in mm.values()):
                    print(f"  [{name}] {v}: 非 memmap 跳过"); continue
                H, W = land.shape
                def get(y, t, _mm=mm):
                    d = np.asarray(_mm[y][t]); return d[0] if d.ndim == 3 else d
                yn = [(y, mm[y].shape[0]) for y in train_years]
                r = accumulate(get, yn, (H, W), True, mode, stride, land, tf=tf)
            else:
                # ERA5: 整年入内存(小), 逐变量处理
                per = {}
                ok = True
                for y in train_years:
                    st = M.load_var_stack(years_files[y], v)
                    if st is None or st.ndim != 3 or min(st.shape[1:]) < 2:
                        ok = False; break
                    per[y] = st
                if not ok:
                    continue
                H, W = next(iter(per.values())).shape[1:]
                def get(y, t, _p=per):
                    return _p[y][t]
                yn = [(y, per[y].shape[0]) for y in train_years]
                r = accumulate(get, yn, (H, W), dynamic, mode, stride, None, tf=tf)
            if r is None:
                print(f"  [{name}] {v}: 无有效像素(可能是海洋变量在陆地域), 跳过"); continue
            mean, std, clim = r
            means[v] = mean; stds[v] = std
            if dynamic:
                clims[v] = smooth_doy(clim, smooth) if mode == "dayofyear" else clim
            print(f"  [{name}] {v:30s}{'(dyn)' if dynamic else '(static)'} mean/std={mean:.3f}/{std:.3f}")
        except Exception as e:
            print(f"  [{name}] {v}: 错误 {e}")

    d = os.path.join(out_dir, name); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "normalize_mean.npz"), **{k: np.float32(v) for k, v in means.items()})
    np.savez_compressed(os.path.join(d, "normalize_std.npz"), **{k: np.float32(v) for k, v in stds.items()})
    np.savez_compressed(os.path.join(d, "climatology.npz"), **clims)
    json.dump(dict(train_years=train_years, clim=mode, n_slots=n_slots(mode), stride=stride,
                   precip_log=precip_log, precip_clip=precip_clip, precip_scale=precip_scale, precip_var=PRECIP,
                   vars_meanstd=list(means), vars_clim=list(clims)),
              open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"  -> {d}/  (mean:{len(means)} std:{len(stds)} clim:{len(clims)} 变量)")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--out", default="stats_train")
    p.add_argument("--clim", choices=["annual", "monthly", "dayofyear"], default="monthly")
    p.add_argument("--smooth", type=int, default=15)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-precip-log", action="store_true", help="降水不做 log1p(默认做)")
    p.add_argument("--precip-scale", type=float, default=1000.0,
                   help="降水原始单位->毫米的倍数(数据是米 -> 1000; 若已是毫米 -> 1)")
    p.add_argument("--precip-clip", type=float, default=0.1,
                   help="降水 drizzle 阈值(毫米): 小于此值 clip 成 0 再 log1p(默认 0.1mm)")
    args = p.parse_args()
    plog = not args.no_precip_log
    print(f"训练年={args.train_years}  clim={args.clim}  stride={args.stride}  "
          f"precip_log={plog} scale={args.precip_scale} clip={args.precip_clip}mm  -> {args.out}/{{era5,daymet}}/")
    print("ERA5:")
    process_side("era5", args.era5_dir, args.train_years, args.clim, args.stride, args.smooth, args.out, False,
                 precip_log=plog, precip_clip=args.precip_clip, precip_scale=args.precip_scale)
    print("Daymet:")
    process_side("daymet", args.daymet_dir, args.train_years, args.clim, args.stride, args.smooth, args.out, True,
                 precip_log=plog, precip_clip=args.precip_clip, precip_scale=args.precip_scale)
    print("\n完成。train/val/test 一律加载同一份:")
    print("  mu=np.load(f'{out}/era5/normalize_mean.npz'); sd=np.load(f'{out}/era5/normalize_std.npz')")
    print("  归一化: (x - mu[var]) / sd[var]")
    print("  气候态: clim=np.load(f'{out}/daymet/climatology.npz')[var][slot]; slot=日期->月(monthly)")


if __name__ == "__main__":
    main()
