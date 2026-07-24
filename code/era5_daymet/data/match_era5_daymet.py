#!/usr/bin/env python
# Packaged implementation; code/match_era5_daymet.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
ERA5(120x240, 0.25°, 黄金帧)  <->  Daymet(720x1440, 2.5 arcmin)  逐日匹配流水线
============================================================================
配合 extract_golden_frames.py 使用:
  - 输入(低分辨, LR): extract_golden_frames.py 的输出  {ERA5_DIR}/{split}/{year}.npz
                       每个变量 shape (365, 120, 240), 0.25°, CONUS box
  - 目标(高分辨, HR): Daymet 2.5 arcmin               {DAYMET_DIR}/{split}/{year}*.npz
                       每个变量 shape (365, 720, 1440), 同一个 CONUS box
  - factor = 6  (720/120 = 1440/240, 0.25° / (2.5/60)° = 6)

只检查这三个变量(npz 内): 2m_temperature_max, 2m_temperature_min, total_precipitation_24hr

本版前提(按你的实际处理):
  * ERA5 与 Daymet 都按"闰年丢 12/31"处理 -> 两端日历一致, 没有闰年漂移。
    => day_idx d 在两端本应是同一天(见 APPLY_GOLDEN_OFFSET 关于 +33h 的说明)。
  * ERA5 与 Daymet 本就是两个不同的 product, 所以"同一天"的空间相关性有天花板:
        温度 tmax/tmin ~ 0.96,  降水 ~ 0.68。
    => 不能用同一个阈值; 本版按变量分别设下限, 并用"温度"来定位日期对齐(信噪比高),
       降水只记录相关性、不作为丢弃依据(降水低相关多是产品差异/干日, 不代表配错日期)。

为什么仍要"每天都查":
  即便日历一致, 仍可能有(a)黄金帧 +33h 造成的整体 ±1 天偏移, (b)个别坏帧/缺测,
  (c)Daymet 纬度方向(origin)不一致。逐日核对能定位并隔离这些天。

"match" 用两条证据:
  (A) 日历对齐: 两端同约定(LEAP_DROP)还原真实日期, 同日期才配对。
  (B) 经验相关性: Daymet 6x6 块平均降到 ERA5 网格, 与 ERA5 同日做皮尔逊相关。
      相关性对单位/线性变换不敏感(°F/°C、log/原值都不影响)。用温度定位偏移, 三变量都报相关性。

用法(只剩 verify; 训练用的 ERA5 由 align_era5_to_daymet.py 产出, 不再用 pair 打包):
  python match_era5_daymet.py verify --year 2016 2020      # 只核对、出报告
  python match_era5_daymet.py verify --split train --workers 16

输出:
  {OUT_DIR}/reports/{year}_dayreport.csv   每天一行(三变量相关性 + flag)
  {OUT_DIR}/reports/{year}_summary.json    该年汇总
============================================================================
"""

import argparse
import contextlib
import csv
import gc
import glob
import json
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean 全 NaN 切片

# ===========================================================================
# 0. 用户可改配置  (★ 跑之前请核对这几项 ★)
# ===========================================================================
# 默认用"对齐到 Daymet 边界后"的新 ERA5(align_era5_to_daymet.py 的输出)。
# 想用原始未对齐的, 命令行加: --era5-dir /lustre/.../era5/0.25_deg_Daily_Golden
ERA5_DIR   = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_Golden_DaymetAligned"
DAYMET_DIR = "/lustre/orion/atm112/world-shared/patrickfan/daymet/2.5_arcmin"
OUT_DIR    = "/lustre/orion/atm112/world-shared/patrickfan/paired_era5_daymet"

splits = {
    'train': list(range(1980, 2018)),
    'val':   [2018, 2019],
    'test':  [2020],
}

FACTOR        = 6            # 720/120 = 1440/240
DAYS_PER_YEAR = 365

# 闰年约定: 你已把 ERA5 和 Daymet 都按"闰年丢 12/31"处理 -> 两端一致, 无闰年漂移。
LEAP_DROP = "dec31"          # "dec31" | "feb29" | "none"

# extract_golden_frames.py 的 +33h: 是否把 ERA5 day_idx d 视作"日期 d+1"。
#   你两端按同一 index 处理 -> 默认 False(day_idx d <-> Daymet day_idx d 同一天)。
#   若校验报告里 global_offset_k 恒为 1、且大量 drift -> 说明 +33h 让 ERA5 整体晚一天,
#   把这里改成 True 再跑即可(脚本会自动按 +33h 重建日期)。
APPLY_GOLDEN_OFFSET = False
GOLDEN_OFFSET_H     = 33

# 只检查这三个变量
CHECK_VARS   = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
# 用"温度"定位日期对齐(场平滑、相关性高); 降水太噪, 不用来定位
PRIMARY_VARS = ["2m_temperature_max", "2m_temperature_min"]

# 逐变量"相关性下限"(ERA5/Daymet 是不同 product, 天花板不同):
#   温度 ~0.96 -> 下限 0.70(低于即明显错配/坏帧)
#   降水 ~0.68 -> 下限 0.10(只作记录, 不作丢弃依据)
CORR_FLOOR = {
    "2m_temperature_max":       0.70,
    "2m_temperature_min":       0.70,
    "total_precipitation_24hr": 0.10,
}
ADAPT_MAD_K = 6.0            # 自适应: 某天温度相关性 < 年内中位数 - K*MAD -> 标 temp_outlier(信息列)

LAG_SCAN       = range(-3, 4)
FILL_SENTINELS = (-9999.0, -999.0, 9.969209968386869e36)
LAND_MASK_VAR  = "land_sea_mask"
LAND_THRESH    = 0.5



def _r(x, n=4):
    """四舍五入或空串(给 CSV/JSON 用)。"""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return round(float(x), n)
    except Exception:
        return ""


# ===========================================================================
# 1. 日历: 把 day_idx 还原成真实日期
# ===========================================================================
def _all_dates(year):
    d, end, out = date(year, 1, 1), date(year, 12, 31), []
    while d <= end:
        out.append(d); d += timedelta(days=1)
    return out


def calendar_365(year, leap_drop=LEAP_DROP):
    """长度恰为 365 的真实日期列表(day_idx -> date), 按闰年约定删一天。"""
    days = _all_dates(year)
    if len(days) == 366:
        if leap_drop == "feb29":
            days = [d for d in days if not (d.month == 2 and d.day == 29)]
        elif leap_drop == "dec31":
            days = [d for d in days if not (d.month == 12 and d.day == 31)]
        elif leap_drop == "none":
            days = days[:365]
        else:
            raise ValueError(f"未知 leap_drop={leap_drop!r}")
    assert len(days) == 365, f"{year}: 期望 365 天, 实得 {len(days)}"
    return days


def daymet_dates(year, leap_drop=LEAP_DROP):
    """Daymet: day_idx d -> 真实日期(默认与 ERA5 同约定 dec31)。"""
    return calendar_365(year, leap_drop)


def era5_golden_dates(year, offset_h=GOLDEN_OFFSET_H, leap_drop=LEAP_DROP):
    """
    +33h 黄金帧版本: 把每年看作 8760 小时连续流, day_idx d 落在 d*24+offset_h,
    其真实日期 = 流中第 (d*24+offset_h)//24 天; 末尾会因 +33h 跨到下一年。
    仅当 APPLY_GOLDEN_OFFSET=True 时使用。
    """
    base, nxt, out = calendar_365(year, leap_drop), None, []
    for d in range(DAYS_PER_YEAR):
        di = (d * 24 + offset_h) // 24
        if di < DAYS_PER_YEAR:
            out.append(base[di])
        else:
            if nxt is None:
                nxt = calendar_365(year + 1, leap_drop)
            out.append(nxt[di - DAYS_PER_YEAR])
    return out


def era5_dates(year):
    """ERA5 黄金帧 day_idx -> 日期。默认按 index 对齐(与 Daymet 同日); +33h 仅在开关打开时生效。"""
    if APPLY_GOLDEN_OFFSET:
        return era5_golden_dates(year)
    return calendar_365(year, LEAP_DROP)


# ===========================================================================
# 2. 文件 IO
# ===========================================================================
def find_year_files(base_dir, year):
    """在 base_dir(可能有 train/val/test 子目录)里找该年 npz; 支持 {year}.npz 与分片 {year}_*.npz。"""
    found = []
    search_dirs = [os.path.join(base_dir, s) for s in ('train', 'val', 'test')] + [base_dir]
    for d in search_dirs:
        if os.path.isfile(os.path.join(d, f"{year}.npz")):
            found.append(os.path.join(d, f"{year}.npz"))
        for p in sorted(glob.glob(os.path.join(d, f"{year}_*.npz")),
                        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
            found.append(p)
    seen, uniq = set(), []
    for p in found:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _to_thw(a):
    """统一成 (T,H,W): (T,1,H,W)->(T,H,W); (H,W)->(1,H,W)。"""
    if a.ndim == 4:
        a = a[:, 0]
    elif a.ndim == 2:
        a = a[None]
    return a


def clean_fill(a):
    """缺测/哨兵 -> NaN, 转 float32。"""
    a = a.astype(np.float32, copy=True)
    for s in FILL_SENTINELS:
        a[a == np.float32(s)] = np.nan
    a[a <= -9000] = np.nan
    return a


def load_var_stack(files, var):
    """
    取某变量, 沿时间轴拼接 -> (T,H,W) float32(已清缺测)。
    非空间变量(0-d 标量如 num_steps_per_shard / extra_steps, 或无法拼接) -> None。
    """
    parts = []
    for f in files:
        with np.load(f, allow_pickle=True) as npz:
            if var in npz.files:
                a = _to_thw(npz[var])
                if a.ndim < 1:          # 0-d 标量, 不是可堆叠数组 -> 跳过(避免 concatenate 报错)
                    return None
                parts.append(a)
    if not parts:
        return None
    try:
        return clean_fill(np.concatenate(parts, axis=0))
    except ValueError:               # 维度不一致等, 当作非空间变量跳过
        return None


def load_static_2d(files, var):
    """取静态 2D 变量(如 mask) -> (H,W); 取不到返回 None。"""
    for f in files:
        with np.load(f, allow_pickle=True) as npz:
            if var in npz.files:
                a = npz[var]
                if a.ndim == 4: a = a[0, 0]
                elif a.ndim == 3: a = a[0]
                if a.ndim == 2:
                    return a.astype(np.float32)
    return None


def has_var(files, var):
    for f in files:
        with np.load(f, allow_pickle=True) as npz:
            if var in npz.files:
                return True
    return False


# ===========================================================================
# 3. 空间: 6x 块平均 + 相关性
# ===========================================================================
def block_mean(stack, factor=FACTOR):
    """(T,H,W) -> (T,H/factor,W/factor), 块平均(NaN 安全)。"""
    T, H, W = stack.shape
    assert H % factor == 0 and W % factor == 0, f"{H}x{W} 不能被 factor={factor} 整除"
    return np.nanmean(stack.reshape(T, H // factor, factor, W // factor, factor), axis=(2, 4))


def pearson_masked(a, b):
    """两张 2D 图在共同有效像素上的皮尔逊相关; 像素太少或常数场 -> NaN。"""
    if a is None or b is None:
        return np.nan
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 30:
        return np.nan
    av = a[m].astype(np.float64); bv = b[m].astype(np.float64)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    return float((av * bv).sum() / denom) if denom != 0 else np.nan


def valid_fraction(frame, land):
    if frame is None:
        return 0.0
    if land is None:
        return float(np.isfinite(frame).mean())
    denom = int(land.sum())
    if denom == 0:
        return float(np.isfinite(frame).mean())
    return float(np.isfinite(frame[land]).sum() / denom)


def best_lag(e_block, dm_block, sample_idx, lags=LAG_SCAN):
    """经验扫描整体偏移 k: corr(era5[i], daymet_block[i+k]) 平均最高的 k。返回 (best_k, {k:mean_corr})。"""
    n = min(len(e_block), dm_block.shape[0])
    scores = {}
    for k in lags:
        cc = [pearson_masked(e_block[i], dm_block[i + k])
              for i in sample_idx if 0 <= i < n and 0 <= i + k < n]
        scores[k] = float(np.nanmean(cc)) if cc else np.nan
    bk = max(scores, key=lambda k: (-1 if np.isnan(scores[k]) else scores[k]))
    return bk, scores


def _prep_era5_var(files, var, land_b):
    """ERA5 某变量 -> (365,120,240), 海洋置 NaN。无则 None。"""
    st = load_var_stack(files, var)
    if st is None:
        return None
    if land_b is not None:
        st[:, ~land_b] = np.nan
    return st


def _prep_daymet_block(files, var, land_b, flip=False):
    """Daymet 某变量 -> 6x 块平均到 (365,120,240), 海洋置 NaN, 可纬度翻转。无则 None。"""
    st = load_var_stack(files, var)
    if st is None:
        return None
    if land_b is not None:
        st[:, ~land_b] = np.nan
    b = block_mean(st, FACTOR)
    del st; gc.collect()
    return b[:, ::-1, :] if flip else b


def _dm_block_task(task):
    """ProcessPoolExecutor 用: 解压 1.5GB 在子进程里, 只把 42MB 的块平均结果传回。"""
    files, var, land_b, flip = task
    return var, _prep_daymet_block(files, var, land_b, flip)


def load_daymet_blocks(files, varlist, land_b, inner_workers=1, flip=False):
    """一次性载入多个 Daymet 变量的 6x 块平均。inner_workers>1 -> 多进程并行解压(各变量一个进程)。"""
    if not files:
        return {}
    if inner_workers and inner_workers > 1 and len(varlist) > 1:
        tasks = [(files, v, land_b, flip) for v in varlist]
        out = {}
        with ProcessPoolExecutor(max_workers=min(inner_workers, len(varlist))) as ex:
            for v, b in ex.map(_dm_block_task, tasks):
                out[v] = b
        return out
    return {v: _prep_daymet_block(files, v, land_b, flip) for v in varlist}


# ===========================================================================
# 4. 单年核对
# ===========================================================================
def verify_year(year, era5_dir=ERA5_DIR, daymet_dir=DAYMET_DIR, strict_calendar=False,
                inner_workers=1):
    """
    逐日核对 ERA5(120x240) 与 Daymet(720x1440) 是否同一天。
    只看 CHECK_VARS 三变量; 用温度(PRIMARY_VARS)定位日期, 降水仅记录。
    inner_workers>1 时, 三个 Daymet 变量并行解压(单年提速)。
    返回 (rows, summary)。不写数据。
    """
    print(f"\n[Verify] {year} ...")
    era5_files    = find_year_files(era5_dir, year)
    dm_files      = find_year_files(daymet_dir, year)
    dm_files_next = find_year_files(daymet_dir, year + 1)
    if not era5_files:
        raise FileNotFoundError(f"找不到 ERA5: {era5_dir} 下的 {year}.npz")
    if not dm_files:
        raise FileNotFoundError(f"找不到 Daymet: {daymet_dir} 下的 {year}*.npz")

    # ---- 4.1 日历(两端同约定): day_idx -> 真实日期 -> (date)->(dm_year,dm_idx) ----
    e_dates = era5_dates(year)
    dm_lookup = {dt: (year, i) for i, dt in enumerate(daymet_dates(year))}
    for i, dt in enumerate(daymet_dates(year + 1)):
        dm_lookup.setdefault(dt, (year + 1, i))
    cal_map = [dm_lookup.get(dt, (None, -1)) for dt in e_dates]
    cross_year = any(dmy == year + 1 for dmy, _ in cal_map)

    # ---- 4.2 mask ----
    e_land  = load_static_2d(era5_files, LAND_MASK_VAR)
    dm_land = load_static_2d(dm_files, LAND_MASK_VAR)
    e_land_b  = (e_land  > LAND_THRESH) if e_land  is not None else None
    dm_land_b = (dm_land > LAND_THRESH) if dm_land is not None else None
    # 对齐后的新 ERA5 带 valid_mask(=Daymet 有效域); 用它把相关性/校验限制在对齐域内
    vmask = load_static_2d(era5_files, "valid_mask")
    if vmask is not None:
        vb = vmask > 0.5
        e_land_b = vb if e_land_b is None else (e_land_b & vb)

    # ---- 4.3 主变量(温度) + 三个 daymet 块平均(可并行解压) ----
    pvar = next((v for v in PRIMARY_VARS if has_var(era5_files, v) and has_var(dm_files, v)), None)
    if pvar is None:
        raise KeyError(f"两端都缺温度主变量 {PRIMARY_VARS}, 无法定位日期对齐")
    e_p = _prep_era5_var(era5_files, pvar, e_land_b)
    # 三变量 Daymet 块平均一次性载入: inner_workers>1 时各变量一个进程并行解压(1.5GB 在子进程, 只回传 42MB)
    blocks = load_daymet_blocks(dm_files, CHECK_VARS, dm_land_b, inner_workers)
    blocks_next = (load_daymet_blocks(dm_files_next, CHECK_VARS, dm_land_b, 1)
                   if (cross_year and dm_files_next) else {})
    dm_p = blocks.get(pvar)
    if dm_p is None:
        raise KeyError(f"Daymet 缺少温度主变量 {pvar}")

    # 朝向(用日历同日期帧判断, 不受整体偏移影响)
    sample_idx = list(range(5, DAYS_PER_YEAR, 15))
    cs, cf = [], []
    for i in sample_idx:
        dmy, j = cal_map[i]
        if dmy == year and 0 <= j < dm_p.shape[0]:
            cs.append(pearson_masked(e_p[i], dm_p[j]))
            cf.append(pearson_masked(e_p[i], dm_p[j][::-1, :]))
    ms = float(np.nanmean(cs)) if cs else np.nan
    mf = float(np.nanmean(cf)) if cf else np.nan
    orient = 'flip' if (np.nan_to_num(mf, nan=-1.0) > np.nan_to_num(ms, nan=-1.0)) else 'same'
    if orient == 'flip':
        blocks = {k: v[:, ::-1, :] for k, v in blocks.items()}
        blocks_next = {k: v[:, ::-1, :] for k, v in blocks_next.items()}
        dm_p = blocks[pvar]
    dm_p_next = blocks_next.get(pvar)

    def dm_pframe(dm_year, idx):
        if idx is None or idx < 0:
            return None
        if dm_year == year and idx < dm_p.shape[0]:
            return dm_p[idx]
        if dm_p_next is not None and dm_year == year + 1 and idx < dm_p_next.shape[0]:
            return dm_p_next[idx]
        return None

    bk, lag_scores = best_lag(e_p, dm_p, sample_idx)
    half = DAYS_PER_YEAR // 2
    bk1, _ = best_lag(e_p, dm_p, [i for i in sample_idx if i < half])
    bk2, _ = best_lag(e_p, dm_p, [i for i in sample_idx if i >= half])
    offset_changed = (bk1 != bk2)
    print(f"  主变量={pvar}; 朝向={orient}(same={ms:.3f}/flip={mf:.3f}); "
          f"整体偏移 k*={bk}(上半年{bk1}/下半年{bk2}) "
          f"=> {'⚠ 偏移年中变化' if offset_changed else '偏移稳定'}")

    # 逐日: 日历 idx 与 邻域经验最佳 idx
    used_idx = np.full(DAYS_PER_YEAR, -1, dtype=int)
    used_dmy = [None] * DAYS_PER_YEAR
    cal_idx  = np.full(DAYS_PER_YEAR, -1, dtype=int)
    corr_p_cal  = np.full(DAYS_PER_YEAR, np.nan)
    corr_p_best = np.full(DAYS_PER_YEAR, np.nan)
    for i in range(DAYS_PER_YEAR):
        dmy, jc = cal_map[i]
        cal_idx[i] = jc
        corr_p_cal[i] = pearson_masked(e_p[i], dm_pframe(dmy, jc))
        cand = set()
        if jc >= 0:
            cand.update([jc - 1, jc, jc + 1])
        cand.add(i + bk)
        bj, bc = -1, -np.inf
        for j in cand:
            c = pearson_masked(e_p[i], dm_pframe(year, j))
            if not np.isnan(c) and c > bc:
                bc, bj = c, j
        corr_p_best[i] = bc if bc != -np.inf else np.nan
        if strict_calendar or bj < 0:
            used_idx[i], used_dmy[i] = jc, dmy
        else:
            used_idx[i], used_dmy[i] = bj, year

    # ---- 4.4 三变量在 used_idx 上的逐日相关性 + 主变量有效性 ----
    corr_by_var = {v: np.full(DAYS_PER_YEAR, np.nan) for v in CHECK_VARS}
    e_valid  = np.full(DAYS_PER_YEAR, np.nan)
    dm_valid = np.full(DAYS_PER_YEAR, np.nan)
    for var in CHECK_VARS:
        e_v = e_p if var == pvar else _prep_era5_var(era5_files, var, e_land_b)
        dm_v, dm_v_next = blocks.get(var), blocks_next.get(var)
        if e_v is None or dm_v is None:
            print(f"    [warn] 变量 {var} 在某端缺失, 该列留空")
            continue

        def dm_vframe(dm_year, idx, _v=dm_v, _vn=dm_v_next):
            if idx is None or idx < 0:
                return None
            if dm_year == year and idx < _v.shape[0]:
                return _v[idx]
            if _vn is not None and dm_year == year + 1 and idx < _vn.shape[0]:
                return _vn[idx]
            return None

        for i in range(DAYS_PER_YEAR):
            f = dm_vframe(used_dmy[i], used_idx[i])
            corr_by_var[var][i] = pearson_masked(e_v[i], f)
            if var == pvar:
                e_valid[i]  = valid_fraction(e_v[i], e_land_b)
                dm_valid[i] = valid_fraction(f, e_land_b)

    # 温度自适应基线(给"离群"信息列, 不用于丢弃)
    prim = [corr_by_var[v] for v in PRIMARY_VARS if v in corr_by_var]
    temp_corr = np.nanmean(np.vstack(prim), axis=0) if prim else np.full(DAYS_PER_YEAR, np.nan)
    med = float(np.nanmedian(temp_corr)) if np.isfinite(temp_corr).any() else np.nan
    mad = (float(np.nanmedian(np.abs(temp_corr - med))) * 1.4826
           if np.isfinite(temp_corr).any() else np.nan)
    adapt_floor = (med - ADAPT_MAD_K * mad) if (np.isfinite(med) and np.isfinite(mad)) else -np.inf

    # ---- 4.5 组装行 + flag ----
    rows = []
    counts = dict(ok=0, low_corr=0, unmatched_date=0, drift=0, invalid_frame=0)
    tfloor = CORR_FLOOR.get(pvar, 0.70)
    pfloor = CORR_FLOOR.get("total_precipitation_24hr", 0.10)
    for i in range(DAYS_PER_YEAR):
        dmy, jc = cal_map[i]
        tcorr = temp_corr[i]
        pcorr = corr_by_var.get("total_precipitation_24hr", np.full(DAYS_PER_YEAR, np.nan))[i]

        if jc < 0:
            flag = "unmatched_date"
        elif np.isnan(tcorr) or e_valid[i] < 0.2 or dm_valid[i] < 0.2:
            flag = "invalid_frame"
        elif tcorr < tfloor:
            flag = "low_corr"          # 温度都对不上 -> 极可能配错日期/坏帧
        elif (not strict_calendar and used_idx[i] != jc
              and not np.isnan(corr_p_best[i]) and not np.isnan(corr_p_cal[i])
              and corr_p_best[i] - corr_p_cal[i] > 0.1):
            flag = "drift"             # 经验最佳 idx ≠ 日历 idx 且明显更好 -> 查 +33h/偏移
        else:
            flag = "ok"
        counts[flag] += 1

        rows.append({
            "era5_idx": i,
            "era5_date": e_dates[i].isoformat(),
            "dm_year_used": used_dmy[i] if used_dmy[i] is not None else "",
            "dm_idx_used": int(used_idx[i]),
            "dm_date_used": (daymet_dates(used_dmy[i])[used_idx[i]].isoformat()
                             if (used_dmy[i] is not None and 0 <= used_idx[i] < DAYS_PER_YEAR) else ""),
            "corr_tmax":  _r(corr_by_var.get("2m_temperature_max", [np.nan] * 365)[i]),
            "corr_tmin":  _r(corr_by_var.get("2m_temperature_min", [np.nan] * 365)[i]),
            "corr_precip": _r(pcorr),
            "offset_days": int(used_idx[i] - i) if used_idx[i] >= 0 else "",
            "era5_valid_frac": _r(e_valid[i], 3),
            "daymet_valid_frac": _r(dm_valid[i], 3),
            "precip_flag": ("" if np.isnan(pcorr) else ("low" if pcorr < pfloor else "ok")),
            "temp_outlier": bool(np.isfinite(tcorr) and tcorr < adapt_floor),
            "flag": flag,
        })

    summary = {
        "year": year, "primary_var": pvar, "check_vars": CHECK_VARS,
        "leap_drop": LEAP_DROP, "apply_golden_offset": APPLY_GOLDEN_OFFSET,
        "is_leap_year": (len(_all_dates(year)) == 366),
        "orientation": orient, "global_offset_k": bk,
        "offset_first_half": bk1, "offset_second_half": bk2,
        "offset_changed_midyear": bool(offset_changed),
        "mean_corr_tmax":  _r(np.nanmean(corr_by_var.get("2m_temperature_max", [np.nan]))),
        "mean_corr_tmin":  _r(np.nanmean(corr_by_var.get("2m_temperature_min", [np.nan]))),
        "mean_corr_precip": _r(np.nanmean(corr_by_var.get("total_precipitation_24hr", [np.nan]))),
        "temp_corr_median": _r(med), "temp_corr_mad": _r(mad), "adapt_floor": _r(adapt_floor),
        "corr_floor": CORR_FLOOR, "lag_scores": {int(k): _r(v) for k, v in lag_scores.items()},
        "n_days": DAYS_PER_YEAR, **{f"n_{k}": v for k, v in counts.items()},
        "verdict": ("CLEAN" if counts["ok"] == DAYS_PER_YEAR else "NEEDS_REVIEW"),
    }
    print(f"  相关性均值: tmax={summary['mean_corr_tmax']} tmin={summary['mean_corr_tmin']} "
          f"precip={summary['mean_corr_precip']}")
    print(f"  汇总: ok={counts['ok']} low_corr={counts['low_corr']} "
          f"unmatched={counts['unmatched_date']} drift={counts['drift']} "
          f"invalid={counts['invalid_frame']} -> {summary['verdict']}")
    return rows, summary


def write_reports(year, rows, summary, out_dir=OUT_DIR):
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    csv_path = os.path.join(rep_dir, f"{year}_dayreport.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(rep_dir, f"{year}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  -> 报告: {csv_path}")
    return csv_path


# ===========================================================================
# 7. CLI
# ===========================================================================
def resolve_years(args):
    if args.year:
        return list(args.year)
    if args.split:
        return splits[args.split]
    return [y for ys in splits.values() for y in ys]


# ---- ProcessPoolExecutor 用的"按年"工作函数(必须在模块级别才能 pickle) ----
# 多进程同时写 stdout 会让日志交错(看着像 [Verify]1995 配 1985 报告), 这里把子进程的
# 内部打印吞掉, 只由主进程按完成顺序打印一行汇总, 日志就干净有序了。
def _verify_worker(task):
    year, era5_dir, daymet_dir, out, strict = task
    try:
        with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
            rows, summ = verify_year(year, era5_dir, daymet_dir, strict_calendar=strict, inner_workers=1)
            write_reports(year, rows, summ, out)
        return summ
    except Exception as e:
        return {"year": year, "verdict": "ERROR", "error": str(e),
                "where": traceback.format_exc().strip().splitlines()[-1]}


def main():
    p = argparse.ArgumentParser(
        description="ERA5(120x240) <-> Daymet(720x1440) 逐日匹配/对齐(只查 tmax/tmin/precip)",
        formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("verify", help="逐日核对 ERA5↔Daymet 是否同一天, 出报告(不产训练数据)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--year", type=int, nargs="+")
    g.add_argument("--split", choices=["train", "val", "test"])
    sp.add_argument("--era5-dir", default=ERA5_DIR)
    sp.add_argument("--daymet-dir", default=DAYMET_DIR)
    sp.add_argument("--out", default=OUT_DIR)
    sp.add_argument("--workers", type=int, default=1, metavar="N",
                    help="并行进程数: 多年时按年并行; 单年时按变量并行(<=3)。每进程峰值约 1.5GB。")
    sp.add_argument("--strict-calendar", action="store_true",
                    help="只信日历对齐, 不让经验相关性覆盖索引")
    args = p.parse_args()

    years   = resolve_years(args)
    workers = max(1, args.workers)
    print("=" * 64)
    print(f"verify  年份={years}  workers={workers}")
    print(f"ERA5  : {args.era5_dir}")
    print(f"Daymet: {args.daymet_dir}")
    print(f"LEAP_DROP={LEAP_DROP}  APPLY_GOLDEN_OFFSET={APPLY_GOLDEN_OFFSET}  查变量={CHECK_VARS}")
    print("=" * 64)

    all_summ = []
    if len(years) > 1 and workers > 1:
        # 多年: 按年并行(最有效, 尤其 train 38 年)
        nproc = min(workers, len(years))
        print(f"[并行] 按年并行 {nproc} 进程(每年内部串行)\n")
        tasks = [(y, args.era5_dir, args.daymet_dir, args.out, args.strict_calendar) for y in years]
        worker = _verify_worker
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for summ in ex.map(worker, tasks):
                all_summ.append(summ)
                v = summ.get("verdict")
                if v in (None, "ERROR"):
                    extra = f"  {summ.get('error', '')}"
                else:
                    extra = (f"  corr tmax/tmin/precip="
                             f"{summ.get('mean_corr_tmax')}/{summ.get('mean_corr_tmin')}/{summ.get('mean_corr_precip')}")
                print(f"[完成] {summ.get('year')}: {v}{extra}")
    else:
        # 单年(或 workers=1): 用 workers 做"变量级"并行
        for y in years:
            try:
                rows, summ = verify_year(y, args.era5_dir, args.daymet_dir,
                                         strict_calendar=args.strict_calendar,
                                         inner_workers=workers)
                write_reports(y, rows, summ, args.out)
                all_summ.append(summ)
            except Exception as e:
                print(f"[ERROR] {y}: {e}")
                all_summ.append({"year": y, "verdict": "ERROR", "error": str(e)})

    print("\n" + "=" * 64 + "\n总览:")
    for s in all_summ:
        v = s.get("verdict", "?")
        if v == "NEEDS_REVIEW":
            extra = (f"  (low_corr={s.get('n_low_corr')}, unmatched={s.get('n_unmatched_date')}, "
                     f"drift={s.get('n_drift')}, k*={s.get('global_offset_k')}, "
                     f"corr tmax/tmin/precip={s.get('mean_corr_tmax')}/{s.get('mean_corr_tmin')}/{s.get('mean_corr_precip')})")
        elif v == "ERROR":
            extra = f"  ({s.get('error')}; {s.get('where', '')})"
        else:
            extra = f"  (corr tmax/tmin/precip={s.get('mean_corr_tmax')}/{s.get('mean_corr_tmin')}/{s.get('mean_corr_precip')})"
        print(f"  {s.get('year')}: {v}{extra}")
    print("=" * 64)


if __name__ == "__main__":
    main()
