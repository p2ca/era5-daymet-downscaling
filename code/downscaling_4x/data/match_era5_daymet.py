#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
match_era5_daymet.py — 数据发现、年份划分与日历
============================================================================
ERA5 侧取 17 变量的动态集, Daymet 侧取 3.75 arcmin(480x960) 的目标与静态场。
两侧都按"闰年丢 12/31"处理成每年恒 365 帧, 因此同一 day_index 指同一天, 无闰年漂移。

`calendar_365()` 是 (year, day_index) -> 真实日期的唯一换算处。跨年取历史帧一律经它,
不在年度文件内做索引加减: 年度文件只有 365 帧, 减索引会在年初静默绕回同年年末。
============================================================================
"""
import glob
import os
from datetime import date, timedelta

import numpy as np

DATA_ROOT = "/lustre/orion/atm112/world-shared/patrickfan"
ERA5_DIR = f"{DATA_ROOT}/era5/0.25_deg_Daily_Golden_DaymetAligned_dynamic_no200"
DAYMET_DIR = f"{DATA_ROOT}/daymet/3.75_arcmin"

splits = {
    "train": list(range(1980, 2018)),
    "val":   [2018, 2019],
    "test":  [2020],
}

DAYS_PER_YEAR = 365
LEAP_DROP = "dec31"

FILL_SENTINELS = (-9999.0, -999.0, 9.969209968386869e36)
LAND_MASK_VAR = "land_sea_mask"
LAND_THRESH = 0.5


# ---------------------------------------------------------------------------
# 日历
# ---------------------------------------------------------------------------
def _all_dates(year):
    d, end, out = date(year, 1, 1), date(year, 12, 31), []
    while d <= end:
        out.append(d); d += timedelta(days=1)
    return out


def calendar_365(year, leap_drop=LEAP_DROP):
    """长度恰为 365 的真实日期列表(day_index -> date), 按闰年约定删一天。"""
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
    return calendar_365(year, leap_drop)


def era5_dates(year, leap_drop=LEAP_DROP):
    return calendar_365(year, leap_drop)


def frame_date(year, day_index):
    """(年, 年内第几天) -> 真实日期。闰年丢 12/31 的约定下, 恰等于 1 月 1 日加偏移。"""
    if not 0 <= day_index < DAYS_PER_YEAR:
        raise ValueError(f"day_index {day_index} 越出 [0,{DAYS_PER_YEAR - 1}]")
    return date(int(year), 1, 1) + timedelta(days=int(day_index))


# ---------------------------------------------------------------------------
# 文件 IO
# ---------------------------------------------------------------------------
def find_year_files(base_dir, year):
    """在 base_dir(可能有 train/val/test 子目录)里找该年 npz; 支持 {year}.npz 与分片 {year}_*.npz。"""
    found = []
    search_dirs = [os.path.join(base_dir, s) for s in ("train", "val", "test")] + [base_dir]
    for d in search_dirs:
        if os.path.isfile(os.path.join(d, f"{year}.npz")):
            found.append(os.path.join(d, f"{year}.npz"))
        for p in sorted(glob.glob(os.path.join(d, f"{year}_*.npz")),
                        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])):
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
    """取某变量沿时间轴拼接 -> (T,H,W) float32(已清缺测); 非空间变量返回 None。"""
    parts = []
    for f in files:
        with np.load(f, allow_pickle=True) as npz:
            if var in npz.files:
                a = _to_thw(npz[var])
                if a.ndim < 1:
                    return None
                parts.append(a)
    if not parts:
        return None
    try:
        return clean_fill(np.concatenate(parts, axis=0))
    except ValueError:
        return None


def load_static_2d(base_dir, var):
    """从数据集根目录的 static.npz 取静态 2D 场 -> (H,W) float32; 取不到返回 None。

    静态场在年度文件里被逐日复制了 365 份, 单文件 8GB; 一律走 static.npz, 不碰年度文件。
    """
    p = os.path.join(base_dir, "static.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if var not in z.files:
            return None
        a = np.asarray(z[var])
    while a.ndim > 2:
        a = a[0]
    return a.astype(np.float32)
