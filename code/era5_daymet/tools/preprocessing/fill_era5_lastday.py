#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
用"粗化的 Daymet 最后一天"填补 ERA5 test/2020 缺失的最后一天
============================================================================
背景: ERA5 黄金帧 +33h 让 2020 的最后一天(day_idx 364)落到 2021, 而 2021 无数据,
      所以 ERA5 这天是上一天(363)的复制(占位)。Daymet 的 364 是真数据。
做法: 把 Daymet day364 (720x1440) 屏蔽海洋 -> 6x 块平均 -> 120x240 -> 按 ERA5 valid_mask
      对齐(域外置 NaN), 写回 ERA5 文件的 day364(只改两边都有的变量: tmax/tmin/precip)。
      其它只有 ERA5 才有的变量(风/位势...)这天仍是 363 的复制(没有 Daymet 对应, 无法粗化)。

注意: 这天的 ERA5 输入≈粗化的 Daymet 目标 -> 该天降尺度会偏易(1/365 天, 影响极小);
      没有 ERA5 真值时这是最合理的占位。

用法:
  python fill_era5_lastday.py \
      --era5-file paired_era5_daymet/test/2020.npz \
      --daymet-file daymet/2.5_arcmin/test/2020_0.npz \
      --out paired_era5_daymet/test/2020.npz        # 默认覆盖(建议先备份)
============================================================================
"""
import argparse
import os
import sys

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import (
    _load_raw_static,
    _squeeze_2d,
    npz_memmap,
)
from era5_daymet.data.match_era5_daymet import block_mean

FACTOR = 6
SHARED = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]


def coarsen_daymet_day(daymet_file, var, day, dm_land, valid_mask):
    """Daymet[var][day] (720x1440) -> 屏蔽海洋 -> 6x 块平均 -> 120x240 -> 对齐 ERA5 域(域外NaN)。"""
    mm = npz_memmap(daymet_file, var)
    if mm is None:
        return None
    d = np.asarray(mm[day]); d = d[0] if d.ndim == 3 else d
    d = d.astype(np.float32).copy()
    d[~dm_land] = np.nan                      # 海洋/域外的填充值(226.9 / 0)不参与平均
    coarse = block_mean(d[None], FACTOR)[0]   # (120,240) nanmean
    if valid_mask is not None:
        coarse[~valid_mask] = np.nan          # 与对齐后的 ERA5 一致(域外 NaN)
    return coarse.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--era5-file", required=True)
    ap.add_argument("--daymet-file", required=True)
    ap.add_argument("--out", default=None, help="默认覆盖 era5-file(建议先备份)")
    ap.add_argument("--day", type=int, default=-1, help="要替换的 day_idx(默认 -1=最后一天)")
    ap.add_argument("--vars", nargs="+", default=SHARED)
    ap.add_argument("--no-compress", action="store_true")
    args = ap.parse_args()
    out = args.out or args.era5_file

    z = dict(np.load(args.era5_file, allow_pickle=True))   # 全部 key 入内存
    nday = next(v.shape[0] for v in z.values() if getattr(v, "ndim", 0) == 3 and v.shape[0] >= 300)
    day = args.day % nday
    dm_land = _squeeze_2d(_load_raw_static([args.daymet_file], M.LAND_MASK_VAR)) > 0.5
    vmask = (z["valid_mask"] > 0.5) if "valid_mask" in z else None

    print(f"ERA5 文件: {args.era5_file}  天数={nday}  替换 day={day}")
    for v in args.vars:
        if v not in z:
            print(f"  [skip] ERA5 无变量 {v}"); continue
        if z[v].ndim != 3 or z[v].shape[1:] != (vmask.shape if vmask is not None else z[v].shape[1:]):
            pass
        coarse = coarsen_daymet_day(args.daymet_file, v, day, dm_land, vmask)
        if coarse is None:
            print(f"  [skip] Daymet 无变量 {v}(或非 memmap)"); continue
        before = z[v][day].copy()
        z[v][day] = coarse
        dup = np.allclose(np.nan_to_num(before), np.nan_to_num(z[v][day - 1]), atol=1e-3)
        print(f"  {v:26s} day{day} 替换为粗化Daymet; 替换前是否=day{day-1}的复制? {dup}; "
              f"新值有限均值={np.nanmean(coarse):.3f}")

    saver = np.savez if args.no_compress else np.savez_compressed
    saver(out, **z)
    print(f"-> 写出 {out}  ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
