#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
把 ERA5(120x240) 的 boundary 对齐到 Daymet 的有效域, 生成"新 ERA5"
============================================================================
为什么:
  ERA5 覆盖整个矩形 box(含海洋、加拿大北部……), 但 Daymet 只在其原生陆地域有真实数据,
  域外是常数填充(tmax/tmin≈226.9K, precip=0)。两者 footprint 不一致。
  本脚本用 Daymet 的有效域做掩膜, 把新 ERA5 在 Daymet 域外的格点置 NaN(或 --fill 指定值),
  于是新 ERA5 与 Daymet 共享同一个边界/footprint。

掩膜怎么来:
  Daymet land_sea_mask (720x1440) > 0.5  ->  6x6 块平均  ->  阈值 --threshold  ->  (120x240) 有效域。
  (Daymet mask 是静态的, 每年一样; 脚本只读这个小变量, 不解压 1.5GB 的大场, 所以很快。)

被对齐(打掩膜)的变量:
  ERA5 npz 里所有最后两维为 (120,240) 的空间场(含 (365,120,240) 动态场和 (120,240) 静态场);
  其余(标量、退化列)原样拷贝。输出额外带 valid_mask(120x240) 和 hr_valid_mask(720x1440)。

用法:
  python align_era5_to_daymet.py --split test
  python align_era5_to_daymet.py --split train --workers 16
  python align_era5_to_daymet.py --year 2020 2019 2018
  python align_era5_to_daymet.py --year 2020 --fill 0          # 域外填 0(而非 NaN)
  python align_era5_to_daymet.py --split test --fill keep       # 不改值, 只附 valid_mask

输出: {OUT_DIR}/{split}/{year}.npz  —— 与原 ERA5 同 key, 空间场已按 Daymet 边界裁掉。
============================================================================
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from era5_daymet.data import match_era5_daymet as M

# 新 ERA5 输出目录(★按需修改★)
OUT_DIR = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_Golden_DaymetAligned"


def daymet_lr_valid(dm_files, factor=M.FACTOR, thr=0.5):
    """Daymet land_sea_mask -> (lr_valid 120x240 bool, hr_valid 720x1440 bool)。取不到返回 (None,None)。"""
    dm = M.load_static_2d(dm_files, M.LAND_MASK_VAR)     # 只读 mask 这一个变量, 很快
    if dm is None:
        return None, None
    hr_valid = dm > M.LAND_THRESH
    lr_valid = M.block_mean(hr_valid.astype(np.float32)[None], factor)[0] > thr
    return lr_valid, hr_valid


def align_year(year, era5_dir, daymet_dir, out_dir, fill=np.nan, thr=0.5, compress=True):
    e_files = M.find_year_files(era5_dir, year)
    d_files = M.find_year_files(daymet_dir, year)
    if not e_files:
        print(f"  [skip] {year}: 找不到 ERA5"); return None
    if not d_files:
        print(f"  [skip] {year}: 找不到 Daymet"); return None
    lr_valid, hr_valid = daymet_lr_valid(d_files, thr=thr)
    if lr_valid is None:
        print(f"  [skip] {year}: Daymet 没有 {M.LAND_MASK_VAR}"); return None

    # 读 ERA5 全部变量(允许分片), 同 key 只取一次
    src = {}
    for f in e_files:
        with np.load(f, allow_pickle=True) as npz:
            for k in npz.files:
                src.setdefault(k, npz[k])

    out, n_masked = {}, 0
    for k, a in src.items():
        a = np.asarray(a)
        if a.ndim >= 2 and a.shape[-2:] == lr_valid.shape:     # 空间场 (...,120,240)
            if fill is not None:                                # fill=None 即 keep, 不改值
                a = a.astype(np.float32, copy=True)
                a[..., ~lr_valid] = fill
                n_masked += 1
        out[k] = a
    out["valid_mask"] = lr_valid                                # (120,240) bool, True=Daymet 有数据
    out["hr_valid_mask"] = hr_valid                             # (720,1440) bool, 给 HR 端 loss 用

    split = next((s for s, ys in M.splits.items() if year in ys), "train")
    od = os.path.join(out_dir, split); os.makedirs(od, exist_ok=True)
    op = os.path.join(od, f"{year}.npz")
    (np.savez_compressed if compress else np.savez)(op, **out)
    mode = "keep(仅附mask)" if fill is None else f"fill={fill}"
    print(f"  {year}: 空间场掩膜 {n_masked} 个, 保留 {lr_valid.mean():.1%} 格点, {mode} -> {op}")
    return op


def _worker(t):
    year, ed, dd, od, fill, thr, comp = t
    try:
        r = align_year(year, ed, dd, od, fill, thr, comp)
        return f"{year}: {'ok' if r else 'skip'}"
    except Exception as e:
        return f"{year}: ERROR {e}"


def main():
    p = argparse.ArgumentParser(description="把 ERA5 边界对齐到 Daymet 有效域, 生成新 ERA5",
                                formatter_class=argparse.RawTextHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--year", type=int, nargs="+")
    g.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--era5-dir", default=M.ERA5_DIR, help="原 ERA5 黄金帧目录")
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--out", default=OUT_DIR, help="新 ERA5 输出目录")
    p.add_argument("--workers", type=int, default=1, metavar="N", help="按年并行进程数")
    p.add_argument("--fill", default="nan", help="Daymet 域外填什么: nan(默认) | 0 | <数值> | keep(不改值只附mask)")
    p.add_argument("--threshold", type=float, default=0.5, help="6x6 块里 Daymet 陆地占比阈值(默认0.5)")
    p.add_argument("--no-compress", action="store_true", help="np.savez 不压缩(更快, 占盘大)")
    args = p.parse_args()

    fill = (np.nan if args.fill == "nan" else None if args.fill == "keep" else float(args.fill))
    compress = not args.no_compress
    years = (args.year if args.year else
             M.splits[args.split] if args.split else
             [y for ys in M.splits.values() for y in ys])
    workers = max(1, args.workers)

    print("=" * 64)
    print(f"对齐 ERA5 -> Daymet 边界   年份={years}  workers={workers}  fill={args.fill}  thr={args.threshold}")
    print(f"原 ERA5 : {args.era5_dir}")
    print(f"Daymet  : {args.daymet_dir}")
    print(f"新 ERA5 : {args.out}")
    print("=" * 64)

    if len(years) > 1 and workers > 1:
        tasks = [(y, args.era5_dir, args.daymet_dir, args.out, fill, args.threshold, compress) for y in years]
        with ProcessPoolExecutor(max_workers=min(workers, len(years))) as ex:
            for r in ex.map(_worker, tasks):
                print(f"[完成] {r}")
    else:
        for y in years:
            align_year(y, args.era5_dir, args.daymet_dir, args.out, fill, args.threshold, compress)
    print("\n完成。新 ERA5 与 Daymet 现在共享同一边界(Daymet 域外已置 " + args.fill + ")。")


if __name__ == "__main__":
    main()
