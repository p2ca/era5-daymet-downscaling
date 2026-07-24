#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
把某年每一天画成一张图: 左列 ERA5(120x240), 右列 Daymet(720x1440)
每个变量一行(默认 3 个变量) -> 每天 3 行 x 2 列。365 天 = 365 张图。
============================================================================
多进程并行: 把 365 天切成 N 段连续区间, 每个进程负责一段, 整年数据只解压一次。

用法:
  python plot_daily_grid.py --year 2020 --workers 8
  python plot_daily_grid.py --year 2020 --workers 12 \
         --vars 2m_temperature_max 2m_temperature_min total_precipitation_24hr
  python plot_daily_grid.py --year 2020 --day-start 0 --day-end 10 --workers 2   # 先试前10天

输出: {OUT}/daily/{year}/day{idx:03d}_{YYYY-MM-DD}.png

说明:
  * 海洋/Daymet 域外用各自 mask 屏蔽成白色; 对齐后的 ERA5 会自动读 valid_mask 限制到对齐域。
  * 每个变量用"整年统一"的色标(便于跨天对比/做动画); 温度=viridis, 降水=log1p+Blues。
  * 内存: 每进程载入 3 个 Daymet 变量(各~1.5GB)≈ 4.5GB。--workers 8 ≈ 36GB,
    在 Frontier 计算节点没问题; 想省内存就把 workers 调小。
============================================================================
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.tools.plotting import plot_match_maps as P


def _mask_flip(e_land_b, dm_land_b):
    """用 mask 重合度判断 Daymet 是否要纬度翻转(便宜, 不加载大场)。"""
    if e_land_b is None or dm_land_b is None:
        return False
    up  = M.block_mean(dm_land_b.astype(np.float32)[None], M.FACTOR)[0] > 0.5
    upf = M.block_mean(dm_land_b[::-1, :].astype(np.float32)[None], M.FACTOR)[0] > 0.5
    return bool(np.mean(e_land_b == upf) > np.mean(e_land_b == up))


def _year_range(arr, precip):
    """整年统一色标(robust 2/98 分位, 忽略 NaN)。"""
    a = P.transform_precip(arr) if precip else arr
    v = a[np.isfinite(a)]
    if v.size == 0:
        return (0.0, 1.0)
    return float(np.percentile(v, 2)), float(np.percentile(v, 98))


def render_chunk(task):
    year, day_lo, day_hi, era5_dir, daymet_dir, out_dir, varlist = task
    try:
        ef = M.find_year_files(era5_dir, year)
        df = M.find_year_files(daymet_dir, year)
        if not ef or not df:
            return f"{year}[{day_lo}:{day_hi}]: 缺 ERA5/Daymet 文件"
        fig_dir = os.path.join(out_dir, "daily", str(year))
        os.makedirs(fig_dir, exist_ok=True)
        extent, _ = P.geo_extent(ef)

        e_land  = M.load_static_2d(ef, M.LAND_MASK_VAR)
        dm_land = M.load_static_2d(df, M.LAND_MASK_VAR)
        e_land_b  = (e_land  > M.LAND_THRESH) if e_land  is not None else None
        dm_land_b = (dm_land > M.LAND_THRESH) if dm_land is not None else None
        vmask = M.load_static_2d(ef, "valid_mask")          # 对齐后的 ERA5 有这个
        if vmask is not None:
            vb = vmask > 0.5
            e_land_b = vb if e_land_b is None else (e_land_b & vb)
        flip = _mask_flip(e_land_b, dm_land_b)
        dates = M.era5_dates(year)

        # 整年载入 3 变量(每段进程只解压一次), 顺手算整年统一色标
        E, D, RNG = {}, {}, {}
        for v in varlist:
            ev = M._prep_era5_var(ef, v, e_land_b)           # (T,120,240) 海洋NaN
            dv = M.load_var_stack(df, v)                     # (T,720,1440)
            if ev is None or dv is None:
                continue
            if dm_land_b is not None:
                dv[:, ~dm_land_b] = np.nan
            if flip:
                dv = dv[:, ::-1, :]
            E[v], D[v] = ev, dv
            lo1, hi1 = _year_range(ev, P.is_precip(v))
            lo2, hi2 = _year_range(dv, P.is_precip(v))
            RNG[v] = (min(lo1, lo2), max(hi1, hi2))
        if not E:
            return f"{year}: 三个变量都缺失"
        vars_ok = list(E.keys())
        Tn = min(min(E[v].shape[0] for v in vars_ok), min(D[v].shape[0] for v in vars_ok))
        nv = len(vars_ok)

        n_done = 0
        for di in range(day_lo, min(day_hi, Tn)):
            fig, axes = P._new_axes(nv, 2, (11, 3.4 * nv))
            for r, v in enumerate(vars_ok):
                precip = P.is_precip(v)
                el = P.transform_precip(E[v][di]) if precip else E[v][di]
                dl = P.transform_precip(D[v][di]) if precip else D[v][di]
                cmap = "Blues" if precip else "viridis"
                unit = "log1p(mm)" if precip else "K"
                vmin, vmax = RNG[v]
                i0 = P.draw(axes[r, 0], el, extent, f"ERA5  {v}  [{unit}]  120x240",
                            cmap, vmin, vmax, e_land)
                i1 = P.draw(axes[r, 1], dl, extent, f"Daymet  {v}  [{unit}]  720x1440",
                            cmap, vmin, vmax, (dm_land[::-1, :] if (flip and dm_land is not None) else dm_land))
                fig.colorbar(i0, ax=axes[r, 0], fraction=0.046, pad=0.04)
                fig.colorbar(i1, ax=axes[r, 1], fraction=0.046, pad=0.04)
            fig.suptitle(f"{year}   day_idx={di:03d}   {dates[di].isoformat()}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(os.path.join(fig_dir, f"day{di:03d}_{dates[di].isoformat()}.png"),
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
            n_done += 1
        del E, D
        import gc; gc.collect()
        return f"{year}[{day_lo}:{min(day_hi, Tn)}]: 出图 {n_done} 张"
    except Exception as e:
        import traceback
        return f"{year}[{day_lo}:{day_hi}]: ERROR {e} | {traceback.format_exc().strip().splitlines()[-1]}"


def main():
    ap = argparse.ArgumentParser(description="逐日画 ERA5(左) vs Daymet(右), 三变量三行",
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--vars", nargs="+", default=M.CHECK_VARS)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--out", default=M.OUT_DIR)
    ap.add_argument("--workers", type=int, default=4, metavar="N")
    ap.add_argument("--day-start", type=int, default=0)
    ap.add_argument("--day-end", type=int, default=M.DAYS_PER_YEAR)
    args = ap.parse_args()

    lo, hi = args.day_start, min(args.day_end, M.DAYS_PER_YEAR)
    W = max(1, args.workers)
    n = hi - lo
    if n <= 0:
        print("没有要画的天。"); return
    per = math.ceil(n / W)
    chunks = [(args.year, s, min(s + per, hi), args.era5_dir, args.daymet_dir, args.out, args.vars)
              for s in range(lo, hi, per)]

    print("=" * 64)
    print(f"逐日画图  year={args.year}  days[{lo}:{hi}]={n}天  vars={args.vars}")
    print(f"workers={W}  (切成 {len(chunks)} 段, 每段整年只解压一次, 每进程约 4.5GB 内存)")
    print(f"ERA5  : {args.era5_dir}\nDaymet: {args.daymet_dir}\n输出  : {args.out}/daily/{args.year}/")
    print(f"cartopy={'有' if P.HAS_CARTOPY else '无(用 land mask 等值线当海岸线)'}")
    print("=" * 64)

    if W > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=min(W, len(chunks))) as ex:
            for r in ex.map(render_chunk, chunks):
                print("[完成]", r)
    else:
        for c in chunks:
            print("[完成]", render_chunk(c))
    print(f"\n完成。图在 {args.out}/daily/{args.year}/  (day000_*.png ... day364_*.png)")


if __name__ == "__main__":
    main()
