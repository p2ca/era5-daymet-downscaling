#!/usr/bin/env python
# Packaged implementation; code/fit_bcsd_coefs.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
fit_bcsd_coefs.py — 把 BCSD 的逐像素系数 (a, b) 拟合出来并★存盘★

为什么需要它: 原实验 train_statistical.py 拟合完就直接评测, ★系数没存★ ——
于是每次想换个空间/换个指标重算 BCSD, 都得重跑 38 年的最小二乘(单变量 ~30 分钟)。
存一次, 以后任何重算都是秒级。

BCSD 定义 (与 train_statistical.py 完全一致):
  温度: Daymet(K)          ~ a * bilinear(ERA5)(K)          + b     (恒等空间)
  降水: log1p(Daymet_mm)   ~ a * log1p(bilinear(ERA5)_mm)   + b     (对数空间)
  -> 降水的预测要用 max(expm1(.),0)/1000 反变换回 m/day 才能与其他方法比

用法 (三个变量可并行):
  python fit_bcsd_coefs.py --var 2m_temperature_max --out runs/bcsd_coefs &
  python fit_bcsd_coefs.py --var 2m_temperature_min --out runs/bcsd_coefs &
  python fit_bcsd_coefs.py --var total_precipitation_24hr --out runs/bcsd_coefs &
"""
import argparse
import os
import sys
import time

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import (
    FACTOR,
    PRECIP,
    _load_raw_static,
    _squeeze_2d,
    fill_nan_daymean,
    make_bilinear,
    npz_memmap,
)


def hr_day(hrmm, t):
    d = np.asarray(hrmm[t], np.float32)
    return d[0] if d.ndim == 3 else d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--var", required=True)
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--out", default="runs/bcsd_coefs")
    a = p.parse_args()

    var = a.var
    precip = (var == PRECIP)
    # 各变量在自己的空间里拟合 (与 train_statistical.py 一致)
    tf = (lambda x: np.log1p(np.maximum(x, 0) * 1000.0)) if precip else (lambda x: x)
    space = "log1p(mm)" if precip else "K"

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, f"{var}.npz")
    if os.path.exists(dst):
        print(f"[{var}] 系数已存在, 跳过 -> {dst}")
        return

    t0 = time.time()
    up = None
    Sx = Sy = Sxx = Sxy = None
    ntr = 0
    for ty in a.train_years:
        ef = M.find_year_files(a.era5_dir, ty); df = M.find_year_files(a.daymet_dir, ty)
        lr = M.load_var_stack(ef, var); hrmm = npz_memmap(df[0], var)
        if up is None:
            up = make_bilinear(lr.shape[1], lr.shape[2], FACTOR)
            H, W = hr_day(hrmm, 0).shape
            Sx = np.zeros((H, W), np.float64); Sy = Sx.copy(); Sxx = Sx.copy(); Sxy = Sx.copy()
        lrf = fill_nan_daymean(tf(lr))
        for t in range(0, lr.shape[0], a.train_stride):
            x = up(lrf[t]); y = tf(hr_day(hrmm, t))
            Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; ntr += 1
        print(f"[{var}] {ty}  累计 {ntr} 天  ({time.time()-t0:.0f}s)", flush=True)

    den = ntr * Sxx - Sx * Sx
    coef_a = np.where(den != 0, (ntr * Sxy - Sx * Sy) / den, 1.0).astype(np.float32)
    coef_b = np.where(den != 0, (Sy - coef_a.astype(np.float64) * Sx) / max(ntr, 1), 0.0).astype(np.float32)
    np.savez_compressed(dst, a=coef_a, b=coef_b, n_train_days=ntr, space=space, var=var)
    print(f"[{var}] 完成 n={ntr} 天, 空间={space} -> {dst}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
