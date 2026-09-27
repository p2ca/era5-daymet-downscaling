#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
fit_bcsd_coefs.py — BCSD 的逐像素系数 (a, b) 拟合并存盘
============================================================================
逐像素最小二乘:

  温度: Daymet(K)              ~ a * bilinear(ERA5)(K)              + b   (恒等空间)
  降水: precip_fwd(Daymet)     ~ a * precip_fwd(bilinear(ERA5))     + b   (log1p(mm) 空间)

降水两侧都走合同的 `precip_fwd`(含 <0.1 mm/day 置零), 与训练目标同一条变换 —— 基线与
神经网络必须被同一个目标定义评分, 否则两者的降水指标不在同一个空间里, 而数值看着都正常。

存一次系数, 之后任何重算/换指标都是秒级。

用法(三个变量可并行):
  python -m downscaling_4x.baselines.fit_bcsd_coefs --var 2m_temperature_max --out runs/...
============================================================================
"""
import argparse
import os
import time

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.downscale_baseline import (fill_nearest_valid, make_bilinear,
                                                    nearest_valid_index, npz_memmap)
from downscaling_4x.evaluation import input_identity as II


def hr_day(mm, t):
    d = np.asarray(mm[t], np.float32)
    while d.ndim > 2:
        d = d[0]
    return d


def main():
    p = argparse.ArgumentParser(description="BCSD 逐像素系数拟合")
    p.add_argument("--var", required=True, choices=C.TARGETS)
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    var = a.var
    precip = (var == C.PRECIP)
    tf = C.precip_fwd if precip else (lambda x: np.asarray(x, np.float32))
    space = "log1p(mm)" if precip else "K"

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, f"{var}.npz")
    if os.path.exists(dst):
        print(f"[{var}] 系数已存在, 跳过 -> {dst}")
        return

    t0 = time.time()
    up = make_bilinear(*C.LR_SHAPE, C.FACTOR)
    # ERA5 在 valid_mask 外整片缺测; 缺测格取最近有效格的值, 与取数模块同一条口径
    vm = M.load_static_2d(a.era5_dir, "valid_mask")
    if vm is None:
        raise FileNotFoundError(f"{a.era5_dir}/static.npz 缺 valid_mask")
    fill_idx = nearest_valid_index(vm > 0.5)
    H, W = C.HR_SHAPE
    Sx = np.zeros((H, W), np.float64); Sy = Sx.copy(); Sxx = Sx.copy(); Sxy = Sx.copy()
    ntr = 0
    for ty in a.train_years:
        ef = M.find_year_files(a.era5_dir, ty)
        df = M.find_year_files(a.daymet_dir, ty)
        if not ef or not df:
            raise FileNotFoundError(f"{ty}: 缺 ERA5 或 Daymet")
        lr = M.load_var_stack(ef, var)
        if lr is None or lr.shape[1:] != C.LR_SHAPE:
            raise ValueError(f"{ty}: ERA5 的 {var} 形状异常 {None if lr is None else lr.shape}")
        hrmm = npz_memmap(df[0], var)
        if hrmm is None:
            raise ValueError(f"{df[0]} 的 {var} 不是未压缩存储, 无法 memmap")
        lrf = fill_nearest_valid(lr, fill_idx)
        for t in range(0, lr.shape[0], a.train_stride):
            x = tf(up(lrf[t])); y = tf(hr_day(hrmm, t))
            Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; ntr += 1
        print(f"[{var}] {ty}  累计 {ntr} 天  ({time.time() - t0:.0f}s)", flush=True)

    den = ntr * Sxx - Sx * Sx
    coef_a = np.where(den != 0, (ntr * Sxy - Sx * Sy) / den, 1.0).astype(np.float32)
    coef_b = np.where(den != 0, (Sy - coef_a.astype(np.float64) * Sx) / max(ntr, 1), 0.0).astype(np.float32)
    ident = II.describe(a.era5_dir)
    # 记下拟合用的输入目录: 真实 ERA5 与 oracle 输入同布局, 评测侧靠这个字段拒绝混用
    np.savez_compressed(dst, a=coef_a, b=coef_b, n_train_days=ntr, space=space, var=var,
                        factor=C.FACTOR, hr_shape=np.array(C.HR_SHAPE),
                        train_years=np.array(a.train_years), train_stride=a.train_stride,
                        era5_dir=ident["era5_dir"], daymet_dir=II.normalize(a.daymet_dir),
                        input_product=ident["input_product"])
    print(f"[{var}] 完成 n={ntr} 天, 空间={space} -> {dst}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
