#!/usr/bin/env python
# Packaged implementation; code/eval_bcsd_both_spaces.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
eval_bcsd_both_spaces.py — 把 BCSD 的降水成绩同时给出两个空间的分数

背景 (docs/archive/results/2026-07-15-results.md / AGENTS.md):
  降水被两套代码在两种空间评测过, RMSE 之间没有换算关系, 跨空间比较 = 假结论。
    train_statistical.py (BCSD)          -> log1p(mm)
    eval_all_methods.py / train_unet.py  -> 原生 m/day
  于是 BCSD 至今无法与 UNet/ViT/插值 在降水上同台。

本脚本做的事:
  1. 用与原实验完全相同的口径重拟合 BCSD (log1p(mm) 空间, 逐 HR 像素最小二乘, 38 年 stride=1)
  2. 在 test_year 上同时评测两个空间:
       - log1p(mm): 与原实验对齐 (自检: bcsd 应复现 0.5453)
       - m/day    : 把 log 空间的预测用 expm1 反变换回物理量, 再与原始 Daymet 比
                    (自检: bilinear-raw 应复现 eval_all_methods 的 0.0046)
  ★两个自检都对上, 才能证明反变换口径没错、数字可以并排。

用法:
  python eval_bcsd_both_spaces.py --train-years $(seq 1980 2017) --test-year 2020 \
      --out runs/exp/20260713-bcsd-precip-both-spaces
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import (
    Acc,
    FACTOR,
    PRECIP,
    _load_raw_static,
    _squeeze_2d,
    fill_nan_daymean,
    make_bilinear,
    npz_memmap,
)

try:
    from scipy.ndimage import zoom as _zoom
except Exception:
    _zoom = None


def _bicubic(a2d, f=FACTOR):
    return _zoom(a2d, f, order=3, grid_mode=True, mode="nearest")


def hr_day(hrmm, t):
    """取 Daymet 第 t 天 -> (H,W) 原生 m/day。npz 里有的年份是 (T,H,W), 有的是 (T,1,H,W)。"""
    d = np.asarray(hrmm[t], np.float32)
    return d[0] if d.ndim == 3 else d


def fwd(a):
    """BCSD 的降水变换: 米 -> mm -> log1p (与 train_statistical.py 完全一致)。"""
    return np.log1p(np.maximum(a, 0) * 1000.0)


def inv(x):
    """反变换: log1p(mm) -> m/day, 并截断到 >= 0。"""
    return np.maximum(np.expm1(x), 0.0) / 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--test-year", type=int, default=M.splits["test"][0])
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--out", default="runs/exp/bcsd-both-spaces")
    a = p.parse_args()

    var = PRECIP
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    df_te = M.find_year_files(a.daymet_dir, a.test_year)
    mask = _squeeze_2d(_load_raw_static(df_te, M.LAND_MASK_VAR)) > 0.5
    print(f"[bcsd2] var={var} 陆地像素={int(mask.sum()):,} / {mask.size:,}", flush=True)

    # ---- 1. 逐 HR 像素最小二乘 (log1p 空间), 38 年 ----
    up = None
    Sx = np.zeros(mask.shape, np.float64); Sy = Sx.copy(); Sxx = Sx.copy(); Sxy = Sx.copy()
    ntr = 0
    for ty in a.train_years:
        ef = M.find_year_files(a.era5_dir, ty); df = M.find_year_files(a.daymet_dir, ty)
        lr = M.load_var_stack(ef, var); hrmm = npz_memmap(df[0], var)
        if up is None:
            up = make_bilinear(lr.shape[1], lr.shape[2], FACTOR)
        lrf = fill_nan_daymean(fwd(lr))
        for t in range(0, lr.shape[0], a.train_stride):
            x = up(lrf[t])
            y = fwd(hr_day(hrmm, t))
            Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; ntr += 1
        print(f"  拟合 {ty}  累计 {ntr} 天  ({time.time()-t0:.0f}s)", flush=True)

    den = ntr * Sxx - Sx * Sx
    coef_a = np.where(den != 0, (ntr * Sxy - Sx * Sy) / den, 1.0).astype(np.float32)
    coef_b = np.where(den != 0, (Sy - coef_a.astype(np.float64) * Sx) / max(ntr, 1), 0.0).astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "bcsd_precip_coef.npz"), a=coef_a, b=coef_b,
                        n_train_days=ntr, space="log1p(mm)")
    print(f"[bcsd2] 拟合完成 n={ntr} 天; 系数已存 (以后不必重拟合)", flush=True)

    # ---- 2. test_year 上同时在两个空间评测 ----
    ef = M.find_year_files(a.era5_dir, a.test_year)
    lr = M.load_var_stack(ef, var)                      # 原生 m/day
    hrmm = npz_memmap(df_te[0], var)
    lrf_log = fill_nan_daymean(fwd(lr))                 # log 空间的 ERA5
    lrf_raw = fill_nan_daymean(lr.astype(np.float32))   # 原生空间的 ERA5
    T = lr.shape[0]

    LOG = {m: Acc() for m in ("bilinear", "bicubic", "bcsd")}
    RAW = {m: Acc() for m in ("bilinear_raw", "bilinear_viaLog", "bicubic_raw", "bcsd")}

    for t in range(T):
        y_raw = hr_day(hrmm, t)                        # 真值: 原生 m/day
        y_log = fwd(y_raw)                              # 真值: log1p(mm)
        mk_t, mk_r = y_log[mask], y_raw[mask]

        x_log = up(lrf_log[t])                          # log 空间上采样
        x_raw = up(lrf_raw[t])                          # 原生空间上采样
        bcsd_log = coef_a * x_log + coef_b              # BCSD 预测(log 空间)

        # --- log1p(mm) 空间 (与原实验同口径) ---
        LOG["bilinear"].add(x_log[mask], mk_t)
        LOG["bcsd"].add(bcsd_log[mask], mk_t)
        if _zoom is not None:
            LOG["bicubic"].add(_bicubic(lrf_log[t])[mask], mk_t)

        # --- m/day 空间 ---
        RAW["bilinear_raw"].add(x_raw[mask], mk_r)                 # 直接上采样原生量(= eval_all_methods 口径)
        RAW["bilinear_viaLog"].add(inv(x_log)[mask], mk_r)         # 先 log 再上采样再反变换(BCSD 管线口径)
        RAW["bcsd"].add(inv(bcsd_log)[mask], mk_r)                 # ★BCSD 反变换回 m/day
        if _zoom is not None:
            RAW["bicubic_raw"].add(_bicubic(lrf_raw[t])[mask], mk_r)

    res = {
        "log1p(mm)": {m: LOG[m].result() for m in LOG if LOG[m].n},
        "m/day":     {m: RAW[m].result() for m in RAW if RAW[m].n},
        "n_test_days": int(T), "n_train_days": int(ntr),
        "train_years": [int(y) for y in a.train_years], "test_year": int(a.test_year),
    }
    json.dump(res, open(os.path.join(a.out, "metrics_both_spaces.json"), "w"),
              indent=2, ensure_ascii=False, default=float)

    # ---- 3. 自检: 两个已知数字必须复现, 否则反变换口径有问题 ----
    print("\n================ 结果 ================")
    for space in ("log1p(mm)", "m/day"):
        print(f"\n[{space}]")
        print(f"  {'方法':<18} {'RMSE':>10} {'MAE':>10} {'bias':>10} {'corr':>8}")
        for m, r in res[space].items():
            print(f"  {m:<18} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['bias']:>10.4f} {r['corr']:>8.4f}")

    print("\n================ 自检 ================")
    ok = True
    c1 = res["log1p(mm)"]["bcsd"]["rmse"]
    hit1 = abs(c1 - 0.5453) < 0.002
    ok &= hit1
    print(f"  log 空间 BCSD RMSE = {c1:.4f}   期望 0.5453 (原实验)   {'✓ 复现' if hit1 else '✗ 对不上'}")
    c2 = res["m/day"]["bilinear_raw"]["rmse"]
    hit2 = abs(c2 - 0.0046) < 0.0002
    ok &= hit2
    print(f"  m/day  bilinear RMSE = {c2:.4f}   期望 0.0046 (eval_all_methods)   {'✓ 复现' if hit2 else '✗ 对不上'}")
    print(f"\n  => {'两个自检都通过, BCSD 的 m/day 数字可以与 UNet/ViT 并排' if ok else '★自检未过, 不要使用这些数字'}")
    print(f"\n-> {a.out}/metrics_both_spaces.json   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
