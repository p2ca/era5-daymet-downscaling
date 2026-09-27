#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
eval_baselines.py — 统计基线在测试年的得分
============================================================================
两个方法, 都是确定性单成员(于是 CRPS 恒等于 MAE, 由公共评测自动成立):

  bilinear  ERA5 双线性上采样到 HR, 不做任何订正
  bcsd      逐像素 a * bilinear(ERA5) + b, 系数由 fit_bcsd_coefs 在训练年拟合

预测一律转回**物理单位**再交给公共评测: 温度是 K, 降水是数据集原生的 m/day。降水的
BCSD 在 log1p(mm) 空间里线性, 因此先在该空间算完再用合同的 `precip_inv` 反变换 ——
反变换前会钳到 log1p <= 8, 免得个别像素的外推被 expm1 放大成天文数字。

降水的指标同时给两套空间: 物理空间与 log1p(mm)(带 `_log` 后缀)。两者的 RMSE 之间没有
换算关系, 排名甚至相反, 所以不能只报一个。

运行: python -m downscaling_4x.baselines.eval_baselines --coefs ... --out ...
============================================================================
"""
import argparse
import json
import os
import time

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.downscale_baseline import (fill_nearest_valid, make_bilinear,
                                                    nearest_valid_index)
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.eval_common import MultiMethodEval

METHODS = ["bilinear", "bcsd"]


def bcsd_predict(x_hr, a, b, is_precip, clip_mm=C.PRECIP_CLIP_MM, scale=C.PRECIP_SCALE):
    """BCSD 逐像素订正: x_hr 是已上采样到 HR 的物理量(温度 K / 降水 m/day)。

    降水在合同的 log1p(mm) 空间里线性(含 <clip_mm 置零), 算完再用 precip_inv 变回 m/day;
    变换必须在上采样★之后★做, 与系数拟合同一顺序 —— log1p 与双线性不可交换。
    落场(det_dump)与本模块的评测都调用这一个函数, 两条路径的 BCSD 才是同一个预测。
    """
    if is_precip:
        return C.precip_inv(a * C.precip_fwd(x_hr, clip_mm, scale) + b, scale=scale)
    return a * x_hr + b


def load_coefs(coef_dir, era5_dir=M.ERA5_DIR, allow_input_mismatch=False):
    """读三份系数; 系数记录的拟合输入目录必须与本次评测的输入目录一致。"""
    out = {}
    for v in C.TARGETS:
        p = os.path.join(coef_dir, f"{v}.npz")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺 BCSD 系数 {p}")
        z = np.load(p, allow_pickle=False)
        II.check(str(z["era5_dir"]) if "era5_dir" in z.files else None, era5_dir,
                 f"BCSD 系数 {p}", allow=allow_input_mismatch)
        if tuple(z["hr_shape"]) != C.HR_SHAPE:
            raise ValueError(f"{p} 的网格 {tuple(z['hr_shape'])} 与合同 {C.HR_SHAPE} 不符")
        want = "log1p(mm)" if v == C.PRECIP else "K"
        if str(z["space"]) != want:
            raise ValueError(f"{p} 的拟合空间 {z['space']} 与预期 {want} 不符")
        out[v] = (z["a"].astype(np.float32), z["b"].astype(np.float32),
                  int(z["n_train_days"]))
    return out


def main():
    ap = argparse.ArgumentParser(description="统计基线评测")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--coefs", required=True, help="fit_bcsd_coefs 的输出目录")
    ap.add_argument("--year", type=int, default=M.splits["test"][0])
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--restrict-mask", default="",
                    help="npz(键 mask)给出的子域布尔掩膜, 与陆地掩膜取交后评测; "
                         "用于只在某个子区域(如 CONUS)报数, 便于与别处的口径对齐")
    ap.add_argument("--tag", default="baselines")
    II.add_arg(ap)
    a = ap.parse_args()

    G.check_domain(a.daymet_dir)
    coefs = load_coefs(a.coefs, a.era5_dir, getattr(a, II.ALLOW_DEST, False))
    stats = Stats(a.era5_dir, a.daymet_dir)
    ds = DownscaleData(a.era5_dir, a.daymet_dir, [a.year], stats, mode="baseline_21")
    H, W = ds.H, ds.W
    land = ds.mask                       # 有效域 = 目标有真值 且 输入有数据
    n_eff, n_daymet = int(ds.mask.sum()), int(ds.daymet_land.sum())
    print(f"[域] Daymet 陆地 {n_daymet} 格 -> 有效域 {n_eff} 格 "
          f"({100 * n_eff / n_daymet:.1f}%); 差额是有目标但无 ERA5 输入的区域", flush=True)
    if a.restrict_mask:
        sub = np.load(a.restrict_mask, allow_pickle=False)["mask"].astype(bool)
        if sub.shape != (H, W):
            raise ValueError(f"子域掩膜形状 {sub.shape} 与网格 {(H, W)} 不符")
        land = land & sub
        print(f"[子域] 限制到 {a.restrict_mask}: 陆地格点 {ds.mask.sum()} -> {land.sum()} "
              f"({100 * land.sum() / max(ds.mask.sum(), 1):.1f}%)", flush=True)
    ds.mask = land
    up = make_bilinear(*C.LR_SHAPE, C.FACTOR)

    lr = {}
    fill_idx = nearest_valid_index(M.load_static_2d(a.era5_dir, "valid_mask") > 0.5)
    ef = M.find_year_files(a.era5_dir, a.year)
    for v in C.TARGETS:
        s = M.load_var_stack(ef, v)
        if s is None:
            raise KeyError(f"{a.year} 的 ERA5 缺 {v}")
        lr[v] = fill_nearest_valid(s, fill_idx)
    n_days = ds.ndays[a.year]

    ev = MultiMethodEval(METHODS, C.TARGETS, H, W, ds.mask,
                         precip_scale=stats.precip_scale, precip_log=stats.precip_log)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    days = list(range(0, n_days, a.eval_stride))
    for k, t in enumerate(days):
        _, raw = ds.target(a.year, t)                       # (3,H,W) 物理量真值
        bil = np.empty((len(C.TARGETS), H, W), np.float32)
        bcs = np.empty_like(bil)
        for i, v in enumerate(C.TARGETS):
            x = up(lr[v][t])                                # 物理单位的上采样场
            bil[i] = x
            ca, cb, _ = coefs[v]
            bcs[i] = bcsd_predict(x, ca, cb, v == C.PRECIP, stats.precip_clip, stats.precip_scale)
        ev.add_day(raw, ds.mask[None].astype(np.float32),
                   {"bilinear": bil[None], "bcsd": bcs[None]})
        if k % 60 == 0:
            print(f"  第 {k + 1}/{len(days)} 天 ({time.time() - t0:.0f}s)", flush=True)

    res = ev.finalize(a.out, a.year, eval_stride=a.eval_stride, tag=a.tag,
                      n_total_days=n_days, make_plots=not a.no_plots, make_maps=not a.no_plots)
    side = {"contract": {"mode_for_targets": "baseline_21", "factor": C.FACTOR,
                         "hr_shape": list(C.HR_SHAPE), "lr_shape": list(C.LR_SHAPE),
                         "domain_lat": list(C.DOMAIN_LAT), "domain_lon": list(C.DOMAIN_LON)},
            "bcsd_train_days": {v: coefs[v][2] for v in C.TARGETS},
            "precip_units": {"physical": "m/day (数据集原生)", "log": "log1p(mm/day)"},
            "eval_domain": {"rule": C.EFFECTIVE_DOMAIN,
                            "daymet_land_cells": n_daymet,
                            "effective_cells": n_eff,
                            "restrict_mask": a.restrict_mask or None,
                            "cells_used": int(land.sum())},
            "input": II.describe(a.era5_dir),
            "era5_dir": a.era5_dir, "daymet_dir": a.daymet_dir, "coefs": a.coefs}
    json.dump(side, open(os.path.join(a.out, f"run_context_{a.tag}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"完成 {len(days)} 天 ({time.time() - t0:.0f}s) -> {a.out}")
    return res


if __name__ == "__main__":
    main()
