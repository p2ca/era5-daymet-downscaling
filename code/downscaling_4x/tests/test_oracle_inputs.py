#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daymet-oracle 输入目录必须与合同取数逐项吻合。

oracle 目录把 ERA5 年度归档里的三个目标成员换成同日 Daymet 的陆地 4x4 块平均, 其余成员
不动。本测试用合同的取数类分别从 oracle 目录与真实目录取数, 核对:

  * 三个替换变量 == 我们自己读到的高分辨率真值经陆地掩膜的 4x4 块平均(float32 舍入内);
  * 其余输入变量逐位等于真实 ERA5;
  * 有效域掩膜、ERA5 valid_mask、四个静态通道(含 Δz)逐位不变;
  * 归一化统计: 未动变量相同, 替换变量确实不同, Daymet 目标统计相同。

抽查覆盖闰年测试年的闰日前后与年首年尾: 两侧的日历约定若不同, 错位会从闰日起出现。
需真实数据, 纯 CPU, 约 2 分钟。

运行: python -m downscaling_4x.tests.test_oracle_inputs
环境变量 ORACLE_ERA5_DIR 可指向别的 oracle 目录。
"""
import os

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.evaluation import input_identity as II

ORACLE = os.environ.get(
    "ORACLE_ERA5_DIR",
    "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_DaymetOracle_4x")
YEARS = [2017, 2020]
SAMPLE_DAYS = {2017: [0, 100, 364], 2020: [0, 1, 58, 59, 60, 181, 363, 364]}
TOL = {C.TARGETS[0]: 2e-3, C.TARGETS[1]: 2e-3, C.PRECIP: 2e-6}   # K, K, m/day


def build():
    s_r, s_o = Stats(M.ERA5_DIR, M.DAYMET_DIR), Stats(ORACLE, M.DAYMET_DIR)
    d_r = DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, YEARS, s_r, mode="history_51",
                        era5_cache_years=len(YEARS))
    d_o = DownscaleData(ORACLE, M.DAYMET_DIR, YEARS, s_o, mode="history_51",
                        era5_cache_years=len(YEARS))
    return s_r, s_o, d_r, d_o


def test_identity(ctx):
    d = II.describe(ORACLE)
    assert d["input_product"] == II.PRODUCT_ORACLE, d
    assert list(d["oracle"]["replaced_variables"]) == list(C.TARGETS)
    assert II.describe(M.ERA5_DIR)["input_product"] == II.PRODUCT_ERA5


def test_stats(ctx):
    s_r, s_o, _, _ = ctx
    for i, v in enumerate(C.ERA5_IN):
        same = s_r.e_mean[i] == s_o.e_mean[i] and s_r.e_std[i] == s_o.e_std[i]
        if v in C.TARGETS:
            assert not same, f"{v} 是替换变量, 统计量应当不同"
        else:
            assert same, f"{v} 未被替换, 统计量应当相同"
    assert np.array_equal(s_r.d_mean, s_o.d_mean) and np.array_equal(s_r.d_std, s_o.d_std)
    assert s_r.static_mean == s_o.static_mean and s_r.static_std == s_o.static_std


def test_mask_and_static_unchanged(ctx):
    _, _, d_r, d_o = ctx
    assert np.array_equal(d_r.mask, d_o.mask) and d_o.mask.sum() > 0
    assert np.array_equal(d_r.era5_valid, d_o.era5_valid)
    for k in C.STATIC_ORDER:
        assert np.array_equal(d_r._static[k], d_o._static[k]), f"静态通道 {k} 不同"


def test_untouched_vars_bitwise(ctx):
    _, _, d_r, d_o = ctx
    for y in YEARS:
        yr_r, yr_o = d_r._era5_year(y), d_o._era5_year(y)
        for v in C.ERA5_IN:
            if v in C.TARGETS:
                continue
            assert np.array_equal(yr_r[v], yr_o[v]), f"{y} {v} 与真实 ERA5 不一致"


def test_replaced_vars_differ_from_real(ctx):
    _, _, d_r, d_o = ctx
    vm = d_o.era5_valid
    for y in YEARS:
        yr_r, yr_o = d_r._era5_year(y), d_o._era5_year(y)
        for v in C.TARGETS:
            diff = float(np.abs(yr_r[v] - yr_o[v])[:, vm].max())
            floor = 1.0 if v != C.PRECIP else 1e-2
            assert diff > floor, f"{y} {v}: oracle 与真实 ERA5 几乎相同 (max|diff|={diff}), 不像被替换过"


def test_replaced_vars_are_land_block_means(ctx):
    _, _, _, d_o = ctx
    f = C.FACTOR
    Hl, Wl = C.LR_SHAPE
    land = d_o.daymet_land
    cnt = land.reshape(Hl, f, Wl, f).sum(axis=(1, 3))
    direct = d_o.era5_valid & (cnt > 0)
    assert direct.sum() > 0.9 * d_o.era5_valid.sum(), "几乎没有可直接对拍的粗格, 测试无分辨力"
    for y in YEARS:
        yr_o = d_o._era5_year(y)
        for v in C.TARGETS:
            worst = 0.0
            for t in SAMPLE_DAYS[y]:
                hr = d_o._hr_day(y, t, v).astype(np.float64)
                if v == C.PRECIP:
                    hr = np.maximum(hr, 0.0)
                hr = np.where(land, hr, 0.0)
                bm = hr.reshape(Hl, f, Wl, f).sum(axis=(1, 3)) / np.maximum(cnt, 1)
                worst = max(worst, float(np.abs(yr_o[v][t].astype(np.float64) - bm)[direct].max()))
            assert worst < TOL[v], f"{y} {v}: oracle 与真值块平均最大差 {worst:.3e} 超过 {TOL[v]}"


TESTS = [test_identity, test_stats, test_mask_and_static_unchanged,
         test_untouched_vars_bitwise, test_replaced_vars_differ_from_real,
         test_replaced_vars_are_land_block_means]


def main():
    ctx = build()
    for fn in TESTS:
        fn(ctx)
        print(f"✓ {fn.__name__}")
    print("test_oracle_inputs: all passed")


if __name__ == "__main__":
    main()
