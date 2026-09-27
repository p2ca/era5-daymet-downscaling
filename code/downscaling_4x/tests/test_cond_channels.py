#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在真实数据上逐槽位核对条件张量: 每个通道装的东西必须和它的名字一致。

只校验通道数抓不到顺序错位 —— 51 通道里 30 个历史通道彼此同构, 换个位置数量照样对得上,
训练照常收敛。这里对每个槽位独立重算一遍期望值再逐点比对: 独立重算不走取数模块的
布局与索引逻辑, 只共用上采样与降水变换这两个定义本身。

需要真实数据, 纯 CPU。

运行: python -m downscaling_4x.tests.test_cond_channels
"""
import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.downscale_baseline import (fill_nearest_valid, make_bilinear,
                                                    nearest_valid_index)
from downscaling_4x.data.frames import FrameIndex

YEARS = [2019, 2020]
MODE = "history_51"


def _raw_era5_day(year, day, var):
    """绕开取数模块, 直接从年度 npz 取一天并上采样归一化。"""
    a = M.load_var_stack(M.find_year_files(M.ERA5_DIR, year), var)
    idx = nearest_valid_index(M.load_static_2d(M.ERA5_DIR, "valid_mask") > 0.5)
    x = make_bilinear(*C.LR_SHAPE, C.FACTOR)(fill_nearest_valid(a, idx)[day])
    if var == C.PRECIP:
        x = C.precip_fwd(x)
    i = C.ERA5_IN.index(var)
    s = Stats()
    return (x - s.e_mean[i]) / s.e_std[i]


def build():
    s = Stats()
    ds = DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, YEARS, s, mode=MODE)
    fi = FrameIndex([2020], YEARS, history_days=C.pairing_history_days(MODE),
                    lags=C.history_lags(MODE), split="test")
    return s, ds, fi


def test_shape_and_layout(ctx):
    s, ds, fi = ctx
    i = fi.frames.index((2020, 5))
    cond, tgt, mask, raw = ds.full(2020, 5, fi.history_of(i))
    assert cond.shape == (51, 480, 960), cond.shape
    assert tgt.shape == (3, 480, 960) and raw.shape == (3, 480, 960)
    assert mask.shape == (1, 480, 960)
    assert np.isfinite(cond).all(), "条件张量不允许有 NaN"
    G.check_domain(M.DAYMET_DIR, hr_shape=cond.shape)


def test_era5_slots_match_independent_recompute(ctx):
    s, ds, fi = ctx
    day = 5
    i = fi.frames.index((2020, day))
    fields = ds.cond_fields(2020, day, fi.history_of(i))
    for var in ("2m_temperature_max", "total_precipitation_24hr", "u_component_of_wind_500"):
        want = _raw_era5_day(2020, day, var)
        got = fields[var]
        assert np.allclose(got, want, atol=1e-5), \
            f"{var} 槽位内容不符, 最大差 {np.abs(got - want).max()}"


def test_wind_slots_are_not_swapped(ctx):
    """850 hPa 的 u/v 同量级同分布, 互换不会被数量守卫发现, 必须逐点比。"""
    s, ds, fi = ctx
    i = fi.frames.index((2020, 5))
    f = ds.cond_fields(2020, 5, fi.history_of(i))
    u = _raw_era5_day(2020, 5, "u_component_of_wind_850")
    v = _raw_era5_day(2020, 5, "v_component_of_wind_850")
    assert np.allclose(f["u_component_of_wind_850"], u, atol=1e-5)
    assert np.allclose(f["v_component_of_wind_850"], v, atol=1e-5)
    assert not np.allclose(u, v, atol=1e-3), "u 与 v 本就不同, 否则本测试没有分辨力"
    for bad in ("10m_u_component_of_wind", "10m_v_component_of_wind"):
        assert bad not in f, f"{bad} 不在本合同里, 不应出现在条件张量"


def test_static_slots(ctx):
    """Δz 必须是"高分辨率高程 − 上采样粗网格高程", 不是绝对高程; 且只除标准差。"""
    s, ds, fi = ctx
    i = fi.frames.index((2020, 5))
    f = ds.cond_fields(2020, 5, fi.history_of(i))
    oro_hr = M.load_static_2d(M.DAYMET_DIR, "orography")
    oro_lr = M.load_static_2d(M.ERA5_DIR, "orography")
    lc = M.load_static_2d(M.DAYMET_DIR, C.LANDCOVER)
    lsm = M.load_static_2d(M.DAYMET_DIR, C.LAND_SEA_MASK)
    land = ds.mask                      # 有效域, 不是 Daymet 陆地掩膜

    idx = nearest_valid_index(M.load_static_2d(M.ERA5_DIR, "valid_mask") > 0.5)
    up = make_bilinear(*C.LR_SHAPE, C.FACTOR)
    want_dz = (oro_hr - up(fill_nearest_valid(oro_lr, idx))) / s.static_std[C.DZ]
    got = f[C.DZ]
    assert np.allclose(got[land], want_dz[land], atol=1e-4), \
        f"Δz 槽位不符, 陆地上最大差 {np.abs(got[land] - want_dz[land]).max()}"
    assert s.static_mean[C.DZ] == 0.0, "Δz 不减均值"
    # 负对照: 同样只除标准差的**绝对高程**必须与 Δz 明显不同, 否则本测试分辨不出两者
    abs_elev = (oro_hr / s.static_std[C.DZ])
    assert not np.allclose(got[land], abs_elev[land], atol=1e-2), "Δz 不应等于绝对高程"
    assert abs(float(got[land].mean())) < abs(float(abs_elev[land].mean())), \
        "Δz 应大致零中心, 绝对高程不是"

    # 绝对高程: z-score, 与 Δz 同源不同用法, 两者绝不能相等
    want_el = (oro_hr - s.static_mean[C.ELEVATION]) / s.static_std[C.ELEVATION]
    assert np.allclose(f[C.ELEVATION][land], want_el[land], atol=1e-4)
    assert s.static_mean[C.ELEVATION] != 0.0, "绝对高程必须减均值"
    assert not np.allclose(f[C.ELEVATION][land], f[C.DZ][land], atol=1e-2), \
        "Δz 与绝对高程若相等, 说明有一个算错了"

    want_lc = (lc - s.static_mean[C.LANDCOVER]) / s.static_std[C.LANDCOVER]
    assert np.allclose(f[C.LANDCOVER][land], want_lc[land], atol=1e-5)

    for name in C.STATIC_ORDER:                      # 有效域外一律为 0
        assert np.all(f[name][~land] == 0.0), f"{name} 在有效域外不是 0"
    m = f[C.LAND_SEA_MASK]
    assert set(np.unique(m)) <= {0.0, 1.0}, "海陆掩膜必须是原值 0/1"
    assert np.array_equal(m > 0.5, land)


def test_effective_domain_is_input_and_target(ctx):
    """有效域必须是"目标有真值 且 输入有数据"的交集, 不是 Daymet 的陆地掩膜。

    直接拿 Daymet 陆地当有效域不会报错, 只会让三分之一的域在没有输入的情况下参与打分。
    """
    s, ds, fi = ctx
    assert np.array_equal(ds.mask, ds.daymet_land & ds.era5_valid_hr)
    assert ds.mask.sum() < ds.daymet_land.sum(), "交集没有真正生效, 本测试无分辨力"
    assert (ds.mask & ~ds.daymet_land).sum() == 0, "有效域不能超出目标覆盖范围"
    # 有效域内每一格都必须有真实(非外推)的 ERA5 输入
    vm = M.load_static_2d(M.ERA5_DIR, "valid_mask") > 0.5
    a = M.load_var_stack(M.find_year_files(M.ERA5_DIR, 2020), "2m_temperature_max")
    assert np.isfinite(a[:, vm]).all(), "ERA5 在 valid_mask 内不应有缺测"
    coarse = ds.mask.reshape(C.LR_SHAPE[0], C.FACTOR, C.LR_SHAPE[1], C.FACTOR).any((1, 3))
    assert not (coarse & ~vm).any(), "有效域落在了 ERA5 没有数据的粗格上"


def test_bilinear_matches_torch(ctx):
    """Δz 依赖上采样算子的口径; numpy 实现必须与 torch 的 align_corners=False 逐点一致,

    否则两侧算出的 Δz 会有亚像素级偏差, 而不会有任何东西报错。
    """
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    a = rng.standard_normal(C.LR_SHAPE).astype(np.float32)
    mine = make_bilinear(*C.LR_SHAPE, C.FACTOR)(a)
    theirs = F.interpolate(torch.from_numpy(a)[None, None], size=C.HR_SHAPE,
                           mode="bilinear", align_corners=False)[0, 0].numpy()
    d = float(np.abs(mine - theirs).max())
    assert d < 1e-5, f"上采样口径不一致, 最大差 {d}"


def test_calendar_slots(ctx):
    s, ds, fi = ctx
    for day in (0, 5, 200):
        i = fi.frames.index((2020, day))
        f = ds.cond_fields(2020, day, fi.history_of(i))
        sin_d, cos_d = C.doy_sincos(day)
        assert np.allclose(f[C.DOY_SIN], sin_d) and np.allclose(f[C.DOY_COS], cos_d)
        assert f[C.DOY_SIN].shape == (480, 960), "时间通道是空间常数场, 但形状要铺满"


def test_history_slots_hold_the_right_days(ctx):
    """跨年那一帧最能暴露问题: 2020 年第 0 天的历史必须落在 2019 年末。"""
    s, ds, fi = ctx
    for day in (0, 1, 5):
        i = fi.frames.index((2020, day))
        hist = fi.history_of(i)
        f = ds.cond_fields(2020, day, hist)
        for lag, (hy, hd) in zip(C.HISTORY_LAGS, hist):
            for var in ("2m_temperature_max", "total_precipitation_24hr"):
                want = _raw_era5_day(hy, hd, var)
                got = f[C.history_name(lag, var)]
                assert np.allclose(got, want, atol=1e-5), \
                    f"day={day} lag={lag} {var} 取错帧, 最大差 {np.abs(got - want).max()}"
    i0 = fi.frames.index((2020, 0))
    assert fi.history_of(i0) == ((2019, 363), (2019, 364)), fi.history_of(i0)


def test_history_differs_from_today(ctx):
    """历史通道若与当天同值, 说明根本没取到历史; 温度日间变化足以区分。"""
    s, ds, fi = ctx
    i = fi.frames.index((2020, 5))
    f = ds.cond_fields(2020, 5, fi.history_of(i))
    today = f["2m_temperature_max"]
    for lag in C.HISTORY_LAGS:
        past = f[C.history_name(lag, "2m_temperature_max")]
        assert not np.allclose(today, past, atol=1e-3), f"lag={lag} 的历史与当天相同"


def test_history_never_reads_target_product(ctx):
    """历史段只能由 ERA5 变量构成, 不允许出现任何目标产品通道。"""
    s, ds, fi = ctx
    hist_names = [n for n in C.cond_layout(MODE) if n.startswith(C.HISTORY_PREFIX)]
    assert len(hist_names) == 30
    for n in hist_names:
        lag, var = n[len(C.HISTORY_PREFIX):].split(":", 1)
        assert int(lag) in C.HISTORY_LAGS
        assert var in C.ERA5_IN, f"{n} 不是 ERA5 变量"


def test_assemble_rejects_wrong_order(ctx):
    """把两个历史通道对调, 通道数不变, assemble 必须当场拒绝。"""
    s, ds, fi = ctx
    i = fi.frames.index((2020, 5))
    f = ds.cond_fields(2020, 5, fi.history_of(i))
    assert ds.assemble(f).shape[0] == 51
    keys = list(f)
    a, b = keys.index(C.history_name(2, "temperature_500")), keys.index(C.history_name(1, "temperature_500"))
    keys[a], keys[b] = keys[b], keys[a]
    from collections import OrderedDict
    swapped = OrderedDict((k, f[k]) for k in keys)
    assert len(swapped) == len(f), "对调后通道数不变, 数量守卫抓不到"
    try:
        ds.assemble(swapped)
    except RuntimeError:
        return
    raise AssertionError("通道顺序错位没有被拒绝")


def test_target_and_mask(ctx):
    s, ds, fi = ctx
    norm, raw = ds.target(2020, 5)
    ip = C.TARGETS.index(C.PRECIP)
    want = (C.precip_fwd(raw[ip]) - s.d_mean[ip]) / s.d_std[ip]
    assert np.allclose(norm[ip], want, atol=1e-5), "降水目标必须走同一条正变换"
    it = C.TARGETS.index("2m_temperature_max")
    assert np.allclose(norm[it], (raw[it] - s.d_mean[it]) / s.d_std[it], atol=1e-5)
    assert ds.mask.dtype == bool and 0.3 < ds.mask.mean() < 0.95


def main():
    ctx = build()
    tests = [
        test_shape_and_layout,
        test_era5_slots_match_independent_recompute,
        test_wind_slots_are_not_swapped,
        test_static_slots,
        test_effective_domain_is_input_and_target,
        test_bilinear_matches_torch,
        test_calendar_slots,
        test_history_slots_hold_the_right_days,
        test_history_differs_from_today,
        test_history_never_reads_target_product,
        test_assemble_rejects_wrong_order,
        test_target_and_mask,
    ]
    for t in tests:
        t(ctx)
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
