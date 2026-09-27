#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 4x 跨产品合同: 通道布局、三档模式、倍率与网格、降水管线、年份划分、域范围。

期望值在本文件里逐条写死, 与 contract 模块互为对照。合同要改, 先改这里再改代码,
两处同时改动才通得过 —— 单侧改动会当场失败。

只依赖 contract / match / grid, 不引入 torch。

运行: python -m downscaling_4x.tests.test_spec_contract
"""
import math

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M

ERA5_INPUT_VARS = [
    "2m_temperature", "2m_temperature_max", "2m_temperature_min",
    "total_precipitation_24hr",
    "volumetric_soil_water_layer_1",
    "geopotential_500", "geopotential_850",
    "specific_humidity_500", "specific_humidity_850",
    "temperature_500", "temperature_850",
    "u_component_of_wind_500", "u_component_of_wind_850",
    "v_component_of_wind_500", "v_component_of_wind_850",
]
STATIC_CHANNELS = ["dz", "elevation", "landcover", "land_sea_mask"]
TIME_CHANNELS = ["doy_sin", "doy_cos"]
BASE_LAYOUT = ERA5_INPUT_VARS + STATIC_CHANNELS + TIME_CHANNELS
MODE_WIDTH = {"baseline_21": 21, "history_control_21": 21, "history_51": 51}


def test_era5_vars():
    assert list(C.ERA5_IN) == ERA5_INPUT_VARS, C.ERA5_IN
    assert len(C.ERA5_IN) == 15
    assert len(set(C.ERA5_IN)) == 15, "ERA5 变量名不能重复"


def test_near_surface_wind_excluded():
    """数据集里有 10m 风, 但本合同不取: 与高空风信息重叠, 且要与上一代当天段逐通道一致。"""
    for v in ("10m_u_component_of_wind", "10m_v_component_of_wind"):
        assert v not in C.ERA5_IN
        assert v not in C.cond_layout("history_51"), f"{v} 不应出现在任何段, 历史段也不行"


def test_static_roster():
    assert list(C.STATIC_ORDER) == STATIC_CHANNELS, C.STATIC_ORDER
    assert set(C.STATIC_NORM) == set(C.STATIC_ORDER), "每个静态通道都要登记归一化方式"
    # Δz 是 HR 高程与上采样 LR 高程之差, 只除标准差不减均值: 填 0 才对应"高分辨率地形
    # 等于粗网格地形"。若改成 z-score, 填 0 的物理含义就没了, 而不会有任何东西报错。
    assert C.STATIC_NORM[C.DZ] == ("scale", "orography"), C.STATIC_NORM[C.DZ]
    # 绝对高程与 Δz 共用 orography 统计, 但必须 z-score: 均值约 800 m, 只除标准差会给
    # 整幅压一个常数偏置。两者用法不同却同源, 是最容易写混的一处。
    assert C.STATIC_NORM[C.ELEVATION] == ("zscore", "orography"), C.STATIC_NORM[C.ELEVATION]
    assert C.STATIC_NORM[C.DZ][1] == C.STATIC_NORM[C.ELEVATION][1] == "orography"
    assert C.STATIC_NORM[C.DZ][0] != C.STATIC_NORM[C.ELEVATION][0]
    assert C.STATIC_NORM[C.LANDCOVER] == ("zscore", "landcover")
    assert C.STATIC_NORM[C.LAND_SEA_MASK][0] == "raw", "海陆掩膜是 0/1, 不做 z-score"
    assert C.STATIC_OCEAN_FILL == 0.0


def test_no_position_channels():
    """不注入 x/y 位置平面: 主干自带位置编码, 从输入端再来一份是重复信息。"""
    for m in C.MODES:
        names = C.cond_layout(m)
        assert not any("coordinate" in n or n in ("x", "y") for n in names), names
    assert not hasattr(C, "coord_planes"), "位置平面的构造函数也应移除, 免得被误用"


def test_time_channel_order():
    assert list(C.TIME_ORDER) == TIME_CHANNELS
    layout = C.cond_layout("baseline_21")
    assert layout.index(C.DOY_SIN) < layout.index(C.DOY_COS)
    assert layout.index(C.STATIC_ORDER[-1]) < layout.index(C.DOY_SIN), "时间通道排在静态之后"


def test_mode_widths():
    for mode, width in MODE_WIDTH.items():
        assert C.cond_channels(mode) == width, (mode, C.cond_channels(mode))
    assert set(C.MODES) == set(MODE_WIDTH)
    assert C.cond_layout("baseline_21") == BASE_LAYOUT
    assert C.cond_layout("baseline_21") == C.cond_layout("history_control_21"), \
        "control 与 baseline 的通道完全相同, 差别只在帧集合"


def test_history_block_order():
    layout = C.cond_layout("history_51")
    assert layout[:21] == BASE_LAYOUT
    assert C.HISTORY_LAGS == (2, 1), "历史顺序固定旧->新"
    expect = [C.history_name(lag, v) for lag in (2, 1) for v in ERA5_INPUT_VARS]
    assert layout[21:] == expect
    assert len(layout[21:]) == 30
    assert len(set(layout)) == len(layout), "通道名不能重复"


def test_control_shares_pairing_window():
    """control 与 58 通道模式必须要求同样多的历史天数, 否则帧集合不同, 增量无法归因。"""
    assert C.pairing_history_days("history_control_21") == 2
    assert C.pairing_history_days("history_51") == 2
    assert C.pairing_history_days("baseline_21") == 0
    assert C.history_lags("history_control_21") == ()


def test_scale_and_grid():
    assert C.FACTOR == 4
    assert C.LR_SHAPE == (120, 240)
    assert C.HR_SHAPE == (480, 960)
    C.check_shapes(C.LR_SHAPE, C.HR_SHAPE)
    for bad in [((120, 240), (720, 1440)), ((120, 240), (481, 960))]:
        try:
            C.check_shapes(*bad)
        except ValueError:
            continue
        raise AssertionError(f"形状 {bad} 本应被拒绝")


def test_domain_matches_grid():
    """域范围要能被网格整除成 FACTOR 对齐的单元, 且经纬跨度与格数自洽。"""
    dlat = (C.DOMAIN_LAT[1] - C.DOMAIN_LAT[0]) / C.HR_SHAPE[0]
    dlon = (C.DOMAIN_LON[1] - C.DOMAIN_LON[0]) / C.HR_SHAPE[1]
    assert math.isclose(dlat, 30.0 / 480), dlat
    assert math.isclose(dlon, 60.0 / 960), dlon
    assert math.isclose(dlat * C.FACTOR, 0.25), "一个 ERA5 单元应恰好覆盖 FACTOR 个 Daymet 单元"
    assert math.isclose(dlon * C.FACTOR, 0.25)
    assert G.extent() == [C.DOMAIN_LON[0], C.DOMAIN_LON[1], C.DOMAIN_LAT[0], C.DOMAIN_LAT[1]]
    la, lo = G.cell_centers()
    assert la.shape == (480,) and lo.shape == (960,)
    assert math.isclose(float(la[0]), 24.0 + dlat / 2)


def test_base_segment_matches_previous_contract():
    """当天那 21 个通道必须与上一代合同逐通道一致 —— 本合同相对它的唯一增量是历史段。

    上一代包若不在(独立部署), 跳过这一条; 本文件写死的 BASE_LAYOUT 仍是主判据。
    """
    try:
        from era5_daymet import contract as OLD
    except Exception as e:
        print(f"    (跳过跨包对照: {e})")
        return
    old = OLD.cond_layout(OLD.DEFAULT_IN)
    new = C.cond_layout("baseline_21")
    assert new == old, [(i, a, b) for i, (a, b) in enumerate(zip(new, old)) if a != b]
    assert OLD.FACTOR != C.FACTOR, "两代的空间倍率本就不同, 别把这条当成合同等价"


def test_doy_sincos():
    assert C.DAYS_PER_YEAR == 365
    s0, c0 = C.doy_sincos(0)
    assert math.isclose(s0, 0.0, abs_tol=1e-12) and math.isclose(c0, 1.0)
    for d in range(C.DAYS_PER_YEAR):
        s, c = C.doy_sincos(d)
        assert math.isclose(s * s + c * c, 1.0, rel_tol=1e-9)
    last = C.doy_sincos(C.DAYS_PER_YEAR - 1)
    step = math.hypot(last[0] - s0, last[1] - c0)
    assert step < 2 * math.pi / C.DAYS_PER_YEAR * 1.01, "年末到年初不允许跳变"
    seen = {C.doy_sincos(d) for d in range(C.DAYS_PER_YEAR)}
    assert len(seen) == C.DAYS_PER_YEAR, "(sin,cos) 对 day index 必须是双射"


def test_targets_and_splits():
    assert C.TARGETS == ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
    assert C.PRECIP == "total_precipitation_24hr"
    assert M.splits["train"] == list(range(1980, 2018))
    assert M.splits["val"] == [2018, 2019]
    assert M.splits["test"] == [2020]
    assert M.DAYS_PER_YEAR == C.DAYS_PER_YEAR
    assert M.LEAP_DROP == C.LEAP_DROP == "dec31"


def test_precip_pipeline():
    assert C.PRECIP_SCALE == 1000.0 and C.PRECIP_CLIP_MM == 0.1 and C.PRECIP_LOG_MAX == 8.0
    x = np.array([0.0, 5e-8, 2e-4, 1e-2], np.float32)          # m/day
    got = C.precip_fwd(x)
    assert got[0] == 0.0
    assert got[1] == 0.0, "0.05 mm 属于毛毛雨, 必须置零"
    assert math.isclose(float(got[2]), math.log1p(0.2), rel_tol=1e-6)
    assert math.isclose(float(got[3]), math.log1p(10.0), rel_tol=1e-6)
    back = C.precip_inv(got)
    assert np.allclose(back[2:], x[2:], rtol=1e-5), (back, x)
    assert math.isclose(float(C.precip_inv(np.array([99.0], np.float32))[0]),
                        float(np.expm1(C.PRECIP_LOG_MAX) / C.PRECIP_SCALE),
                        rel_tol=1e-6), "反变换前必须先钳上界"
    assert float(C.precip_fwd(np.array([-1.0], np.float32))[0]) == 0.0, "负降水按 0 处理"


def test_calendar_leap_convention():
    """闰年丢 12/31; frame_date 与 calendar_365 必须逐帧一致。"""
    for y in (2019, 2020):
        cal = M.calendar_365(y)
        assert len(cal) == 365
        assert all(M.frame_date(y, i) == cal[i] for i in range(365))
    assert M.calendar_365(2020)[-1].month == 12 and M.calendar_365(2020)[-1].day == 30
    assert M.calendar_365(2019)[-1].day == 31
    assert any(d.month == 2 and d.day == 29 for d in M.calendar_365(2020)), "闰日保留"


def main():
    tests = [
        test_era5_vars,
        test_near_surface_wind_excluded,
        test_static_roster,
        test_no_position_channels,
        test_time_channel_order,
        test_mode_widths,
        test_history_block_order,
        test_control_shares_pairing_window,
        test_scale_and_grid,
        test_domain_matches_grid,
        test_base_segment_matches_previous_contract,
        test_doy_sincos,
        test_targets_and_splits,
        test_precip_pipeline,
        test_calendar_leap_convention,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
