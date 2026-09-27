#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定因果历史配对: 日期回查、缺历史丢帧、闰年边界、以及"年内减索引"的负对照。

历史帧取错是本管线代价最高的一类错误 —— 形状、通道数、量级全部正常, 训练照常收敛,
只有指标悄悄变差。所以这里不只验证"正确实现能过", 还显式构造错误实现并要求它**被判定为
与正确实现不同**: 只证明前者, 等于没证明这个测试有分辨力。

不依赖数据文件, 只用日历。

运行: python -m downscaling_4x.tests.test_causal_pairing
"""
from datetime import timedelta

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.frames import FrameIndex, assert_same_pairing

ALL_YEARS = list(range(1980, 2021))


def test_date_table_is_bijective():
    fi = FrameIndex([2020], ALL_YEARS, history_days=0, split="test")
    assert len(fi.date_to_frame) == len(ALL_YEARS) * 365
    for (y, d), key in ((v, k) for k, v in fi.date_to_frame.items()):
        assert M.frame_date(y, d) == key


def test_history_dates_are_exactly_previous_days():
    fi = FrameIndex([2019, 2020], ALL_YEARS, history_days=2,
                    lags=C.HISTORY_LAGS, split="val")
    assert len(fi) > 0
    for i, (y, d) in enumerate(fi.frames):
        cur = M.frame_date(y, d)
        got = [M.frame_date(*h) for h in fi.history_of(i)]
        assert got == [cur - timedelta(days=2), cur - timedelta(days=1)], (cur, got)


def test_leap_year_drops_two_frames_next_january():
    """2020 丢了 12/31, 所以 2021 的 1/1 与 1/2 取不到完整历史。"""
    fi = FrameIndex([2021], [2020, 2021], history_days=2, lags=C.HISTORY_LAGS)
    assert fi.dropped == 2, fi.dropped
    assert fi.frames[0] == (2021, 2), fi.frames[0]
    assert len(fi) == 363


def test_non_leap_predecessor_drops_nothing():
    fi = FrameIndex([2020], [2019, 2020], history_days=2, lags=C.HISTORY_LAGS)
    assert fi.dropped == 0 and len(fi) == 365
    assert fi.history_of(0) == ((2019, 363), (2019, 364)), fi.history_of(0)


def test_first_year_without_predecessor():
    fi = FrameIndex([2020], [2020], history_days=2, lags=C.HISTORY_LAGS)
    assert fi.dropped == 2 and len(fi) == 363
    assert fi.frames[0] == (2020, 2)


def test_no_history_keeps_every_frame():
    fi = FrameIndex([2020], ALL_YEARS, history_days=0)
    assert len(fi) == 365 and fi.dropped == 0
    assert all(h == () for h in fi.history)


def test_control_and_history_share_frame_set():
    """control 与 51 通道模式必须逐帧相同, 否则历史通道的增量无法归因。"""
    kw = dict(split_years=M.splits["train"], available_years=ALL_YEARS, split="train")
    ctrl = FrameIndex(history_days=C.pairing_history_days("history_control_21"),
                      lags=C.history_lags("history_control_21"), **kw)
    full = FrameIndex(history_days=C.pairing_history_days("history_51"),
                      lags=C.history_lags("history_51"), **kw)
    assert ctrl.frames == full.frames
    assert_same_pairing(ctrl, full)
    plain = FrameIndex(history_days=C.pairing_history_days("baseline_21"), **kw)
    assert len(plain) > len(full), "不要求历史的模式帧更多"
    try:
        assert_same_pairing(plain, full)
    except ValueError:
        return
    raise AssertionError("帧集合不同却没有被拒绝")


def test_negative_control_year_internal_index_arithmetic():
    """负对照: 在年度文件内直接减索引(必然 wrap 回同年年末)必须与正确配对不同。

    这正是要防的写法 —— 它不会报错, 只会在每年前两天取到十一个月之后的天气。
    """
    fi = FrameIndex([2020], [2019, 2020], history_days=2, lags=C.HISTORY_LAGS)
    wrong = {}
    for (y, d) in fi.frames:
        wrong[(y, d)] = tuple((y, (d - lag) % M.DAYS_PER_YEAR) for lag in C.HISTORY_LAGS)
    right = {f: fi.history_of(i) for i, f in enumerate(fi.frames)}
    diff = [f for f in fi.frames if wrong[f] != right[f]]
    assert diff, "负对照与正确实现必须存在分歧, 否则本测试没有分辨力"
    assert set(diff) == {(2020, 0), (2020, 1)}, diff
    assert right[(2020, 0)] == ((2019, 363), (2019, 364))
    assert wrong[(2020, 0)] == ((2020, 363), (2020, 364)), "错误实现应绕回同年年末"


def test_signature_changes_with_frame_set():
    a = FrameIndex([2020], ALL_YEARS, history_days=2, lags=C.HISTORY_LAGS, split="test")
    b = FrameIndex([2020], ALL_YEARS, history_days=2, lags=C.HISTORY_LAGS, split="test")
    c = FrameIndex([2020], [2020], history_days=2, lags=C.HISTORY_LAGS, split="test")
    assert a.signature() == b.signature(), "同样输入必须给同样签名"
    assert a.signature() != c.signature(), "可用年份不同导致帧集合不同, 签名必须变"
    meta = a.metadata()
    assert meta["policy"] == "real_calendar_dates_v1"
    assert meta["required_history_days"] == 2 and meta["history_lags"] == [2, 1]
    assert len(meta["frame_pairing_sha256"]) == 64


def test_invalid_configurations_are_rejected():
    for kw in (dict(history_days=-1),
               dict(history_days=2, lags=(0, 1)),
               dict(history_days=2, lags=(1, 1)),
               dict(history_days=1, lags=(2,))):
        try:
            FrameIndex([2020], ALL_YEARS, **kw)
        except ValueError:
            continue
        raise AssertionError(f"非法配置 {kw} 本应被拒绝")


def main():
    tests = [
        test_date_table_is_bijective,
        test_history_dates_are_exactly_previous_days,
        test_leap_year_drops_two_frames_next_january,
        test_non_leap_predecessor_drops_nothing,
        test_first_year_without_predecessor,
        test_no_history_keeps_every_frame,
        test_control_and_history_share_frame_set,
        test_negative_control_year_internal_index_arithmetic,
        test_signature_changes_with_frame_set,
        test_invalid_configurations_are_rejected,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
