#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_stage_b_sampling_stream.py — 阶段 B 取帧流的分片正确性
============================================================================
阶段 B 每步由全体 rank 各取一帧。若 index_offset 按 rank 号偏移 1(而非按每 rank 的
取样数分块), 相邻 rank 的取帧序列会几乎完全重合 —— 等效批量塌缩、训练集大半从未被用到,
而且不报错、不产生 NaN, 只会让结果悄悄变差。此处把该口径锁死。

用法:
    python -m era5_daymet.tests.test_stage_b_sampling_stream
============================================================================
"""
import numpy as np

from era5_daymet.training.train_stage_b import FrameStream

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


class _Frames:
    """只提供取帧所需的最小接口。"""

    def __init__(self, n):
        self.years = [1980]
        self.ndays = {1980: n}

    def full(self, y, t):
        raise AssertionError("本测试不应触碰真实数据")


def _stream(n_frames, world, steps, rank):
    return FrameStream(_Frames(n_frames), None, "2m_temperature_max", steps,
                       1234, rank * steps, world * steps)


def _seq(ds, steps, stride=1):
    out = []
    n = len(ds.frames)
    for i in range(0, steps, stride):
        g = ds.epoch * ds.epoch_span + ds.index_offset + i
        out.append(int(ds._perm_for(g // n)[g % n]))
    return out


def main():
    n, world, steps = 13870, 64, 900
    print(f"训练帧 {n:,} | rank {world} | 每 rank 取样 {steps:,}")

    seqs = {r: _seq(_stream(n, world, steps, r), steps) for r in range(world)}

    print("\n跨 rank 分片:")
    ov01 = len(set(seqs[0]) & set(seqs[1]))
    check("相邻 rank 不应大量重合", ov01 / steps < 0.15, f"共享 {ov01}/{steps}")
    allf = set().union(*seqs.values())
    total = world * steps
    expect = n * (1 - (1 - 1 / n) ** total)          # 有放回覆盖的期望
    check("全体 rank 覆盖训练集", len(allf) > 0.9 * min(n, expect),
          f"{len(allf):,}/{n:,}")

    print("\n单 rank 内无放回:")
    s0 = seqs[0]
    check("一遍数据内不重复", len(set(s0[:min(steps, n)])) == min(steps, n),
          f"{len(set(s0[:min(steps, n)]))}/{min(steps, n)}")

    print("\n曝光均匀性:")
    cnt = np.bincount(np.concatenate([np.array(v) for v in seqs.values()]), minlength=n)
    used = cnt[cnt > 0]
    check("每帧曝光次数至多差 1", used.max() - used.min() <= 1,
          f"min={used.min()} max={used.max()}")

    print("\n与 rank 号无关的确定性:")
    a = _seq(_stream(n, world, steps, 7), steps)
    b = _seq(_stream(n, world, steps, 7), steps)
    check("同参数两次构造给出同一序列", a == b)

    print("\n跨遍换置换:")
    small = 100
    ds = FrameStream(_Frames(small), None, "2m_temperature_max", 250, 1234, 0, 250)
    s = _seq(ds, 250)
    check("第 0 遍与第 1 遍顺序不同", s[:small] != s[small:2 * small])
    check("每遍各自无放回", len(set(s[:small])) == small and len(set(s[small:2 * small])) == small)

    print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
