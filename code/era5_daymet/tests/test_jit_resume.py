#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_jit_resume.py — train_jit 断点接力的逐位复现
============================================================================
同一份合成数据上, "一次跑完" 与 "中途保存断点 + 续训" 的损失曲线必须逐点相同
(取帧序列由全局序号决定 + RNG 状态随断点入盘), 终态 EMA 也必须一致。
稠密与 MoE 各验一遍(MoE 额外覆盖路由偏置 buffer 与负载记录的保存/恢复)。

Run: python -m era5_daymet.tests.test_jit_resume
============================================================================
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from era5_daymet.training import train_jit


class FakeData:
    """确定性合成整幅数据: 5 条件通道, 3 目标通道, 全陆地。"""

    def __init__(self, years):
        self.years = list(years)
        self.ndays = {y: (7 if y % 2 == 0 else 5) for y in self.years}
        self.H, self.W = 8, 12

    def full(self, y, t):
        rng = np.random.default_rng([y, t])
        cond = rng.standard_normal((5, self.H, self.W)).astype(np.float32)
        tgt = np.stack([0.5 * cond[0] + 0.1 * k for k in range(3)]).astype(np.float32)
        mask = np.ones((1, self.H, self.W), np.float32)
        return cond, tgt, mask, tgt


BASE = ["--target", "2m_temperature_max", "--cond-ch", "5",
        "--hidden", "32", "--depth", "2", "--heads", "2", "--patch", "4",
        "--bottleneck", "8", "--batch", "2", "--val-every", "8",
        "--ckpt-every", "8", "--warmup-samples", "8", "--noise-scale", "1.0",
        "--snap-every", "0", "--workers", "0", "--seed", "7", "--save-rng", "1"]
MOE = ["--moe", "--experts", "4", "--experts-per-tok", "2",
       "--moe-intermediate", "16"]


def run(out, duration, resume="", extra=()):
    argv = BASE + list(extra) + ["--out", str(out), "--duration", str(duration)]
    if resume:
        argv += ["--resume", str(resume)]
    train_jit.main(argv, data=(FakeData([2000, 2001]), FakeData([2002])))


def curve(out):
    hist = json.loads((Path(out) / "loss_history.json").read_text())
    return [(h["samples"], h["train"], h["val"]) for h in hist]


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def check_partition():
    """多 rank 取帧分片: 全局序号无重叠无遗漏, 每遍数据无放回覆盖全部帧。"""
    data = FakeData([2000, 2001])
    n = sum(data.ndays.values())
    world, steps, batch = 3, 8, 2                       # 共 48 样本 = 4 遍数据
    got = []
    for r in range(world):
        ds = train_jit.JitFrameStream(data, "2m_temperature_max", steps * batch,
                                      1234, (r * steps) * batch)
        got += [(ds.index_offset + i, ds.frame_of(i)) for i in range(len(ds))]
    gs = sorted(g for g, _ in got)
    check("多 rank 全局序号恰好铺满无重叠", gs == list(range(world * steps * batch)))
    full = sorted((y, t) for y in data.years for t in range(data.ndays[y]))
    passes = {}
    for g, fr in got:
        passes.setdefault(g // n, []).append(fr)
    check("每遍数据无放回覆盖全部帧",
          all(sorted(fr) == full for fr in passes.values()))


def main():
    check_partition()
    for label, extra in (("dense", ()), ("moe", MOE)):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a", Path(td) / "b"
            run(a, 48, extra=extra)                                 # 一次跑完
            run(b, 24, extra=extra)                                 # 前半
            run(b, 48, resume=b / "last.pt", extra=extra)           # 续后半
            ca, cb = curve(a), curve(b)
            check(f"[{label}] 曲线逐点一致 ({len(ca)} 点)", ca == cb and len(ca) >= 5)
            ea = torch.load(a / "last.pt", map_location="cpu", weights_only=False)
            eb = torch.load(b / "last.pt", map_location="cpu", weights_only=False)
            same_ema = all(torch.equal(ea["ema1"][k], eb["ema1"][k]) for k in ea["ema1"])
            same_mod = all(torch.equal(v, eb["model"][k])
                           for k, v in ea["model"].items())
            check(f"[{label}] 终态 model 与 EMA 逐位一致", same_ema and same_mod)
            if label == "moe":
                keys = [k for k in ea["model"] if "e_score_correction_bias" in k]
                check("[moe] 路由偏置 buffer 已随断点保存", len(keys) > 0)
                has_load = any("moe_load" in h for h in
                               json.loads((a / "loss_history.json").read_text()))
                check("[moe] 负载份额已入损失曲线", has_load)

    print("test_jit_resume: 全部通过")


if __name__ == "__main__":
    main()
