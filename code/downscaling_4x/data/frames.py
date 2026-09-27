#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
frames.py — 帧集合与因果历史配对
============================================================================
把 (年, 年内第几天) 全部换算成真实日期建一张跨年的表, 再按真实日期回查前几天。
**不在年度文件内做索引加减**: 年度文件恰好 365 帧, `day_index - 2` 在年初会绕回同年
12 月下旬, 取到的是十一个月之后的天气, 而形状、通道数、量级全部正常, 没有任何东西会报错。

规则:
  * 前 `history_days` 天必须**全部**存在, 缺一天就丢弃该帧; 不复制、不 wrap、不补零。
  * 需要哪几天由 `history_days` 决定, 与实际用作通道的 `lags` 无关 —— 于是"用历史"和
    "不用历史但只在同一批帧上训练"的两档模型共享逐帧相同的帧集合, 二者之差才只剩通道。
  * 闰年丢 12/31, 所以闰年次年的 1 月 1-2 日取不到完整历史, 会被丢掉。这是预期行为,
    `dropped` 计数会把它显式报出来。
  * 历史只回查 ERA5(预测时已存在的量), 目标产品不参与配对, 也不进条件张量。

`signature()` 把帧集合与配对关系压成一个 sha256。两个 run 要并排比较, 先比这个签名:
不同签名意味着训练/评测的帧集合不同, 指标不可比, 而通道数可能完全一样。
============================================================================
"""
import hashlib
import json
from datetime import timedelta

from downscaling_4x.data import match_era5_daymet as M

PAIRING_POLICY = "real_calendar_dates_v1"


class FrameIndex:
    """某个划分的可用帧与它们的历史帧。

    available_years 是**建日期表**用的年份全集, 通常给 train+val+test 的并集: 验证集年初
    要回查上一个训练年的 ERA5, 只放本划分的年份会把年初几帧误判为缺历史。
    """

    def __init__(self, split_years, available_years, history_days=0, lags=(), split=""):
        self.split = split
        self.history_days = int(history_days)
        self.lags = tuple(int(v) for v in lags)
        if self.history_days < 0:
            raise ValueError("history_days 不能为负")
        if any(v < 1 for v in self.lags):
            raise ValueError("历史 lag 必须是正的因果偏移")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("历史 lag 不能重复")
        if self.lags and max(self.lags) > self.history_days:
            raise ValueError(f"lag {self.lags} 超出配对窗口 {self.history_days} 天")

        self.date_to_frame = {}
        for y in sorted(set(available_years)):
            for d in range(M.DAYS_PER_YEAR):
                key = M.frame_date(y, d)
                if key in self.date_to_frame:
                    raise ValueError(f"两个帧解析到同一天 {key}")
                self.date_to_frame[key] = (y, d)

        self.frames, self.required, self.history = [], [], []
        self.dropped = 0
        for y in split_years:
            for d in range(M.DAYS_PER_YEAR):
                cur = M.frame_date(y, d)
                need = [cur - timedelta(days=k) for k in range(self.history_days, 0, -1)]
                if any(k not in self.date_to_frame for k in need):
                    self.dropped += 1
                    continue
                self.frames.append((y, d))
                self.required.append(tuple(self.date_to_frame[k] for k in need))
                self.history.append(tuple(self.date_to_frame[cur - timedelta(days=v)]
                                          for v in self.lags))
        if len(set(self.frames)) != len(self.frames):
            raise ValueError("帧集合内有重复")

    def __len__(self):
        return len(self.frames)

    def history_of(self, i):
        """第 i 帧按 self.lags 顺序给出的历史帧 [(年, 天), ...]。"""
        return self.history[i]

    def signature(self):
        """帧集合 + 配对关系的 sha256; 帧集合一变签名就变。"""
        payload = json.dumps(
            {"policy": PAIRING_POLICY,
             "required_history_days": self.history_days,
             "split": self.split,
             "pairs": [[list(f), [list(h) for h in r]]
                       for f, r in zip(self.frames, self.required)]},
            sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def metadata(self):
        """写进 checkpoint / meta.json 的配对档案。"""
        return {"policy": PAIRING_POLICY,
                "required_history_days": self.history_days,
                "history_lags": list(self.lags),
                "split": self.split,
                "eligible_frames": len(self.frames),
                "dropped_missing_history": self.dropped,
                "frame_pairing_sha256": self.signature()}


def assert_same_pairing(a, b):
    """两个 FrameIndex 的帧集合必须逐帧相同, 否则它们的指标不可并表。"""
    sa, sb = a.signature(), b.signature()
    if sa != sb:
        raise ValueError(f"帧集合不同, 指标不可比: {sa[:12]} vs {sb[:12]} "
                         f"({len(a)} 帧 vs {len(b)} 帧)")
