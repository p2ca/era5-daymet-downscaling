#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
mu_cache.py — 阶段 A 回归均值 μ 的缓存读取
============================================================================
阶段 B 每个训练样本都要用到全域 μ, 而 μ 只是 (阶段 A 权重, 某一天) 的确定性函数, 因此
预先算好存盘反复读取, 与阶段 B 的条件通道数、patch 尺寸、噪声参数均无关。

缓存按 `<目标>/<年份>.npy` 存放, 形状 (ndays, H, W), float16, 归一化空间。读取时 memmap,
不整年载入内存。

★ 缓存与产生它的阶段 A checkpoint 强绑定: manifest 里记了每个 checkpoint 的 SHA-256,
`verify()` 对不上就直接抛错(fail-closed)。阶段 A 一旦重训, 旧缓存必须重建。
============================================================================
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class MuCache:
    """按 (目标, 年, 日) 取全域 μ。"""

    def __init__(self, root, targets):
        self.root = Path(root)
        self.targets = list(targets)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"缓存缺少 manifest.json: {mpath}")
        self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        missing = [t for t in self.targets if t not in self.manifest["checkpoints"]]
        if missing:
            raise ValueError(f"manifest 未覆盖目标 {missing}")
        self._mm = {}

    def verify(self, ckpt_paths):
        """核对缓存所绑定的阶段 A checkpoint; 不一致直接抛错。

        `ckpt_paths` 为 {目标: checkpoint 路径}。
        """
        for t in self.targets:
            want = self.manifest["checkpoints"][t]["sha256"]
            got = file_sha256(ckpt_paths[t])
            if got != want:
                raise ValueError(
                    f"{t}: 缓存由另一个 checkpoint 生成 (manifest {want[:12]}…, "
                    f"当前 {got[:12]}…)。阶段 A 重训后必须重建缓存。")
        return True

    def years(self):
        return list(self.manifest["years"])

    def _arr(self, target, year):
        key = (target, int(year))
        if key not in self._mm:
            p = self.root / target / f"{year}.npy"
            if not p.exists():
                raise FileNotFoundError(f"缓存缺少 {p}")
            self._mm[key] = np.load(p, mmap_mode="r")
        return self._mm[key]

    def get(self, target, year, day):
        """取一天的全域 μ, 返回 float32 的 (H, W)。"""
        return np.asarray(self._arr(target, year)[day], dtype=np.float32)

    def get_stacked(self, year, day):
        """按 self.targets 的顺序堆成 (C, H, W)。"""
        return np.stack([self.get(t, year, day) for t in self.targets], 0)

    def coverage(self):
        """返回 {年: 天数}, 供完整性检查。"""
        out = {}
        for y in self.years():
            n = None
            for t in self.targets:
                a = self._arr(t, y)
                if n is None:
                    n = a.shape[0]
                elif a.shape[0] != n:
                    raise ValueError(f"{y}: 各目标天数不一致 {t}={a.shape[0]} vs {n}")
            out[int(y)] = int(n)
        return out
