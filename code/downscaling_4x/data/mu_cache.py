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

★ 缓存与产生它的阶段 A checkpoint 强绑定: manifest 里记了每个 checkpoint 的 SHA-256
与当时的 ERA5 输入变量表, `verify()` 两者都核, 对不上就直接抛错(fail-closed)。
阶段 A 一旦重训或输入口径变更, 旧缓存必须重建。
============================================================================
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C


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
        """核对缓存的输入口径与所绑定的阶段 A checkpoint; 不一致直接抛错。

        `ckpt_paths` 为 {目标: checkpoint 路径}。

        输入口径必须单独核: μ 是逐日的 (H, W) 标量场, 口径不符**不会引发任何形状错误**
        —— 残差 y - μ 照样算得出来, 阶段 B 照常收敛, 只是训练目标从一开始就是错的。
        SHA 校验挡不住这种情况: 拿旧阶段 A checkpoint 配旧缓存时它是通过的, 分叉的是
        缓存与**当前代码**的条件通道口径。
        """
        got = list(self.manifest.get("cond_layout", []))
        want = C.cond_layout(self.manifest.get("mode") or C.DEFAULT_MODE)
        if not got:
            raise ValueError("μ 缓存的 manifest 没有记 cond_layout, 无法核对口径; 请重建缓存。")
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         min(len(got), len(want)))
            raise ValueError(
                f"μ 缓存的条件口径与现行合同不符: 缓存 {len(got)} 通道 / 现行 {len(want)} 通道, "
                f"首个分歧在第 {first} 位 {got[first:first+2]} vs {want[first:first+2]}。"
                "阶段 A 口径变更后必须重建缓存。")
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
        """取一天的全域 μ, 返回 float32 的 (H, W)。

        取不到历史的帧在缓存里是 NaN(阶段 A 本就算不出 μ)。阶段 B 的帧集合与建缓存时用的
        是同一个 FrameIndex, 正常不会碰到; 真碰到就当场抛错, 而不是拿 NaN 去构造残差 ——
        NaN 残差会让 loss 变 NaN 后一路传下去, 表面上像是学习率没调好。
        """
        a = np.asarray(self._arr(target, year)[day], dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError(
                f"μ 缓存 {target} {year} 第 {day} 天含非有限值; 该帧在建缓存时没有完整历史, "
                "阶段 B 不应取它。请核对两边用的 FrameIndex 是否同一套。")
        return a

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
