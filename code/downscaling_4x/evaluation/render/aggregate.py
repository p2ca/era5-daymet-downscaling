# -*- coding: utf-8 -*-
"""共享聚合: 一次扫描逐日场目录, 归约成可复用对象, 持久化到 <fields>/agg/。

同一次渲染里多个模块共用 mass_matrix / annual_mean, 且重跑或加新图时直接秒读缓存, 不再
重扫 365 个文件。缓存按 (归约类型, 场名, 分区层级) 命名, 与 --out 无关, 故落在场目录旁。
"""
import hashlib
from pathlib import Path

import numpy as np

from downscaling_4x.evaluation.metrics import precip_log_mm


class Aggregator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.dir = Path(ctx.fields) / "agg"
        self.dir.mkdir(exist_ok=True)
        self._mem = {}
        # 分区指纹: 按分区归约的缓存必须带上它, 否则换一份 --regions 会静默命中旧分区的结果
        self.part_tag = hashlib.md5(
            np.ascontiguousarray(ctx.region_id.astype(np.int16))).hexdigest()[:8]

    def _load(self, key):
        if key in self._mem:
            return self._mem[key]
        p = self.dir / f"{key}.npz"
        if p.exists():
            z = np.load(p)
            d = {k: z[k] for k in z.files}
            self._mem[key] = d
            return d
        return None

    def _save(self, key, **arrs):
        np.savez_compressed(self.dir / f"{key}.npz", **arrs)
        self._mem[key] = arrs

    def mass_matrix(self, field, level="region"):
        """(分区 × 月) 误差质量 M 与格点数 K, 形状 (P+1, 13)(0 行/列弃用)。

        平均 = M/K; 分区贡献率 = 行和归一; 月贡献率 = 列和归一; 单区逐月 = 某行内部归一。
        对场里 NaN(陆外)自动跳过。level: 'region'(19) | 'compound'(8)。
        """
        key = f"mass_{field}_{level}_{self.part_tag}"
        c = self._load(key)
        if c is not None:
            return c["M"], c["K"]
        ctx = self.ctx
        rid, names = ctx.partition(level)
        P = len(names)
        land = ctx.land
        rid_l = rid[land]
        M = np.zeros((P + 1, 13))
        K = np.zeros((P + 1, 13))
        for y, t in ctx.available_days(field):
            C = np.load(ctx.field_path(field, y, t))[land].astype(np.float64)
            good = np.isfinite(C)
            mo = int(ctx.month_of_day[y][t])
            M[:, mo] += np.bincount(rid_l[good], weights=C[good], minlength=P + 1)
            K[:, mo] += np.bincount(rid_l[good], minlength=P + 1)
        self._save(key, M=M, K=K)
        return M, K

    def annual_mean_log(self, field):
        """逐像素年均 log1p 场 (H,W): 每日先做合同的 log 变换(置零后 log1p)再跨日平均。

        注意与 log1p(年均) 不是一回事: 这里逐日压缩再平均, 与 crps_log 场同口径,
        两张图才可以放在一起读。
        """
        key = f"annual_log_{field}"
        c = self._load(key)
        if c is not None:
            return c["mean"]
        ctx = self.ctx
        s = np.zeros((ctx.H, ctx.W))
        cnt = np.zeros((ctx.H, ctx.W))
        for y, t in ctx.available_days(field):
            a = np.load(ctx.field_path(field, y, t))
            m = np.isfinite(a)
            s[m] += precip_log_mm(a[m])
            cnt[m] += 1
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        self._save(key, mean=mean.astype(np.float32), cnt=cnt.astype(np.int32))
        return mean

    def annual_mean(self, field):
        """逐像素年均场 (H,W)(对每像素只在有限值上平均); 陆外/全缺为 NaN。"""
        key = f"annual_{field}"
        c = self._load(key)
        if c is not None:
            return c["mean"]
        ctx = self.ctx
        s = np.zeros((ctx.H, ctx.W))
        cnt = np.zeros((ctx.H, ctx.W))
        for y, t in ctx.available_days(field):
            a = np.load(ctx.field_path(field, y, t))
            m = np.isfinite(a)
            s[m] += a[m]
            cnt[m] += 1
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        self._save(key, mean=mean.astype(np.float32), cnt=cnt.astype(np.int32))
        return mean
