#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
dataset.py — 训练集统计与取数
============================================================================
`Stats` 读数据集自带的 train-only 均值/标准差(标量), `DownscaleData` 按 (年, 年内第几天)
切出条件张量、目标、陆地掩膜与原值真值。

条件张量的拼接**由通道名驱动**: 先建一张 {通道名: 二维场} 的表, 再严格按
`contract.cond_layout(mode)` 的名字顺序堆叠, 并断言两者的**名字序列**逐项相同。
只校验通道数是不够的 —— 51 通道里有 30 个同构的历史通道, 顺序错了数量照样对得上,
模型照常训练, 指标只是悄悄变差。

历史通道只取 ERA5。目标产品在任何时刻都不进条件张量, 历史帧也不例外。

`self.mask` 是有效域(目标有真值 且 输入有数据), 不等于 Daymet 的陆地掩膜 ——
后者把加拿大与墨西哥也算作陆地, 而那里没有 ERA5 输入。两者的区别见 contract。
============================================================================
"""
import json
import os
from collections import OrderedDict

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.downscale_baseline import (fill_nearest_valid, make_bilinear,
                                                    nearest_valid_index, npz_memmap)


class Stats:
    """训练年的均值/标准差(每个变量一个标量)与降水变换参数。"""

    def __init__(self, era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR,
                 in_vars=None, out_vars=None):
        self.in_vars = list(in_vars or C.ERA5_IN)
        self.out_vars = list(out_vars or C.TARGETS)
        em, es = _load_stats(era5_dir)
        dm, ds = _load_stats(daymet_dir)
        self.e_mean = np.array([em[v] for v in self.in_vars], np.float32)
        self.e_std = np.array([_nonzero(es[v], v) for v in self.in_vars], np.float32)
        self.d_mean = np.array([dm[v] for v in self.out_vars], np.float32)
        self.d_std = np.array([_nonzero(ds[v], v) for v in self.out_vars], np.float32)
        # Δz 只除标准差(mean 记 0), landcover 减均值再除标准差, 海陆掩膜不取统计量
        self.static_mean, self.static_std = {}, {}
        for name, (mode, key) in C.STATIC_NORM.items():
            if mode == "raw":
                continue
            if key not in ds or key not in dm:
                raise KeyError(f"{daymet_dir} 的统计里缺静态场 {key}")
            self.static_std[name] = float(_nonzero(ds[key], key))
            self.static_mean[name] = float(dm[key]) if mode == "zscore" else 0.0
        meta = _load_meta(daymet_dir)
        self.precip_log = bool(meta.get("precip_log", True))
        self.precip_clip = float(meta.get("precip_clip_mm", C.PRECIP_CLIP_MM))
        self.precip_scale = float(meta.get("precip_scale", C.PRECIP_SCALE))
        if self.precip_scale != C.PRECIP_SCALE or self.precip_clip != C.PRECIP_CLIP_MM:
            raise ValueError(f"数据集的降水变换参数 (scale={self.precip_scale}, "
                             f"clip={self.precip_clip}) 与合同不符")


def _load_stats(base_dir):
    out = []
    for name in ("normalize_mean.npz", "normalize_std.npz"):
        p = os.path.join(base_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺归一化统计 {p}")
        with np.load(p, allow_pickle=False) as z:
            out.append({k: float(np.asarray(z[k]).ravel()[0]) for k in z.files})
    return out


def _load_meta(base_dir):
    p = os.path.join(base_dir, "meta.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def _nonzero(v, name):
    if not np.isfinite(v) or v == 0:
        raise ValueError(f"{name} 的标准差为 {v}, 无法归一化")
    return v


class DownscaleData:
    """按 (年, 年内第几天) 取条件张量与目标。

    ERA5 按年缓存(LRU), Daymet 目标走未压缩 npz 的 memmap 逐日读, 静态场只从 static.npz 读
    一次 —— 年度文件里静态场被逐日复制了 365 份, 单文件 8GB, 不碰。
    """

    def __init__(self, era5_dir, daymet_dir, years, stats, mode=C.DEFAULT_MODE,
                 in_vars=None, out_vars=None, era5_cache_years=3):
        self.era5_dir, self.daymet_dir = era5_dir, daymet_dir
        self.s = stats
        self.mode = mode
        self.in_vars = list(in_vars or C.ERA5_IN)
        self.out_vars = list(out_vars or C.TARGETS)
        self.layout = C.cond_layout(mode)
        self.lags = C.history_lags(mode)
        self.f = C.FACTOR
        self.years = list(years)
        self._lr = OrderedDict()
        self._lr_cap = max(1, int(era5_cache_years))
        self._hr, self.ndays = {}, {}

        for y in self.years:
            df = M.find_year_files(daymet_dir, y)
            if not df:
                raise FileNotFoundError(f"{y}: 找不到 Daymet 文件")
            self._hr[y] = {}
            for v in self.out_vars:
                mm = npz_memmap(df[0], v)
                if mm is None:
                    raise ValueError(f"{df[0]} 的 {v} 不是未压缩存储, 无法 memmap")
                self._hr[y][v] = mm
            self.ndays[y] = int(self._hr[y][self.out_vars[0]].shape[0])

        oro_hr = M.load_static_2d(daymet_dir, "orography")
        lc = M.load_static_2d(daymet_dir, C.LANDCOVER)
        lsm = M.load_static_2d(daymet_dir, C.LAND_SEA_MASK)
        if oro_hr is None or lc is None or lsm is None:
            raise FileNotFoundError(f"{daymet_dir}/static.npz 缺 orography/landcover/land_sea_mask")
        self.H, self.W = oro_hr.shape
        self.Hl, self.Wl = C.LR_SHAPE
        C.check_shapes(C.LR_SHAPE, (self.H, self.W))
        self.daymet_land = lsm > M.LAND_THRESH
        if not self.daymet_land.any():
            raise ValueError(f"{daymet_dir} 的海陆掩膜没有陆地")
        self._up = make_bilinear(self.Hl, self.Wl, self.f)

        # Δz = 高分辨率高程 − 上采样的粗网格高程。
        oro_lr = M.load_static_2d(era5_dir, "orography")
        valid_lr = M.load_static_2d(era5_dir, "valid_mask")
        if oro_lr is None:
            raise FileNotFoundError(f"{era5_dir}/static.npz 缺 orography, 无法构造 Δz")
        if oro_lr.shape != C.LR_SHAPE:
            raise ValueError(f"ERA5 高程形状 {oro_lr.shape} 与合同 {C.LR_SHAPE} 不符")
        vm = (valid_lr > 0.5) if valid_lr is not None else np.isfinite(oro_lr)
        if (~np.isfinite(oro_lr) & vm).any():
            raise ValueError(f"{era5_dir} 的高程在 valid_mask 内含非有限值")
        # 有效域 = 目标有真值 且 输入有数据(见 contract.EFFECTIVE_DOMAIN)。
        # self.mask 是全包唯一的"哪些格点算数"的定义, loss、指标与静态填充都用它。
        self.era5_valid_hr = np.repeat(np.repeat(vm, self.f, 0), self.f, 1)
        self.mask = self.daymet_land & self.era5_valid_hr
        if not self.mask.any():
            raise ValueError("有效域为空: Daymet 陆地与 ERA5 valid_mask 无交集")
        # ERA5 在 valid_mask 外整片缺测, 而本合同的陆地掩膜远大于它: 缺测区就在要算损失的
        # 地方, 填法直接决定那一片的输入。用最近有效格的值, 不用全域均值。
        self._fill_idx = nearest_valid_index(vm)
        self.era5_valid = vm
        dz = oro_hr - self._up(fill_nearest_valid(oro_lr, self._fill_idx))

        raw = {C.DZ: dz, C.ELEVATION: oro_hr, C.LANDCOVER: lc,
               C.LAND_SEA_MASK: self.mask.astype(np.float32)}
        self._static = {}
        for name in C.STATIC_ORDER:
            norm, _ = C.STATIC_NORM[name]   # 不叫 mode: 会遮蔽本函数的 mode 参数
            a = raw[name]
            if norm != "raw":
                a = (a - self.s.static_mean[name]) / self.s.static_std[name]
            out = np.full((self.H, self.W), C.STATIC_OCEAN_FILL, np.float32)
            out[self.mask] = a[self.mask]                 # 陆地外一律填 0
            self._static[name] = out
        if not all(np.isfinite(v).all() for v in self._static.values()):
            raise ValueError("归一化后的静态通道含非有限值")

    # ---------------- ERA5 逐年缓存 ----------------
    def _era5_year(self, year):
        got = self._lr.get(year)
        if got is not None:
            self._lr.move_to_end(year)
            return got
        ef = M.find_year_files(self.era5_dir, year)
        if not ef:
            raise FileNotFoundError(f"{year}: 找不到 ERA5 文件")
        got = {}
        for v in self.in_vars:
            a = M.load_var_stack(ef, v)
            if a is None:
                raise KeyError(f"{year} 的 ERA5 缺变量 {v}")
            if a.shape[1:] != C.LR_SHAPE:
                raise ValueError(f"{year} 的 {v} 形状 {a.shape[1:]} 与合同 {C.LR_SHAPE} 不符")
            if not np.isfinite(a[:, self.era5_valid]).all():
                raise ValueError(f"{year} 的 {v} 在 valid_mask 内含缺测")
            got[v] = fill_nearest_valid(a, self._fill_idx)
        self._lr[year] = got
        while len(self._lr) > self._lr_cap:
            self._lr.popitem(last=False)
        return got

    def _era5_norm_day(self, year, day, var):
        """某天某变量: 上采样到 HR -> (降水先做正变换) -> z-score。"""
        i = self.in_vars.index(var)
        x = self._up(self._era5_year(year)[var][day])
        if var == C.PRECIP and self.s.precip_log:
            x = C.precip_fwd(x, self.s.precip_clip, self.s.precip_scale)
        return (x - self.s.e_mean[i]) / self.s.e_std[i]

    # ---------------- 条件张量 ----------------
    def cond_fields(self, year, day, history=None):
        """{通道名: (H,W)} 的有序表; 键的顺序即合同顺序, 由 assemble() 再核一遍。

        history 是按 self.lags 顺序给出的 [(年, 天), ...]; 需要历史的模式必须显式传入,
        缺省不会自行推算 —— 年内减索引正是要避免的那类错误。
        """
        f = OrderedDict()
        for v in self.in_vars:
            f[v] = self._era5_norm_day(year, day, v)
        for name in C.STATIC_ORDER:
            f[name] = self._static[name]                      # 已在构造时归一化并填过海
        sin_d, cos_d = C.doy_sincos(day)                      # 空间常数场, 已在 [-1,1]
        f[C.DOY_SIN] = np.full((self.H, self.W), sin_d, np.float32)
        f[C.DOY_COS] = np.full((self.H, self.W), cos_d, np.float32)
        if self.lags:
            if history is None or len(history) != len(self.lags):
                raise ValueError(f"{self.mode} 需要 {len(self.lags)} 个历史帧, 收到 {history!r}")
            for lag, (hy, hd) in zip(self.lags, history):
                for v in self.in_vars:
                    f[C.history_name(lag, v)] = self._era5_norm_day(hy, hd, v)
        return f

    def assemble(self, fields):
        """{通道名: (H,W)} -> (C,H,W); 名字序列必须与合同逐项相同。"""
        got, want = list(fields), self.layout
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            raise RuntimeError(f"条件通道名序列与合同不符: 共 {len(got)} vs {len(want)}, "
                               f"首个分歧在第 {first} 位 {got[first:first + 3]} vs {want[first:first + 3]}")
        return np.stack([np.asarray(fields[n], np.float32) for n in want], 0)

    # ---------------- 目标 ----------------
    def _hr_day(self, year, day, var):
        d = np.asarray(self._hr[year][var][day], np.float32)
        while d.ndim > 2:
            d = d[0]
        return d

    def target(self, year, day):
        """返回 (归一化目标, 原值真值); 原值保持数据集单位, 供评测反变换用。"""
        raw = np.stack([self._hr_day(year, day, v) for v in self.out_vars], 0)
        t = raw.copy()
        for i, v in enumerate(self.out_vars):
            if v == C.PRECIP and self.s.precip_log:
                t[i] = C.precip_fwd(t[i], self.s.precip_clip, self.s.precip_scale)
        norm = (t - self.s.d_mean[:, None, None]) / self.s.d_std[:, None, None]
        return norm.astype(np.float32), raw

    def full(self, year, day, history=None):
        """整幅一帧: (cond, target, mask, raw_target)。"""
        cond = self.assemble(self.cond_fields(year, day, history))
        norm, raw = self.target(year, day)
        return cond, norm, self.mask[None].astype(np.float32), raw

    def crop(self, cond, norm, mask, raw, y0, x0, Ph, Pw=None):
        """整幅结果上切块; 起点必须落在 FACTOR 网格上, 保证与 LR 单元对齐。"""
        Pw = Pw or Ph
        if y0 % self.f or x0 % self.f or Ph % self.f or Pw % self.f:
            raise ValueError(f"切块位置与尺寸必须是 {self.f} 的整数倍")
        sl = (slice(None), slice(y0, y0 + Ph), slice(x0, x0 + Pw))
        return cond[sl], norm[sl], mask[sl], raw[sl]
