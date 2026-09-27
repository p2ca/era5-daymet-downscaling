#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
dataset.py — 归一化统计与整幅/patch 取数(纯 numpy, 不依赖 torch)
============================================================================
Stats 只加载 compute_norm_stats 产出的均值/标准差/气候态, 不重算; DownscaleData 按年
持有 ERA5 低分辨率场与 Daymet 高分辨率场, 按需切出 (cond, target, mask, 原值真值)。

训练与评测共用同一份取数实现: 条件通道的拼接顺序、降水的 log1p 变换、静态通道的归一化
都在这里定死, 两侧任一处自行实现都会让 val 与 test 的输入口径悄悄分叉。
通道顺序与降水单位空间的定义见 era5_daymet.contract。
============================================================================
"""
import json
import os

import numpy as np

from era5_daymet.contract import FACTOR, PRECIP, cond_channels, doy_sincos, precip_fwd
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import (
    _load_raw_static,
    _squeeze_2d,
    fill_nan_daymean,
    make_bilinear,
    npz_memmap,
)


class Stats:
    def __init__(self, stats_dir, in_vars, out_vars):
        e_m = np.load(os.path.join(stats_dir, "era5", "normalize_mean.npz"))
        e_s = np.load(os.path.join(stats_dir, "era5", "normalize_std.npz"))
        d_m = np.load(os.path.join(stats_dir, "daymet", "normalize_mean.npz"))
        d_s = np.load(os.path.join(stats_dir, "daymet", "normalize_std.npz"))
        meta = json.load(open(os.path.join(stats_dir, "daymet", "meta.json")))
        self.e_mean = np.array([float(e_m[v]) for v in in_vars], np.float32)
        self.e_std = np.array([max(float(e_s[v]), 1e-6) for v in in_vars], np.float32)
        self.d_mean = np.array([float(d_m[v]) for v in out_vars], np.float32)
        self.d_std = np.array([max(float(d_s[v]), 1e-6) for v in out_vars], np.float32)
        self.precip_log = bool(meta.get("precip_log", False))             # 与 stats 一致的降水变换
        self.precip_clip = float(meta.get("precip_clip", 0.1))
        self.precip_scale = float(meta.get("precip_scale", 1000.0))
        # 静态条件通道(Δz / 绝对高程 / landcover)的归一化量。缺失时直接报错而不是兜底:
        # 兜一个默认值会让这几个通道悄悄跑到错误的量纲上, 训练照常收敛, 只是结果变差。
        for k in ("orography", "landcover"):
            if k not in d_m.files or k not in d_s.files:
                raise KeyError(f"{stats_dir}/daymet 缺少静态量 {k!r} 的 mean/std, "
                               f"静态条件通道无法归一化; 需重新运行 compute_norm_stats")
        self.oro_mean = float(d_m["orography"])
        self.oro_std = max(float(d_s["orography"]), 1e-6)
        self.lc_mean = float(d_m["landcover"])
        self.lc_std = max(float(d_s["landcover"]), 1e-6)


class DownscaleData:
    def __init__(self, era5_dir, daymet_dir, years, in_vars, out_vars, stats, factor=FACTOR):
        self.in_vars, self.out_vars, self.s, self.f = in_vars, out_vars, stats, factor
        self.lr, self.hr, self.dz, self.oro, self.lc, self.lsm, self.mask, self.ndays = ({} for _ in range(8))
        upref = None
        for y in years:
            ef = M.find_year_files(era5_dir, y); df = M.find_year_files(daymet_dir, y)
            if not ef or not df:
                raise FileNotFoundError(f"{y}: 缺 ERA5/Daymet")
            self.lr[y] = {v: fill_nan_daymean(M.load_var_stack(ef, v)) for v in in_vars}
            self.hr[y] = {v: npz_memmap(df[0], v) for v in out_vars}
            T, Hl, Wl = next(iter(self.lr[y].values())).shape
            self.ndays[y] = T
            if upref is None:
                upref = make_bilinear(Hl, Wl, factor); self.Hl, self.Wl = Hl, Wl
                self.H, self.W = Hl * factor, Wl * factor
            lro = M.load_var_stack(ef, "orography")
            lro = fill_nan_daymean(lro)[0] if lro is not None else np.zeros((Hl, Wl), np.float32)
            hro = _squeeze_2d(_load_raw_static(df, "orography"))
            if hro is None:
                raise FileNotFoundError(f"{y}: Daymet 缺少 orography, 无法构造 Δz 与绝对高程条件通道")
            self.oro[y] = hro.astype(np.float32)                     # HR 绝对高程(米), 无填充值
            self.dz[y] = self.oro[y] - upref(lro.astype(np.float32))
            lc = _squeeze_2d(_load_raw_static(df, "landcover"))
            self.lc[y] = (lc if lc is not None else np.zeros((self.H, self.W), np.float32)).astype(np.float32)
            lsm = _squeeze_2d(_load_raw_static(df, M.LAND_MASK_VAR))
            self.lsm[y] = (lsm > 0.5).astype(np.float32)
            self.mask[y] = (lsm > 0.5)
        self.years = list(years)

    def _hr(self, y, v, t):
        d = np.asarray(self.hr[y][v][t]); return d[0] if d.ndim == 3 else d

    def get_patch(self, y, t, y0, x0, Ph, Pw=None):
        Pw = Pw or Ph; f = self.f; s = self.s
        ly0, lx0, lph, lpw = y0 // f, x0 // f, Ph // f, Pw // f
        up = make_bilinear(lph, lpw, f)
        cin = []
        for i, v in enumerate(self.in_vars):
            x = up(self.lr[y][v][t, ly0:ly0 + lph, lx0:lx0 + lpw].astype(np.float32))
            if v == PRECIP and s.precip_log:                  # ERA5 输入降水: ×scale+clip+log1p(去 drizzle)
                x = precip_fwd(x, s.precip_clip, s.precip_scale)
            cin.append((x - s.e_mean[i]) / s.e_std[i])
        cin = np.stack(cin, 0)                                                         # (Cin,Ph,Pw)
        # Δz 天然零中心, 只除 std(填 0 恰好等于"HR 地形 = 粗网格地形"); 绝对高程均值 ~800m,
        # 必须整体 z-score, 否则整幅压一个常数偏置。两者的 mean/std 只在陆地上统计。
        dz = (self.dz[y][y0:y0 + Ph, x0:x0 + Pw] / s.oro_std)[None]
        elev = ((self.oro[y][y0:y0 + Ph, x0:x0 + Pw] - s.oro_mean) / s.oro_std)[None]
        lc = ((self.lc[y][y0:y0 + Ph, x0:x0 + Pw] - s.lc_mean) / s.lc_std)[None]
        lsm = self.lsm[y][y0:y0 + Ph, x0:x0 + Pw][None]
        parts = [cin, dz, elev, lc, lsm]
        sin_d, cos_d = doy_sincos(t)                      # 时间通道: 空间常数, 已在 [-1,1] 不再归一化
        parts += [np.full((1, Ph, Pw), sin_d, np.float32), np.full((1, Ph, Pw), cos_d, np.float32)]
        cond = np.concatenate(parts, 0).astype(np.float32)                             # (Cin,Ph,Pw)
        want = cond_channels(self.in_vars)
        if cond.shape[0] != want:                         # 拼接顺序与合同分叉时当场喊停
            raise RuntimeError(f"条件通道数 {cond.shape[0]} 与合同 {want} 不符")
        hr = np.stack([self._hr(y, v, t)[y0:y0 + Ph, x0:x0 + Pw] for v in self.out_vars], 0).astype(np.float32)
        ht = hr.copy()
        for i, v in enumerate(self.out_vars):
            if v == PRECIP and s.precip_log:                  # Daymet 目标降水: 同样 ×scale+clip+log1p
                ht[i] = precip_fwd(ht[i], s.precip_clip, s.precip_scale)
        target = ((ht - s.d_mean[:, None, None]) / s.d_std[:, None, None]).astype(np.float32)
        m = self.mask[y][y0:y0 + Ph, x0:x0 + Pw].astype(np.float32)
        return cond, target, m[None], hr                      # hr 保持原值, 给评测当真值

    def random_patch(self, rng, P, min_land=0.3, tries=20):
        for _ in range(tries):
            y = self.years[rng.integers(len(self.years))]
            t = int(rng.integers(self.ndays[y]))
            y0 = int(rng.integers(0, (self.H - P) // self.f + 1)) * self.f
            x0 = int(rng.integers(0, (self.W - P) // self.f + 1)) * self.f
            if self.mask[y][y0:y0 + P, x0:x0 + P].mean() >= min_land:
                return self.get_patch(y, t, y0, x0, P)
        return self.get_patch(y, t, y0, x0, P)

    def full(self, y, t):
        return self.get_patch(y, t, 0, 0, self.H, self.W)
