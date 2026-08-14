#!/usr/bin/env python
# Packaged implementation; code/downscale_baseline.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
downscale_baseline.py — 共享工具库 (不单独运行)
============================================================================
只放被多个脚本复用的低层工具, 自己没有 main。被以下脚本 import:
  train_downscale.py   : npz_memmap / make_bilinear / fill_nan_daymean / _squeeze_2d / _load_raw_static
  train_statistical.py : 以上 + nearest_day / Acc / LAT_EDGES / LON_EDGES / FACTOR / PRECIP
  eval_common.py       : Acc / LAT_EDGES / LON_EDGES
  fill_era5_lastday.py : npz_memmap / _squeeze_2d / _load_raw_static
  plot_compare.py      : npz_memmap

工具清单:
  npz_memmap(path,var)      对"未压缩" .npz 直接 memmap 某变量(不读整块 1.5GB)
  make_bilinear(H,W,f)      返回 (H,W)->(H*f,W*f) 双线性上采样函数(权重预计算)
  nearest_day(lr2d,f)       6x6 块复制最近邻上采样
  fill_nan_daymean(lr)      逐天用当天有效均值填 NaN(防插值时 NaN 扩散)
  Acc                       逐(天,像素)累计 -> RMSE/MAE/bias/Pearson r
  _squeeze_2d / _squeeze_thw / _load_raw / _load_raw_static   IO/形状小工具
  LAT_EDGES / LON_EDGES / FACTOR / PRECIP                     常量
============================================================================
注: 统计降尺度基线(nearest/bilinear/bicubic/BCSD)已移到 train_statistical.py。
"""
import os
import struct
import sys
import zipfile

import numpy as np

from era5_daymet.contract import FACTOR
from era5_daymet.data import match_era5_daymet as M  # noqa: F401
LAT_EDGES = (23.625, 53.625)
LON_EDGES = (-125.125, -65.125)
PRECIP = "total_precipitation_24hr"


# ---------------------------------------------------------------------------
# 形状 / IO 小工具
# ---------------------------------------------------------------------------
def _squeeze_thw(a):
    a = np.asarray(a)
    if a.ndim == 4:        # (T,1,H,W)
        a = a[:, 0]
    return a


def _squeeze_2d(a):
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a.astype(np.float32)


def _load_raw(files, var):
    parts = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if var in z.files:
                parts.append(_squeeze_thw(z[var]))
    return np.concatenate(parts, 0).astype(np.float32) if parts else None


def _load_raw_static(files, var):
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if var in z.files:
                return z[var]
    return None


# ---------------------------------------------------------------------------
# 上采样
# ---------------------------------------------------------------------------
def make_bilinear(H, W, f=FACTOR):
    """返回一个 (H,W)->(H*f,W*f) 的双线性上采样函数(索引/权重预计算一次)。"""
    yy = np.clip((np.arange(H * f) + 0.5) / f - 0.5, 0, H - 1)
    xx = np.clip((np.arange(W * f) + 0.5) / f - 0.5, 0, W - 1)
    y0 = np.floor(yy).astype(int); y1 = np.minimum(y0 + 1, H - 1); wy = (yy - y0).astype(np.float32)
    x0 = np.floor(xx).astype(int); x1 = np.minimum(x0 + 1, W - 1); wx = (xx - x0).astype(np.float32)

    def up(a):
        top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
        bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
        return (top * (1 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)
    return up


def nearest_day(lr2d, f=FACTOR):
    return np.repeat(np.repeat(lr2d, f, 0), f, 1)


def make_bicubic(H, W, f=FACTOR):
    """返回 (H,W)->(H*f,W*f) 的双三次上采样函数(scipy.ndimage.zoom order=3); 无 scipy 时回退双线性。"""
    try:
        from scipy.ndimage import zoom
    except Exception:
        zoom = None
    if zoom is None:
        return make_bilinear(H, W, f)
    def up(a):
        return zoom(np.asarray(a, np.float32), f, order=3, mode="nearest").astype(np.float32)
    return up


def fill_nan_daymean(lr):
    """LR(对齐后含 NaN 域外) 逐天用当天有效均值填 NaN, 防止插值时 NaN 扩散。"""
    out = lr.astype(np.float32, copy=True)
    for t in range(out.shape[0]):
        m = np.isfinite(out[t])
        if m.any():
            out[t][~m] = out[t][m].mean()
        else:
            out[t][:] = 0.0
    return out


# ---------------------------------------------------------------------------
# 未压缩 .npz 的零拷贝 memmap
# ---------------------------------------------------------------------------
def _read_npy_header(fp, ver):
    """兼容不同 numpy 版本读取 .npy 头(新版去掉了 _read_array_header)。"""
    fmt = np.lib.format
    if hasattr(fmt, "_read_array_header"):
        return fmt._read_array_header(fp, ver)
    reader = getattr(fmt, f"read_array_header_{ver[0]}_0", None)
    if reader is None:
        raise ValueError(f"unsupported .npy version {ver}")
    return reader(fp)


def npz_memmap(path, var):
    """对'未压缩'的 .npz 直接 memmap 某变量(不把整块 1.5GB 读进内存)。压缩则返回 None。"""
    with zipfile.ZipFile(path) as zf:
        name = var + ".npy"
        if name not in zf.namelist():
            return None
        zi = zf.getinfo(name)
        if zi.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, "rb") as fp:
        fp.seek(zi.header_offset)
        nlen, elen = struct.unpack("<HH", fp.read(30)[26:30])
        fp.seek(zi.header_offset + 30 + nlen + elen)
        ver = np.lib.format.read_magic(fp)
        shape, fortran, dtype = _read_npy_header(fp, ver)
        data_off = fp.tell()
    if fortran:
        return None
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=data_off)


# ---------------------------------------------------------------------------
# 指标累计器
# ---------------------------------------------------------------------------
class Acc:
    """逐(天,像素)池化累计 -> RMSE/MAE/bias/Pearson r。"""
    def __init__(self):
        self.n = 0.0
        self.se = self.ae = self.be = 0.0
        self.sp = self.st = self.spp = self.stt = self.spt = 0.0

    def add(self, pred, tgt):
        d = pred - tgt
        self.n += d.size
        self.se += float((d * d).sum()); self.ae += float(np.abs(d).sum()); self.be += float(d.sum())
        p = pred.astype(np.float64); t = tgt.astype(np.float64)
        self.sp += p.sum(); self.st += t.sum()
        self.spp += (p * p).sum(); self.stt += (t * t).sum(); self.spt += (p * t).sum()

    def result(self):
        n = max(self.n, 1)
        rmse = (self.se / n) ** 0.5; mae = self.ae / n; bias = self.be / n
        cov = self.spt / n - (self.sp / n) * (self.st / n)
        vp = self.spp / n - (self.sp / n) ** 2; vt = self.stt / n - (self.st / n) ** 2
        r = cov / ((vp * vt) ** 0.5) if vp > 0 and vt > 0 else float("nan")
        return dict(rmse=round(rmse, 4), mae=round(mae, 4), bias=round(bias, 4),
                    corr=round(r, 4), n=int(self.n))
