#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
grid.py — 域的经纬范围: 从数据派生, 并与合同核对
============================================================================
数据集根目录的 `lat.npy` / `lon.npy` 给的是网格单元的**下边界**(升序, 等距)。域的范围因此是
`[lat[0], lat[-1] + 步长]`, 而不是 `[lat.min(), lat.max()]` —— 后者会漏掉最后一个单元的宽度,
在 4x 网格上就是 0.0625 度。经度按东经存放(235..295), 统一转成西经负值。

写死一组经纬常量的代价是无声的: 画图只是把场铺在错误的矩形里, 州界、地名、按经纬框选区
全部整体平移, 图照出、指标照算, 没有任何东西会报错。所以这里的规矩是**从数据算, 再与合同
比对**, 不符就抛错。
============================================================================
"""
import os

import numpy as np

from downscaling_4x import contract as C

TOL = 1e-6


def _axis_bounds(values):
    """等距升序的单元下边界数组 -> (起, 止, 步长); 非等距抛错。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        raise ValueError("坐标轴至少要有两个点")
    step = np.diff(v)
    if not np.allclose(step, step[0], atol=1e-9):
        raise ValueError(f"坐标轴非等距: 步长范围 [{step.min()}, {step.max()}]")
    return float(v[0]), float(v[-1] + step[0]), float(step[0])


def domain_from_files(base_dir):
    """读 base_dir 的 lat.npy / lon.npy -> ((lat0,lat1), (lon0,lon1), (nlat,nlon))。"""
    lat_p, lon_p = os.path.join(base_dir, "lat.npy"), os.path.join(base_dir, "lon.npy")
    if not (os.path.isfile(lat_p) and os.path.isfile(lon_p)):
        raise FileNotFoundError(f"{base_dir} 缺 lat.npy / lon.npy, 无法核对域范围")
    lat = np.load(lat_p); lon = np.load(lon_p)
    lat0, lat1, _ = _axis_bounds(lat)
    lon0, lon1, _ = _axis_bounds(lon)
    if lon0 >= 180.0:                      # 东经 0..360 -> 西经负值
        lon0, lon1 = lon0 - 360.0, lon1 - 360.0
    return (lat0, lat1), (lon0, lon1), (int(np.asarray(lat).size), int(np.asarray(lon).size))


def check_domain(base_dir, hr_shape=None):
    """核对数据的域范围与网格数是否与合同一致; 不符抛错, 一致则返回域范围。"""
    lat, lon, (nlat, nlon) = domain_from_files(base_dir)
    bad = []
    if abs(lat[0] - C.DOMAIN_LAT[0]) > TOL or abs(lat[1] - C.DOMAIN_LAT[1]) > TOL:
        bad.append(f"纬度域 {lat} 期望 {C.DOMAIN_LAT}")
    if abs(lon[0] - C.DOMAIN_LON[0]) > TOL or abs(lon[1] - C.DOMAIN_LON[1]) > TOL:
        bad.append(f"经度域 {lon} 期望 {C.DOMAIN_LON}")
    if (nlat, nlon) != C.HR_SHAPE:
        bad.append(f"网格数 {(nlat, nlon)} 期望 {C.HR_SHAPE}")
    if hr_shape is not None and tuple(hr_shape[-2:]) != (nlat, nlon):
        bad.append(f"场形状 {tuple(hr_shape[-2:])} 与坐标轴 {(nlat, nlon)} 不符")
    if bad:
        raise ValueError("域与合同不符: " + "; ".join(bad))
    return lat, lon


def extent():
    """matplotlib imshow 的 extent: [lon0, lon1, lat0, lat1], 配 origin='lower'。"""
    return [C.DOMAIN_LON[0], C.DOMAIN_LON[1], C.DOMAIN_LAT[0], C.DOMAIN_LAT[1]]


def aspect():
    """按域中心纬度做的经纬轴长宽比, 使地图不被横向拉伸。"""
    mid = 0.5 * (C.DOMAIN_LAT[0] + C.DOMAIN_LAT[1])
    return 1.0 / float(np.cos(np.deg2rad(mid)))


def cell_centers():
    """(纬度中心, 经度中心) 两个一维数组, 长度分别为 HR_SHAPE。"""
    (la0, la1), (lo0, lo1) = C.DOMAIN_LAT, C.DOMAIN_LON
    nh, nw = C.HR_SHAPE
    dlat, dlon = (la1 - la0) / nh, (lo1 - lo0) / nw
    return (la0 + (np.arange(nh) + 0.5) * dlat), (lo0 + (np.arange(nw) + 0.5) * dlon)
