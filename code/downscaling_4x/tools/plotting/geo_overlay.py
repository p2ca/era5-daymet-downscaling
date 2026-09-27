#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geo_overlay.py — 地图上的地理参照图层

提供三种叠加要素, 让一张性能地图能独立读出"在哪":
  州界      从 GeoJSON 直接解析并按经纬度画线, 只用标准库 json + matplotlib
  地形晕渲  由 Daymet 高程做山体阴影, 使山脉的形状可辨认
  地名标注  主要山系与地形区的名称, 位置为人工指定的经纬度

GeoJSON 为纯坐标数组, 用 json 读出后直接画线即可, 不需要 geopandas / shapely / cartopy。
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LightSource

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STATES_GEOJSON = PROJECT_ROOT / "data/geo/us-states.json"

# 主要山系与地形区的标注位置 (经度, 纬度, 名称)
PLACE_LABELS = [
    (-120.6, 38.6, "Sierra\nNevada"),
    (-121.6, 45.4, "Cascades"),
    (-123.4, 41.0, "Klamath"),
    (-110.5, 44.2, "N. Rockies"),
    (-105.8, 39.3, "Front\nRange"),
    (-107.6, 37.6, "San Juan"),
    (-111.6, 40.6, "Wasatch"),
    (-110.5, 37.0, "Colorado\nPlateau"),
    (-117.0, 39.8, "Great\nBasin"),
    (-112.5, 33.0, "Sonoran\nDesert"),
    (-102.0, 41.5, "High\nPlains"),
    (-93.3, 36.3, "Ozarks"),
    (-83.2, 36.0, "S. Appalachians"),
    (-74.6, 43.6, "Adirondacks"),
    (-71.6, 44.2, "White\nMts"),
    (-90.5, 32.5, "Miss.\nAlluvial\nPlain"),
    (-81.5, 28.5, "Florida\nPeninsula"),
    (-121.6, 36.9, "Central\nValley"),
]


def state_polylines(path=STATES_GEOJSON):
    """返回州界折线列表 [(lon 数组, lat 数组), ...]; 文件缺失时返回空列表。"""
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        gt = geom.get("type")
        rings = []
        if gt == "Polygon":
            rings = geom["coordinates"]
        elif gt == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        for ring in rings:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 1:
                out.append((arr[:, 0], arr[:, 1]))
    return out


def draw_states(ax, color="0.25", lw=0.45, alpha=0.85, path=STATES_GEOJSON):
    for lon, lat in state_polylines(path):
        ax.plot(lon, lat, color=color, lw=lw, alpha=alpha, zorder=3,
                solid_joinstyle="round")


def hillshade(elevation, land, extent, azdeg=315, altdeg=45, vert_exag=0.02):
    """由高程生成山体阴影 (0..1); 非陆地置 NaN 以免海面产生阴影纹理。"""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    dx = (extent[1] - extent[0]) / elevation.shape[1]
    dy = (extent[3] - extent[2]) / elevation.shape[0]
    hs = ls.hillshade(np.where(land, elevation, 0.0), vert_exag=vert_exag, dx=dx, dy=dy)
    return np.where(land, hs, np.nan)


def draw_place_labels(ax, fontsize=6.6, color="0.12", labels=PLACE_LABELS):
    for lon, lat, name in labels:
        ax.text(lon, lat, name, fontsize=fontsize, color=color, ha="center", va="center",
                zorder=6, linespacing=0.95, fontweight="bold",
                path_effects=_halo())


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.9, foreground="white", alpha=0.85)]
