#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_regions.py — 域内标准分区(Bukovsky 骨架 + 地形细分)的构建与锚图

基础是 Bukovsky (2011) 北美分区: Ricketts 陆地生态区的简化版(与 NEON 对齐), 29 个
基本区 + 13 个合并组, 官方掩膜为 0.5° 网格。本脚本把它落到 Daymet 1/24° 网格:

  1. 基本区掩膜最近邻重栅格化, 未覆盖的陆地像素按最近已分配区补齐;
  2. 地形细分(Bukovsky 在 0.5° 尺度上无法区分的两处): 在 PacificNW/PacificSW 内,
     以 25 km 高斯平滑高程超过阈值的最大连通域分别拆出 Cascades 与 SierraNevada;
  3. 陆地占比过小的残片并入最近邻区(合并规则写入 spec);
  4. 孤块归并: 与本区主体不相连、又主要被别的陆地区包住的小块整块并入共边最长的邻区,
     使每个区在陆地上连通(海岛与湖岸块以水为界, 不属此列, 保持原属);
  5. 每个基本区按与官方合并组掩膜的最大重叠归入合并组, 细分区继承母区归属。

输出 <out>/: regions_<版本>.npz(region_id/compound_id/名称表/spec), 锚图若干(地形晕渲
基底 + 半透明分区色 + 边界 + 区名), 以及官方掩膜 zip 的存档副本。
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.tools.plotting import geo_overlay as GEO

EXT = G.extent()
ASPECT = G.aspect()

# 基本区候选(域内可能出现的陆地区; 域外的读进来也只是全零, 无害)
BASIC = ["PacificNW", "PacificSW", "GreatBasin", "Southwest", "Mezquital",
         "NRockies", "SRockies", "NPlains", "CPlains", "SPlains", "Prairie",
         "GreatLakes", "Appalachia", "DeepSouth", "Southeast", "MidAtlantic",
         "NorthAtlantic", "WestBoreal", "EastBoreal", "WestTaiga", "EastTaiga"]
COMPOUND = ["MtWest", "Desert", "GreatPlains", "Central", "East", "EastCoast",
            "South", "WetSouth", "Rockies", "Boreal", "Taigas", "NorthernNA"]
# 地形细分: 山脉 = 地理窗口内平滑高程超阈值的最大连通域, 两坡像素不论原属哪个区
# 都并入新区(Bukovsky 的海岸/内陆分界沿山脊走, 只从单一母区取会丢掉东坡)。
# nominal_parent 只用于继承合并组归属。10 km 平滑保留山间谷地作为天然分隔。
SPLITS = [
    {"child": "Cascades", "nominal_parent": "PacificNW",
     "window": (-122.9, -119.6, 40.3, 53.625), "elev_thr_m": 900.0},
    {"child": "SierraNevada", "nominal_parent": "PacificSW",
     "window": (-121.3, -117.6, 35.0, 40.3), "elev_thr_m": 1500.0},
]
SMOOTH_KM = 10.0

def load_mask(d, name):
    f = netcdf_file(str(d / f"{name}.nc"), "r", mmap=False)
    m = np.asarray(f.variables["mask"][:], np.float64)
    lat = np.asarray(f.variables["lat"][:], np.float64)
    lon = np.asarray(f.variables["lon"][:], np.float64)
    f.close()
    return m > 0.5, lat, lon


def regrid_ids(mask_dir, names, H, W):
    """0.5° 掩膜按像素中心最近邻取到目标网格 -> (H,W) 的 1 基区号, 0=未分配。"""
    lat_px, lon_px = G.cell_centers()
    lon_px = lon_px + 360.0
    first = load_mask(mask_dir, names[0])
    glat, glon = first[1], first[2]
    iy = np.clip(np.round((lat_px - glat[0]) / (glat[1] - glat[0])).astype(int), 0, len(glat) - 1)
    ix = np.clip(np.round((lon_px - glon[0]) / (glon[1] - glon[0])).astype(int), 0, len(glon) - 1)
    ids = np.zeros((H, W), np.int16)
    overlap = 0
    for k, nm in enumerate(names, 1):
        m = load_mask(mask_dir, nm)[0]
        hit = m[np.ix_(iy, ix)]
        overlap += int((hit & (ids > 0)).sum())
        ids[hit & (ids == 0)] = k
    return ids, overlap


def outside_neighbours(ids, blk):
    """块 blk 的四邻中落在块外的那些格的区号(0 = 水或域外)。"""
    H, W = ids.shape
    rs, cs = np.nonzero(blk)
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = np.clip(rs + dr, 0, H - 1), np.clip(cs + dc, 0, W - 1)
        keep = ~blk[rr, cc]
        out.append(ids[rr[keep], cc[keep]])
    return np.concatenate(out) if out else np.zeros(0, ids.dtype)


def merge_orphan_blocks(ids, names, max_px):
    """把飞地并进周边区, 让每个区在陆地上连通。返回归并记录。

    判据分两步, 目的是只动"细分留下的孤块"而不碰真岛:
      * 只看非最大连通块, 且像素数 < max_px;
      * 该块四邻周长里陆地邻居过半才算飞地 —— 海岛、湖心岛、湖岸碎块的周长以水为主,
        并进"最近的陆地区"反而会把它们扔给隔着水面的另一个区, 所以一律不动。
    整块并入共边最长的邻区(不按距离逐像素切), 这样不会在原处留下新的接缝。
    """
    rec = []
    while True:
        moved = False
        for k in range(1, len(names) + 1):
            sel = ids == k
            if not sel.any():
                continue
            lab, nc = label(sel)
            if nc <= 1:
                continue
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            body = int(sz.argmax())
            for c in range(1, nc + 1):
                if c == body or sz[c] >= max_px:
                    continue
                blk = lab == c
                nb = outside_neighbours(ids, blk)
                land_nb = nb[nb > 0]
                if len(land_nb) * 2 <= len(nb):
                    continue                      # 以水为界 -> 真岛, 保持原属
                v, ct = np.unique(land_nb, return_counts=True)
                dst = int(v[int(ct.argmax())])
                ids[blk] = dst
                rec.append({"from": names[k - 1], "into": names[dst - 1],
                            "px": int(sz[c]),
                            "shared_edge_pct": round(100 * ct.max() / len(land_nb), 1)})
                moved = True
        if not moved:
            return rec


def largest_component(mask):
    lab, n = label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros_like(mask)
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    return lab == int(sz.argmax())


def draw_map(fig_path, ids, names, oro, land, title, label_min_px=1500,
             crop=None, figsize=(13.2, 7.6), fontsize=7.6):
    """crop=(lon0,lon1,lat0,lat1): 直接裁剪数组绘制放大图, 不依赖坐标轴缩放。"""
    Hf, Wf = land.shape
    ext = list(EXT)
    if crop is not None:
        lo0, lo1, la0, la1 = crop
        x0 = max(int((lo0 - EXT[0]) / (EXT[1] - EXT[0]) * Wf), 0)
        x1 = min(int(np.ceil((lo1 - EXT[0]) / (EXT[1] - EXT[0]) * Wf)), Wf)
        y0 = max(int((la0 - EXT[2]) / (EXT[3] - EXT[2]) * Hf), 0)
        y1 = min(int(np.ceil((la1 - EXT[2]) / (EXT[3] - EXT[2]) * Hf)), Hf)
        ids, oro, land = ids[y0:y1, x0:x1], oro[y0:y1, x0:x1], land[y0:y1, x0:x1]
        ext = [EXT[0] + x0 * (EXT[1] - EXT[0]) / Wf, EXT[0] + x1 * (EXT[1] - EXT[0]) / Wf,
               EXT[2] + y0 * (EXT[3] - EXT[2]) / Hf, EXT[2] + y1 * (EXT[3] - EXT[2]) / Hf]
    H, W = land.shape
    # 高程着色地势图: 颜色 = 海拔(hypsometric), 亮度 = 山体阴影, 海洋为淡蓝
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    norm = matplotlib.colors.Normalize(vmin=-1200.0, vmax=3600.0)
    cmap = plt.get_cmap("terrain")
    rgb = ls.shade(np.where(land, oro, 0.0), cmap=cmap, norm=norm,
                   blend_mode="soft", vert_exag=0.02,
                   dx=(ext[1] - ext[0]) / W, dy=(ext[3] - ext[2]) / H)
    rgb[~land] = matplotlib.colors.to_rgba("#dce8f2")
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=ext, aspect=ASPECT,
              interpolation="bilinear", zorder=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    # 分区 = 黑色实线勾边, 不填色
    lonc = np.linspace(ext[0], ext[1], W, endpoint=False) + (ext[1] - ext[0]) / W / 2
    latc = np.linspace(ext[2], ext[3], H, endpoint=False) + (ext[3] - ext[2]) / H / 2
    for k in range(1, len(names) + 1):
        sel = (ids == k)
        if sel.any():
            ax.contour(lonc, latc, sel.astype(float), levels=[0.5],
                       colors="#101418", linewidths=1.35, zorder=4)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.01)
    cb.set_label("elevation [m]")
    cb.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    for k, nm in enumerate(names, 1):
        sel = ids == k
        if sel.sum() < label_min_px:
            continue
        body = largest_component(sel)
        ys, xs = np.nonzero(body)
        cy, cx = ys.mean(), xs.mean()
        j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        lon = ext[0] + (xs[j] + 0.5) * (ext[1] - ext[0]) / W
        lat = ext[2] + (ys[j] + 0.5) * (ext[3] - ext[2]) / H
        ax.text(lon, lat, nm, fontsize=fontsize, fontweight="bold", color="#101418",
                ha="center", va="center", zorder=6, path_effects=GEO._halo())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="构建域内标准分区")
    ap.add_argument("--masks", required=True, help="Bukovsky region-masks 解压目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1", help="产物版本号, 决定输出文件名与 spec 里的 version")
    ap.add_argument("--scrap-frac", type=float, default=0.004,
                    help="陆地占比低于此值的区并入最近邻区")
    a = ap.parse_args()
    mask_dir, out = Path(a.masks), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 陆地 = 有效域(daymet_land AND era5_valid), 与训练 loss 及全部指标同一掩膜:
    # 本产品的 Daymet 陆地掩膜把加拿大与墨西哥也算作陆地, 而那里没有 ERA5 输入,
    # 分区若覆盖到那里, 区域排名会混入永远是 NaN 的像素。
    oro = M.load_static_2d(M.DAYMET_DIR, "orography").astype(np.float64)
    lsm = M.load_static_2d(M.DAYMET_DIR, C.LAND_SEA_MASK)
    valid_lr = M.load_static_2d(M.ERA5_DIR, "valid_mask")
    if lsm is None or valid_lr is None or oro is None:
        raise SystemExit("static.npz 缺 orography / land_sea_mask / valid_mask")
    land = (lsm > M.LAND_THRESH) & np.repeat(np.repeat(valid_lr > 0.5, C.FACTOR, 0),
                                             C.FACTOR, 1)
    H, W = land.shape
    n_land = int(land.sum())

    ids, overlap = regrid_ids(mask_dir, BASIC, H, W)
    ids[~land] = 0
    print(f"[regions] 基本区重栅格化: 掩膜重叠像素 {overlap}(以先序为准); "
          f"未覆盖陆地 {int((land & (ids == 0)).sum())} px")

    # 最近邻补洞(海岸缘/掩膜缝隙)
    hole = land & (ids == 0)
    if hole.any():
        idx = distance_transform_edt(ids == 0, return_distances=False, return_indices=True)
        ids[hole] = ids[tuple(i[hole] for i in idx)]
    names = list(BASIC)

    # 地形细分
    px_km = 111.0 * (EXT[3] - EXT[2]) / H
    oro_s = gaussian_filter(np.where(land, oro, 0.0), sigma=SMOOTH_KM / px_km)
    lat_px, lon_px = G.cell_centers()
    spec_splits = []
    for sp in SPLITS:
        lo0, lo1, la0, la1 = sp["window"]
        win = ((lat_px[:, None] >= la0) & (lat_px[:, None] <= la1)
               & (lon_px[None, :] >= lo0) & (lon_px[None, :] <= lo1))
        body = largest_component(win & land & (oro_s >= sp["elev_thr_m"]))
        if not body.any():
            raise RuntimeError(f"{sp['child']}: 窗口内无超过 {sp['elev_thr_m']} m 的连通域")
        src = {names[k - 1]: int(((ids == k) & body).sum())
               for k in np.unique(ids[body]) if k > 0}
        names.append(sp["child"])
        ids[body] = len(names)
        spec_splits.append({**sp, "smooth_km": SMOOTH_KM, "px": int(body.sum()),
                            "taken_from": src})
        print(f"[regions] 细分 {sp['child']}: {int(body.sum())} px "
              f"({100 * body.sum() / n_land:.2f}% 陆地), 来源 {src}")

    # 残片合并
    merged = []
    while True:
        cnt = np.bincount(ids[land], minlength=len(names) + 1)
        small = [k for k in range(1, len(names) + 1)
                 if 0 < cnt[k] < a.scrap_frac * n_land
                 and names[k - 1] not in [s["child"] for s in spec_splits]]
        if not small:
            break
        k = min(small, key=lambda q: cnt[q])
        sel = ids == k
        ids[sel] = 0
        idx = distance_transform_edt(ids == 0, return_distances=False, return_indices=True)
        ids[sel] = ids[tuple(i[sel] for i in idx)]
        merged.append({"region": names[k - 1], "px": int(cnt[k]),
                       "into": "最近邻区"})
        print(f"[regions] 残片并入: {names[k - 1]} ({cnt[k]} px, "
              f"{100 * cnt[k] / n_land:.3f}% 陆地)")

    # 孤块归并(飞地 -> 共边最长的邻区)
    orphans = merge_orphan_blocks(ids, names, a.scrap_frac * n_land)
    for o in orphans:
        print(f"[regions] 孤块并入: {o['from']} {o['px']} px -> {o['into']} "
              f"(共边 {o['shared_edge_pct']}%)")
    print(f"[regions] 孤块归并共 {len(orphans)} 块 / "
          f"{sum(o['px'] for o in orphans)} px")

    # 压缩区号(去掉被并空的), 按陆地占比排序命名表
    cnt = np.bincount(ids[land], minlength=len(names) + 1)
    keep = [k for k in range(1, len(names) + 1) if cnt[k] > 0]
    keep.sort(key=lambda q: -cnt[q])
    remap = np.zeros(len(names) + 1, np.int16)
    final = []
    for r, k in enumerate(keep, 1):
        remap[k] = r
        final.append(names[k - 1])
    ids = remap[ids]

    # 合并组归属: 与官方合并组掩膜的最大重叠; 细分区继承母区
    comp_ids, _ = regrid_ids(mask_dir, COMPOUND, H, W)
    parent_of = {s["child"]: s["nominal_parent"] for s in spec_splits}
    comp_of = {}
    for r, nm in enumerate(final, 1):
        probe = parent_of.get(nm, nm)
        if probe != nm:
            comp_of[nm] = comp_of.get(probe)
        sel = ids == (final.index(probe) + 1 if probe in final else r)
        if not sel.any():
            continue
        cc = np.bincount(comp_ids[sel], minlength=len(COMPOUND) + 1)
        cc[0] = 0
        comp_of[nm] = COMPOUND[int(cc.argmax()) - 1] if cc.sum() else "Other"
    # 官方 13 合并组掩膜不覆盖太平洋海岸条带, 太平洋侧四区归入自定义 Pacific 组
    for nm in ("PacificNW", "PacificSW", "Cascades", "SierraNevada"):
        if nm in final:
            comp_of[nm] = "Pacific"
    comp_names = sorted({v for v in comp_of.values() if v})
    comp_map = np.zeros_like(ids)
    for r, nm in enumerate(final, 1):
        c = comp_of.get(nm)
        if c in comp_names:
            comp_map[ids == r] = comp_names.index(c) + 1

    share = {nm: round(100 * cnt[k] / n_land, 2) for nm, k in zip(final, keep)}
    print(f"[regions] 最终 {len(final)} 区: " +
          ", ".join(f"{nm} {share[nm]}%" for nm in final))

    spec = {"version": a.version,
            "basis": "Bukovsky (2011) NARCCAP regionalization: Ricketts 生态区简化版"
                     "(NEON 对齐), 官方 0.5° 掩膜最近邻重栅格化到 Daymet 1/24° 网格",
            "citation": "Bukovsky, M.S., 2011: Masks for the Bukovsky regionalization "
                        "of North America, RISC, IMAGe, NCAR, Boulder, CO",
            "splits": spec_splits, "merged_scraps": merged, "merged_orphans": orphans,
            "scrap_frac": a.scrap_frac, "land_share_pct": share,
            "compound_of": comp_of}
    np.savez_compressed(out / f"regions_{a.version}.npz",
                        region_id=ids.astype(np.int16),
                        region_names=np.array(final),
                        compound_id=comp_map.astype(np.int16),
                        compound_names=np.array(comp_names),
                        land=land, spec=json.dumps(spec, ensure_ascii=False))
    json.dump(spec, open(out / f"regions_{a.version}_spec.json", "w"), indent=1,
              ensure_ascii=False)

    draw_map(out / "regions_map.png", ids, final, oro, land,
             f"domain regions {a.version}  ({len(final)} regions; Bukovsky-based, "
             "terrain-refined; elevation-colored relief)")
    draw_map(out / "regions_compound_map.png", comp_map, comp_names, oro, land,
             f"compound regions  ({len(comp_names)} groups)")
    draw_map(out / "regions_west_zoom.png", ids, final, oro, land,
             "western regions with terrain splits (Cascades / SierraNevada)",
             crop=(-125.1, -101.0, 28.0, 53.6), figsize=(11.0, 9.2),
             fontsize=8.6, label_min_px=800)
    print(f"[regions] 完成 -> {out}")


if __name__ == "__main__":
    main()
