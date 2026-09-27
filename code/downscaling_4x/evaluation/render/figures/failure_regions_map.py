# -*- coding: utf-8 -*-
"""失败地区分类图 —— 地形晕渲底 + 两类失败区浅色叠加(低透明, 不压地形)。

失败区名单由 --failure-regions 人工从错误率排名里给定。按阶段A得分分两类:
  类1「A就错, B也错」= 区 mean MAE_A 高(A 本就是误差源);
  类2「A本好, B反而差」= 区 mean MAE_A ≤ 全域(区间)中位 且 CRPSS<0(加 B 确实恶化)。
底图复用 build_regions 的高程着色晕渲(LightSource + terrain), 叠加用低 alpha 让地形透出。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from downscaling_4x.data import grid as G
from downscaling_4x.tools.plotting import geo_overlay as GEO
from downscaling_4x.evaluation.render.registry import figure

C1 = "#c0392b"   # 类1: 暖红
C2 = "#5b3a91"   # 类2: 紫


@figure("failure_regions_map", needs=("crps",), full_year=True)
def render(ctx):
    fr = ctx.failure_regions
    if not fr:
        print("[failure_regions_map] 未给 --failure-regions, 跳过")
        return []
    try:
        ctx.ensure_mae_a()
        Ma, Ka = ctx.agg.mass_matrix("mae_a", "region")
    except RuntimeError as e:
        print(f"[failure_regions_map] 跳过(缺 μ/truth): {e}")
        return []
    Mb, Kb = ctx.agg.mass_matrix("crps", "region")
    crps_b = Mb[1:, 1:].sum(1) / np.maximum(Kb[1:, 1:].sum(1), 1)
    mae_a = Ma[1:, 1:].sum(1) / np.maximum(Ka[1:, 1:].sum(1), 1)
    crpss = 1.0 - crps_b / np.maximum(mae_a, 1e-9)
    med_a = float(np.median(mae_a))
    idx = {n: i for i, n in enumerate(ctx.region_names)}

    type1, type2 = [], []
    for n in fr:
        if n not in idx:
            print(f"[failure_regions_map] 未知区名 {n}, 跳过")
            continue
        i = idx[n]
        (type2 if (mae_a[i] <= med_a and crpss[i] < 0) else type1).append(n)

    land = ctx.land
    H, W = land.shape
    ext = G.extent()
    aspect = G.aspect()
    ls = mcolors.LightSource(azdeg=315, altdeg=45)
    norm = mcolors.Normalize(vmin=-1200.0, vmax=3600.0)
    rgb = ls.shade(np.where(land, ctx.elevation(), 0.0), cmap=plt.get_cmap("terrain"),
                   norm=norm, blend_mode="soft", vert_exag=0.02,
                   dx=(ext[1] - ext[0]) / W, dy=(ext[3] - ext[2]) / H)
    rgb[~land] = mcolors.to_rgba("#dce8f2")
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=ext, aspect=aspect, interpolation="bilinear", zorder=0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    overlay = np.zeros((H, W, 4))
    for n in type1:
        overlay[ctx.region_id == idx[n] + 1] = (*mcolors.to_rgb(C1), 0.32)
    for n in type2:
        overlay[ctx.region_id == idx[n] + 1] = (*mcolors.to_rgb(C2), 0.32)
    ax.imshow(overlay, origin="lower", extent=ext, aspect=aspect, interpolation="nearest", zorder=2)

    lonc = np.linspace(ext[0], ext[1], W, endpoint=False) + (ext[1] - ext[0]) / W / 2
    latc = np.linspace(ext[2], ext[3], H, endpoint=False) + (ext[3] - ext[2]) / H / 2
    # 全部分区的边界(细黑线), 让失败区能在完整分区图里定位
    for k in range(1, len(ctx.region_names) + 1):
        sel = (ctx.region_id == k)
        if sel.any():
            ax.contour(lonc, latc, sel.astype(float), levels=[0.5],
                       colors="#101418", linewidths=0.7, zorder=3)
    for n in fr:
        if n not in idx:
            continue
        rid_k = idx[n] + 1
        sel = (ctx.region_id == rid_k)
        ax.contour(lonc, latc, sel.astype(float), levels=[0.5],
                   colors=(C2 if n in type2 else C1), linewidths=1.4, zorder=4)
        ys, xs = np.nonzero(sel & land)
        if len(ys):
            lon = ext[0] + (xs.mean() + 0.5) * (ext[1] - ext[0]) / W
            lat = ext[2] + (ys.mean() + 0.5) * (ext[3] - ext[2]) / H
            ax.text(lon, lat, str(ctx.display_id(n)), fontsize=11, fontweight="bold",
                    color="#101418", ha="center", va="center", zorder=6,
                    path_effects=GEO._halo())

    handles = [mpatches.Patch(color=C1, alpha=0.55, label="type1: A wrong, B wrong"),
               mpatches.Patch(color=C2, alpha=0.55, label="type2: A good, B worse")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"failure regions classified by Stage-A  ·  {ctx.target}", fontsize=10)
    t1 = [f"{ctx.display_id(n)}:{n}" for n in type1]
    t2 = [f"{ctx.display_id(n)}:{n}" for n in type2]
    print(f"[failure_regions_map] 类1(A错B错)={t1} | 类2(A好B差)={t2} | 区间中位MAE_A={med_a:.3f}")
    return ctx.savefig(fig, "failure_regions_map")
