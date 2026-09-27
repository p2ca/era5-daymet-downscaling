#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_capacity_vs_terrain.py — 路由容量与地形复杂度的关系(地图 + 定量 + 残差)
============================================================================
逐像素平均激活专家数 k 与地形的四张单图:

  relief_k_contour   地形晕渲作底(高程着色 + 山体阴影), k 用★实线等值线★叠加
  k_map_oro_contour  k 用填色, 高程用★细实线等值线★叠加
  k_vs_roughness     逐像素 (地形粗糙度, k) 的二维密度 + 分箱中位数与四分位带;
                     固定 top-K 的模型在同一张上是一条水平线, 对比直观
  k_residual_map     k 减去"粗糙度单独能解释的部分"(按粗糙度分位分箱的中位数)后的残差图,
                     回答"这是不是只是个地形检测器"

★不使用半透明填色压地形★: 两层信息一层用面、一层用线, 各占一个通道。

k 的算法不做 token->像素展开: 每个 token 覆盖 patch×patch 块, 用按 offset 缓存的
"像素 -> token 下标"直接 gather, 每天一次 219k 的取值, 与展开成 (层, 专家, 像素) 等价但便宜得多。
地形粗糙度 = P×P 块内高程标准差落回像素, 与 stage_b_pairwise_regions 的定义一致。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_capacity_vs_terrain \\
      --model DEC=runs/exp/<eval2020> --flat TC=2.0 \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag> [--members 0]
============================================================================
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=值 形式, 得到 {s!r}")
    a, b = s.split("=", 1)
    return a.strip(), b.strip()


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def block_roughness(oro, land, P):
    """P×P 块内高程标准差, 落回像素(块内只计陆地格点)。"""
    H, W = land.shape
    z = np.where(land, oro, 0.0)
    m = land.astype(np.float64)
    ps = lambda x: x.reshape(H // P, P, W // P, P).sum((1, 3))
    n = ps(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = ps(z * z) / np.maximum(n, 1) - (ps(z) / np.maximum(n, 1)) ** 2
    return np.repeat(np.repeat(np.sqrt(np.maximum(var, 0.0)), P, 0), P, 1)


def pixel_k(dump, land, members, year, days=None):
    """逐像素平均激活专家数(层与文件上平均), 用 token 下标 gather, 不展开。"""
    info = RD.load_routing_meta(dump)
    gh, gw = info["token_grid"]
    patch, L = info["patch"], info["n_moe_layers"]
    items = [(y, d, m) for (y, d, m) in RD.available_routing(dump)
             if y == year and m in members and (days is None or d in days)]
    if not items:
        raise SystemExit(f"{dump}: 没有符合条件的路由文件")
    Y, X = np.nonzero(land)
    cache, acc, n = {}, np.zeros(Y.size), 0
    t0 = time.time()
    for i, (y, d, m) in enumerate(items):
        r = RD.load_routing(dump, y, d, m)
        key = (int(r["offset"][0]), int(r["offset"][1]))
        if key not in cache:
            cache[key] = ((Y + key[0]) // patch) * gw + ((X + key[1]) // patch)
        acc += (r["k_sum"].sum(0) / (float(r["nfwd"]) * L))[cache[key]]
        n += 1
        if (i + 1) % 60 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return acc / max(n, 1), len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir(含 routing/)")
    ap.add_argument("--flat", action="append", default=None, help="label=常数 k 的模型(如 TC=2.0), 只画进定量图")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--patch", type=int, default=16, help="粗糙度的块边长(px)")
    ap.add_argument("--bins", type=int, default=40, help="粗糙度分箱数(等频)")
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    flats = [(l, float(v)) for l, v in (parse_spec(s) for s in (a.flat or []))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(Path(specs[0][1]), a.regions, C.TARGETS[0], out, years=(a.year,),
                        model_tag="capacity-terrain", scales_path=out / "scales.json")
    land = ctx.land
    H, W = land.shape
    oro = ctx.elevation()
    rough_px = block_roughness(oro, land, a.patch)
    rough = rough_px[land]
    ext = G.extent(); aspect = G.aspect()
    lonc = np.linspace(ext[0], ext[1], W, endpoint=False) + (ext[1] - ext[0]) / W / 2
    latc = np.linspace(ext[2], ext[3], H, endpoint=False) + (ext[3] - ext[2]) / H / 2
    res = {"year": a.year, "members": a.members, "patch": a.patch, "models": {}}
    saved = {"rough": rough_px, "land": land, "oro": oro}

    for label, d in specs:
        k1, nfiles = pixel_k(Path(d), land, a.members, a.year)
        kmap = np.full((H, W), np.nan); kmap[land] = k1
        saved[f"k_{slug(label)}"] = kmap
        sl = slug(label)

        # ① 地形晕渲作底 + k 的实线等值线
        ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
        norm = matplotlib.colors.Normalize(vmin=-1200.0, vmax=3600.0)
        cmap = plt.get_cmap("terrain")
        rgb = ls.shade(np.where(land, oro, 0.0), cmap=cmap, norm=norm, blend_mode="soft",
                       vert_exag=0.02, dx=(ext[1] - ext[0]) / W, dy=(ext[3] - ext[2]) / H)
        rgb[~land] = matplotlib.colors.to_rgba("#dce8f2")
        fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
        ax.imshow(rgb, origin="lower", extent=ext, aspect=aspect, interpolation="bilinear", zorder=0)
        lv = [x for x in (1.0, 2.0, 4.0, 6.0, 8.0) if np.nanmin(kmap) < x < np.nanmax(kmap)]
        # 域外保持 NaN: 填 0 会在海岸线上画出一条不存在的等值线
        cs = ax.contour(lonc, latc, np.ma.masked_invalid(kmap), levels=lv,
                        colors="#111111", linewidths=1.3, zorder=4)
        ax.clabel(cs, fmt="k=%.0f", fontsize=8)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.01); cb.set_label("elevation [m]")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label} · terrain relief with iso-contours of routed experts", fontsize=10)
        ctx.savefig(fig, f"relief_k_contour_{sl}")

        # ② k 填色 + 高程实线等值线
        vmin, vmax, cm2 = ctx.scale("k_map", data=kmap, cmap="magma", label="mean routed experts")
        fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
        im = ax.imshow(kmap, origin="lower", extent=ext, aspect=aspect, cmap=cm2,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_facecolor("0.85")
        ax.contour(lonc, latc, np.where(land, oro, np.nan), levels=[500, 1000, 1500, 2000, 2500, 3000],
                   colors="#f2f2f2", linewidths=0.5, alpha=1.0, zorder=3)
        fig.colorbar(im, ax=ax, shrink=0.78, label="mean routed experts")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label} · routed experts, elevation contours (500 m)", fontsize=10)
        ctx.savefig(fig, f"k_map_oro_contour_{sl}")

        # ③ 定量: 粗糙度 vs k
        q = np.quantile(rough, np.linspace(0, 1, a.bins + 1))
        q = np.unique(q)
        idx = np.clip(np.digitize(rough, q[1:-1]), 0, len(q) - 2)
        cx = np.array([np.median(rough[idx == i]) for i in range(len(q) - 1)])
        med = np.array([np.median(k1[idx == i]) for i in range(len(q) - 1)])
        p25 = np.array([np.percentile(k1[idx == i], 25) for i in range(len(q) - 1)])
        p75 = np.array([np.percentile(k1[idx == i], 75) for i in range(len(q) - 1)])
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        ax.hexbin(rough, k1, gridsize=70, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        ax.fill_between(cx, p25, p75, color="#c0392b", alpha=0.18, lw=0, label=f"{label} IQR")
        ax.plot(cx, med, color="#c0392b", lw=2.0, label=f"{label} median")
        for fl, fv in flats:
            ax.axhline(fv, color="#2c3e50", lw=1.8, ls="--", label=f"{fl} (fixed top-K = {fv:g})")
        from scipy.stats import spearmanr
        rho = float(spearmanr(rough, k1).correlation)
        ax.set_xlim(0, float(np.percentile(rough, 99.5)))   # 右端极稀疏的尾巴不占版面
        ax.set_xlabel(f"terrain roughness: elevation std in {a.patch}x{a.patch} block [m]")
        ax.set_ylabel("mean routed experts per token")
        ax.set_title(f"{label} · capacity vs terrain roughness (Spearman ρ = {rho:.3f})", fontsize=10)
        ax.legend(fontsize=8)
        ctx.savefig(fig, f"k_vs_roughness_{sl}")

        # ④ 残差: k 减去粗糙度分箱中位数
        khat = med[idx]
        resid = np.full((H, W), np.nan); resid[land] = k1 - khat
        lim = float(np.nanpercentile(np.abs(resid), 99))
        fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
        im = ax.imshow(resid, origin="lower", extent=ext, aspect=aspect, cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_facecolor("0.85"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.78, label="k − median k at same roughness")
        ax.set_title(f"{label} · what routing does beyond terrain roughness", fontsize=10)
        ctx.savefig(fig, f"k_residual_map_{sl}")
        saved[f"resid_{sl}"] = resid

        res["models"][label] = {"dump": d, "n_files": nfiles, "spearman_k_roughness": rho,
                                "k_mean": float(k1.mean()), "k_p01": float(np.percentile(k1, 1)),
                                "k_p99": float(np.percentile(k1, 99)),
                                "k_median_lowest_rough_decile": float(np.median(k1[rough <= np.quantile(rough, 0.1)])),
                                "k_median_highest_rough_decile": float(np.median(k1[rough >= np.quantile(rough, 0.9)])),
                                "resid_abs_median": float(np.nanmedian(np.abs(k1 - khat)))}
        print(f"[{label}] ρ(粗糙度,k)={rho:.3f}  k 均值 {k1.mean():.3f}  "
              f"最平十分位中位 {res['models'][label]['k_median_lowest_rough_decile']:.3f}  "
              f"最陡十分位中位 {res['models'][label]['k_median_highest_rough_decile']:.3f}", flush=True)

    ctx.write_scales()
    np.savez_compressed(out / "capacity_terrain.npz", **saved)
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
