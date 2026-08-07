#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
EDA ③ 为什么 BCSD 赢 tmax (登录节点, 纯数据+已存系数, 不用模型)
------------------------------------------------------------------
BCSD = 逐像素仿射 a*bilinear(ERA5)+b, 截距 b 吃下"时间不变的精细订正"(地形/海岸)。
=> BCSD 只能复现降尺度细节里的【静态部分】, 复现不了【随天变化】的精细结构。
本脚本量化: 每个变量的降尺度细节 detail_t = truth_t - bilinear_t 里, 静态成分占多少。
  frac_static = Var_space(clim_detail) / (Var_space(clim_detail) + mean_t Var_space(anom_t))
  frac_static 高 => 细节几乎是固定花样 => BCSD 的 b 直接复现 => 赢(tmax 预期)。
  frac_static 低 => 细节随天变 => BCSD 静态模板抓不住 => 输(tmin/precip 预期)。
温度在 K, 降水在 log1p(mm)(对齐 BCSD 拟合空间)。
输出: runs/exp/20260720-eda-bcsd-why/{bcsd_why.png, summary.json}
"""
import os, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.data.downscale_baseline import make_bilinear
from era5_daymet.training import train_downscale as TD
from era5_daymet.paths import PROJECT_ROOT

# 路径一律相对仓库根锚定, 从任何 cwd 运行都写入正式 runs/ (不再在 code/ 下误建目录)
STATS  = str(PROJECT_ROOT / "runs/stats/train_dayofyear")
COEFS  = str(PROJECT_ROOT / "runs/bcsd_coefs")
FACTOR = 6
VARS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
LAB  = {"2m_temperature_max": "tmax (K)", "2m_temperature_min": "tmin (K)", "total_precipitation_24hr": "precip log1p(mm)"}
PRECIP = "total_precipitation_24hr"
logmm = lambda x: np.log1p(np.maximum(x, 0.0) * 1000.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=1, help="1=全年(汇报口径); 调大只为快速探索")
    ap.add_argument("--box", type=int, default=384, help="全陆地取样方块边长(px)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs/exp/20260720-eda-bcsd-why"))
    args = ap.parse_args()
    YEAR, OUTDIR = args.year, args.out
    os.makedirs(OUTDIR, exist_ok=True)

    stats = TD.Stats(STATS, VARS, VARS)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [YEAR], VARS, VARS, stats)
    up = make_bilinear(d.Hl, d.Wl, FACTOR)
    mask = d.mask[YEAR]; by, bx, bs = TD.pick_land_box(mask, min(args.box, d.H, d.W))
    days = list(range(0, d.ndays[YEAR], args.stride))
    print(f"[eda3] box=({by},{bx},{bs}) all-land={mask[by:by+bs,bx:bx+bs].all()}  {len(days)} days", flush=True)

    res = {}; bmaps = {}; climmaps = {}
    for v in VARS:
        prec = (v == PRECIP); tf = logmm if prec else (lambda x: x)
        # 收集 detail_t
        det = []
        for t in days:
            truth = tf(np.asarray(d._hr(YEAR, v, t)).astype(np.float32))
            bil = up(tf(d.lr[YEAR][v][t].astype(np.float32)))
            det.append((truth - bil)[by:by+bs, bx:bx+bs])
        det = np.stack(det, 0)                      # (T,bs,bs)
        clim = det.mean(0)                          # 静态成分
        anom = det - clim[None]                     # 随天变化成分
        static_power = float(clim.var())
        anom_power = float(anom.var(axis=(1, 2)).mean())
        frac = static_power / (static_power + anom_power + 1e-12)
        res[v] = dict(frac_static=frac, static_power=static_power, anom_power=anom_power)
        climmaps[v] = clim
        # BCSD 截距 b (它免费借来的静态结构)
        cf = np.load(os.path.join(COEFS, f"{v}.npz"))
        bmaps[v] = cf["b"][by:by+bs, bx:bx+bs]
        print(f"[eda3] {v:30s} frac_static = {frac:.3f}  (static={static_power:.4g} anom={anom_power:.4g})", flush=True)

    # ---- 画图: 每变量一列: [detail静态成分 clim] [BCSD截距b] ; 底部 frac_static 柱 ----
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.25)
    for j, v in enumerate(VARS):
        a0 = fig.add_subplot(gs[0, j])
        im = a0.imshow(climmaps[v], origin="lower", cmap="RdBu_r"); fig.colorbar(im, ax=a0, fraction=.046)
        a0.set_title(f"detail STATIC part (clim)\n{LAB[v]}", fontsize=10); a0.set_xticks([]); a0.set_yticks([])
        a1 = fig.add_subplot(gs[1, j])
        im = a1.imshow(bmaps[v], origin="lower", cmap="viridis"); fig.colorbar(im, ax=a1, fraction=.046)
        a1.set_title(f"BCSD intercept b (borrowed static)\n{LAB[v]}", fontsize=10); a1.set_xticks([]); a1.set_yticks([])
    axb = fig.add_subplot(gs[2, :])
    names = [LAB[v] for v in VARS]; fr = [res[v]["frac_static"] for v in VARS]
    bars = axb.bar(names, fr, color=["#2c7fb8", "#7fcdbb", "#d95f0e"])
    for b, f in zip(bars, fr): axb.text(b.get_x()+b.get_width()/2, f+0.01, f"{f:.2f}", ha="center", fontsize=11)
    axb.set_ylim(0, 1); axb.set_ylabel("frac_static  (higher = BCSD-friendly)")
    axb.set_title("static fraction of downscaling detail:  high -> BCSD reproduces it via intercept b -> BCSD wins")
    axb.grid(True, axis="y", alpha=.3)
    fig.suptitle(f"EDA-3  why BCSD wins tmax:  detail = static(terrain, BCSD gets free via b) + daily-varying(only NN can chase)\n"
                 f"year={YEAR}  {len(days)} days  box={bs}px", fontsize=12)
    fig.savefig(os.path.join(OUTDIR, "bcsd_why.png"), dpi=130, bbox_inches="tight")
    json.dump(dict(year=YEAR, days=days, box=bs, vars=res), open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1)
    print(f"[eda3] saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
