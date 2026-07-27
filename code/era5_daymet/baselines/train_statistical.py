#!/usr/bin/env python
# Packaged implementation; code/train_statistical.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
train_statistical.py — 统计降尺度基线 (ERA5 120x240 -> Daymet 720x1440, 6x)
============================================================================
不需要 GPU / 不训练网络, 给出任何深度模型必须超过的下限。本地就能跑。
  train = 1980-2017 (--train-years)  在这些年上拟合逐像素 BCSD 回归
  test  = 2020      (--test-year)    评测
评测域: Daymet land_sea_mask>0.5 (域外是常数填充, 一律排除)。

四个基线:
  nearest : 6x6 块复制(平凡下限)
  bilinear: 双线性插值(标准下限)
  bicubic : 双三次插值(scipy.ndimage.zoom order=3; 无 scipy 则自动跳过)
  bcsd    : 逐 HR 像素线性回归  Daymet ≈ a·bilinear(ERA5) + b
            a,b 在训练年最小二乘拟合; 逐像素 intercept 等价学到 ERA5
            缺失的静态订正(地形/海岸线)。
降水在 log1p(mm) 空间评测(已在表头标注 unit)。

用法:
  自测(合成数据, 秒级, 不需 GPU/torch 可选):
      python train_statistical.py --smoke
  正式:
      python train_statistical.py --stats-dir unused \
          --era5-dir era5/0.25_deg_Daily_Golden_DaymetAligned \
          --daymet-dir daymet/2.5_arcmin \
          --train-years $(seq 1980 2017) --test-year 2020 \
          --out runs/stat --workers 3
输出: {out}/metrics_test{year}.json  +  {out}/demo_{var}_day###.png
============================================================================
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.paths import PROJECT_ROOT
from era5_daymet.data.downscale_baseline import (
    Acc,
    FACTOR,
    LAT_EDGES,
    LON_EDGES,
    PRECIP,
    _load_raw_static,
    _squeeze_2d,
    fill_nan_daymean,
    make_bilinear,
    nearest_day,
    npz_memmap,
)

try:
    from scipy.ndimage import zoom as _zoom
except Exception:
    _zoom = None


def _hr_day(hrmm, t, tf):
    d = np.asarray(hrmm[t])
    if d.ndim == 3:                      # (1,720,1440)
        d = d[0]
    return tf(d.astype(np.float32))


def _bicubic(a2d, f=FACTOR):
    """双三次上采样 (H,W)->(H*f,W*f); 无 scipy 返回 None。"""
    if _zoom is None:
        return None
    return _zoom(a2d, f, order=3, mode="nearest").astype(np.float32)


def _demo_fig(var, unit, lr2d, panels, tgt2d, mask, di, year, out_dir):
    ext = [LON_EDGES[0], LON_EDGES[1], LAT_EDGES[0], LAT_EDGES[1]]
    def msk(a):
        o = a.astype(np.float32).copy(); o[~mask] = np.nan; return o
    tgt_m = msk(tgt2d)
    vmin = float(np.nanpercentile(tgt_m, 2)); vmax = float(np.nanpercentile(tgt_m, 98))
    cells = [("ERA5 -> nearest 6x", msk(nearest_day(lr2d)))]
    cells += [(name, msk(img)) for name, img in panels]
    cells += [("Daymet target", tgt_m)]
    bcsd = dict(panels).get("BCSD")
    if bcsd is not None:
        cells.append(("|BCSD - Daymet|", np.abs(msk(bcsd) - tgt_m)))
    n = len(cells); cols = 3; rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(5.3 * cols, 4.4 * rows))
    ax = np.atleast_2d(ax)
    for k in range(rows * cols):
        r, c = divmod(k, cols); a = ax[r, c]
        if k >= n:
            a.axis("off"); continue
        title, img = cells[k]
        err = title.startswith("|")
        im = a.imshow(img, origin="lower", extent=ext, cmap=("magma" if err else "viridis"),
                      vmin=(0 if err else vmin), vmax=(float(np.nanpercentile(img, 98)) or 1.0 if err else vmax),
                      aspect="auto")
        a.set_title(f"{title}  [{unit}]", fontsize=9); fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f"{year} {var}  test day_idx={di}  ERA5->Daymet 6x 统计基线", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"demo_{var}_day{di:03d}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)


def evaluate_var(task):
    """train_years 上拟合逐像素 BCSD; test_year 上评测 nearest/bilinear/bicubic/bcsd。"""
    var, era5_dir, daymet_dir, train_years, test_year, out_dir, train_stride = task
    try:
        precip = (var == PRECIP)
        tf = (lambda a: np.log1p(np.maximum(a, 0) * 1000.0)) if precip else (lambda a: a)  # 米->mm->log1p
        unit = "log1p(mm)" if precip else "K"
        up = None                                          # 按实际 LR 网格懒构建(real=120x240, smoke=24x24)
        df_te = M.find_year_files(daymet_dir, test_year)
        if not df_te:
            return var, {"error": f"缺 test {test_year} Daymet"}
        mask = _squeeze_2d(_load_raw_static(df_te, M.LAND_MASK_VAR)) > 0.5

        # ---- train_years 上逐像素最小二乘累计 ----
        Sx = np.zeros(mask.shape, np.float64); Sy = Sx.copy(); Sxx = Sx.copy(); Sxy = Sx.copy()
        ntr = 0
        for ty in train_years:
            ef = M.find_year_files(era5_dir, ty); df = M.find_year_files(daymet_dir, ty)
            lr = M.load_var_stack(ef, var); hrmm = npz_memmap(df[0], var) if df else None
            if lr is None or hrmm is None:
                return var, {"error": f"train {ty} 缺 {var} 或 Daymet 非未压缩"}
            if up is None:
                up = make_bilinear(lr.shape[1], lr.shape[2], FACTOR)
            lrf = fill_nan_daymean(tf(lr))
            for t in range(0, lr.shape[0], train_stride):
                x = up(lrf[t]); y = _hr_day(hrmm, t, tf)
                Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; ntr += 1
        den = ntr * Sxx - Sx * Sx
        a = np.where(den != 0, (ntr * Sxy - Sx * Sy) / den, 1.0).astype(np.float32)
        b = np.where(den != 0, (Sy - a.astype(np.float64) * Sx) / max(ntr, 1), 0.0).astype(np.float32)

        # ---- test_year 上评测 ----
        ef = M.find_year_files(era5_dir, test_year)
        lr = M.load_var_stack(ef, var); hrmm = npz_memmap(df_te[0], var)
        if lr is None or hrmm is None:
            return var, {"error": f"test {test_year} 缺 {var}"}
        lrf = fill_nan_daymean(tf(lr)); Tt = lr.shape[0]
        methods = ["nearest", "bilinear", "bcsd"] + (["bicubic"] if _zoom is not None else [])
        accs = {m: Acc() for m in methods}
        for t in range(Tt):
            x = up(lrf[t]); y = _hr_day(hrmm, t, tf); tgt = y[mask]
            accs["bilinear"].add(x[mask], tgt)
            accs["bcsd"].add((a * x + b)[mask], tgt)
            accs["nearest"].add(nearest_day(lrf[t])[mask], tgt)
            if _zoom is not None:
                accs["bicubic"].add(_bicubic(lrf[t])[mask], tgt)
        res = {m: accs[m].result() for m in methods}

        di = min(195, Tt - 1)
        xd = up(lrf[di])
        panels = [("Bilinear", xd), ("BCSD", a * xd + b)]
        if _zoom is not None:
            panels.insert(1, ("Bicubic", _bicubic(lrf[di])))
        _demo_fig(var, unit, lrf[di], panels, _hr_day(hrmm, di, tf), mask, di, test_year, out_dir)
        res.update(unit=unit, demo_day=di, train_years=[int(y) for y in train_years],
                   test_year=int(test_year), n_train_days=ntr, scipy=(_zoom is not None))
        return var, res
    except Exception as e:
        import traceback
        return var, {"error": str(e), "where": traceback.format_exc().strip().splitlines()[-1]}


def main():
    ap = argparse.ArgumentParser(description="统计降尺度基线 (nearest/bilinear/bicubic/BCSD)",
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--vars", nargs="+", default=M.CHECK_VARS)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs/stat"))
    ap.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"], help="拟合 BCSD 的年(默认 1980-2017)")
    ap.add_argument("--test-year", type=int, default=M.splits["test"][0], help="评测年(默认 2020)")
    ap.add_argument("--train-stride", type=int, default=1, help="训练天下采样(>1 更快, 略降精度)")
    ap.add_argument("--workers", type=int, default=3, help="按变量并行")
    ap.add_argument("--stats-dir", default=None, help="(统计基线用不到, 仅为命令行口径统一保留)")
    ap.add_argument("--smoke", action="store_true", help="合成数据秒级自测(不需 GPU)")
    args = ap.parse_args()

    if args.smoke:
        import tempfile
        from era5_daymet.training import train_downscale as TD
        root = tempfile.mkdtemp()
        era5, daymet, _ = TD.make_synth(root)
        args.era5_dir, args.daymet_dir = era5, daymet
        args.vars = TD.TARGETS; args.train_years = [2018, 2019]; args.test_year = 2020
        args.out = os.path.join(root, "stat"); args.workers = 1

    tasks = [(v, args.era5_dir, args.daymet_dir, args.train_years, args.test_year, args.out, args.train_stride)
             for v in args.vars]
    print("=" * 74)
    print(f"统计基线: train={args.train_years[0]}-{args.train_years[-1]} -> test={args.test_year}  "
          f"vars={args.vars}  bicubic={'on' if _zoom is not None else 'off(无 scipy)'}")
    print("=" * 74)
    results = {}
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            for v, r in ex.map(evaluate_var, tasks):
                results[v] = r
    else:
        for tsk in tasks:
            v, r = evaluate_var(tsk); results[v] = r

    print(f"\n{'variable':28s} {'method':9s} {'RMSE':>9s} {'MAE':>9s} {'bias':>9s} {'corr':>7s}")
    print("-" * 75)
    for v, r in results.items():
        if "error" in r:
            print(f"{v:28s} ERROR: {r['error']}"); continue
        for m in ("nearest", "bilinear", "bicubic", "bcsd"):
            if m not in r:
                continue
            mm = r[m]
            print(f"{v:28s} {m:9s} {mm['rmse']:9.3f} {mm['mae']:9.3f} {mm['bias']:9.3f} {mm['corr']:7.3f}")
        print("-" * 75)
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"metrics_test{args.test_year}.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n指标 JSON: {outp} ; demo 图在同目录")


if __name__ == "__main__":
    main()
