#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
export_calibration_bundle.py — 把落场打成一个自包含的校准数据包
============================================================================
只导出校准需要的东西, 量化成整型以便传输, 并把反量化参数与口径写进 meta.json,
使这个包离开工作区之后仍然可读、可验证。

包内结构:
  meta.json      口径、来源实验、量化参数、逐场的自检结果
  pixels.npz     每个导出格点的 row/col/lon/lat/region_id/elevation/roughness
  days.npz       日索引、月份、doy
  truth.npz      (n_days, n_pixels) 真值
  <label>.npz    每个模型一份: ens_mean / spread / crps / rank
  factors.npz    逐 (天, 区域) 的 13 个 ERA5 当日因子(若给了 --factors)

量化: 温度类用 offset 273.15、scale 0.01 K; spread 与 crps 用 offset 0、scale 0.001;
rank 原样 int8。反量化 = v * scale + offset, 参数在 meta.json 的 quant 段。

自检: 每个模型从导出的(反量化后)数组重算池化 CRPS, 与该落场 metrics.json 对拍,
差超过量化分辨率即报错 —— 量化写错不会有任何东西报错, 只是每个数都偏了。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.export_calibration_bundle \\
      --model MU=runs/exp/<jda1>/fields2020/<target> --model DEC=runs/exp/<dec eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz --stride 1 --out <bundle dir>
============================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import parse_spec, slug

# 量化分辨率: 温度用 0.002 K(int16 覆盖 207.6–338.7 K), spread/crps 用 0.001。
# 相对 CRPS 0.55 K 与最细波长带 MAE 0.13 K 都在千分之几, 可忽略; 改动它要重算下面的范围检查。
QUANT = {"truth": (273.15, 0.002), "ens_mean": (273.15, 0.002),
         "spread": (0.0, 0.001), "crps": (0.0, 0.001)}


LOADER = '#!/usr/bin/env python\n"""校准数据包的加载示例; 本地只需要 numpy。口径见同目录 meta.json。\n\n内存提示: 每个场是 (n_days, n_pixels), float32 约 320 MB(全分辨率)。逐个模型用完即释放,\n不要一次把所有模型全展开。只要某些区域或月份时, 先按 pixels.npz 的 region_id 或\ndays.npz 的 month 取下标, 再反量化。\n"""\nimport json\nimport numpy as np\nfrom pathlib import Path\n\nHERE = Path(__file__).parent\nMETA = json.load(open(HERE / "meta.json"))\n\n\ndef load(name, keys=None, dtype=np.float32):\n    """按 meta.json 的 quant 段反量化; 未登记的字段原样返回。keys 可只取需要的场。"""\n    z = np.load(HERE / f"{name}.npz")\n    out = {}\n    for k in (keys or z.files):\n        qz = META["quant"].get(k)\n        out[k] = (z[k].astype(dtype) * dtype(qz["scale"]) + dtype(qz["offset"])) if qz else z[k]\n    return out\n\n\nif __name__ == "__main__":\n    px, dy = load("pixels"), load("days")\n    truth = load("truth")["truth"]\n    print(f"格点 {truth.shape[1]:,}  天 {truth.shape[0]}  目标 {META[\'target\']}  年 {META[\'year\']}")\n    for lab in META["sources"]:\n        have = np.load(HERE / f"{lab}.npz").files\n        m = load(lab, keys=[k for k in ("ens_mean", "spread", "crps") if k in have])\n        err = m["ens_mean"] - truth\n        rmse = float(np.sqrt((err.astype(np.float64) ** 2).mean()))\n        line = (f"{lab:<5} CRPS {m[\'crps\'].mean():.4f} (集群记录 "\n                f"{META[\'selfcheck\'][lab][\'metrics_json_crps\']})  "\n                f"MAE {np.abs(err).mean():.4f}  RMSE {rmse:.4f}")\n        if "spread" in m:\n            sp = float(m["spread"].mean())\n            line += f"  spread {sp:.4f}  spread/RMSE {sp / rmse:.4f}"\n        print(line)\n        del m, err\n    lab0 = next(l for l in META["sources"] if "rank" in np.load(HERE / f"{l}.npz").files)\n    r = np.load(HERE / f"{lab0}.npz")["rank"].astype(np.int64).ravel()\n    nb = int(r.max()) + 1\n    h = np.bincount(r[r >= 0], minlength=nb) / max((r >= 0).sum(), 1)\n    print()\n    print(f"{lab0} 名次直方图({nb} 档, 理想每档 {1 / nb:.4f}):")\n    print("  " + " ".join(f"{v:.4f}" for v in h))\n    print(f"  两端两档合计 {h[0] + h[-1]:.4f}  (理想 {2 / nb:.4f}; 偏高 = 欠离散)")\n\n    # EMOS/NGR 起手: 逐 (区域, 月) 拟合 mu\' = a + b*ens_mean, sigma\' = c + d*spread,\n    # 直接最小化高斯 CRPS(有闭式解)。系数必须用留出的日子拟合, 否则是自己拟合自己。\n'


def q(arr, name):
    off, sc = QUANT[name]
    v = np.rint((arr - off) / sc)
    if np.nanmin(v) < -32768 or np.nanmax(v) > 32767:
        raise SystemExit(f"{name} 量化后超出 int16 范围, 需调整 scale")
    return v.astype(np.int16)


def dq(arr, name):
    off, sc = QUANT[name]
    return arr.astype(np.float64) * sc + off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=场目录, 可多次")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=1, help="空间抽样步长; 1 = 全部有效格点")
    ap.add_argument("--factors", default=None, help="factor_screen_regions 落的 factors.npz, 顺带打包")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y = a.year
    ctx = RenderContext(specs[0][1], a.regions, a.target, out, years=(y,), model_tag="bundle")
    land = ctx.land
    H, W = land.shape
    rows, cols = np.nonzero(land)
    keep = np.ones(rows.size, bool) if a.stride <= 1 else ((rows % a.stride == 0) & (cols % a.stride == 0))
    rows, cols = rows[keep], cols[keep]
    npix = rows.size
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    lon0, lon1, lat0, lat1 = G.extent()
    lon = lon0 + (cols + 0.5) * (lon1 - lon0) / W
    lat = lat0 + (rows + 0.5) * (lat1 - lat0) / H
    P = 16
    elev = ctx.elevation()
    z = np.where(land, elev, 0.0)
    n_k = land.astype(np.float64).reshape(H // P, P, W // P, P).sum((1, 3))
    s1 = z.reshape(H // P, P, W // P, P).sum((1, 3))
    s2 = (z * z).reshape(H // P, P, W // P, P).sum((1, 3))
    with np.errstate(invalid="ignore"):
        rough = np.sqrt(np.maximum(s2 / np.maximum(n_k, 1) - (s1 / np.maximum(n_k, 1)) ** 2, 0.0))
    rough = np.repeat(np.repeat(rough, P, 0), P, 1)

    np.savez_compressed(out / "pixels.npz", row=rows.astype(np.int16), col=cols.astype(np.int16),
                        lon=lon.astype(np.float32), lat=lat.astype(np.float32),
                        region_id=ctx.region_id[rows, cols].astype(np.int8),
                        elevation=elev[rows, cols].astype(np.float32),
                        roughness=rough[rows, cols].astype(np.float32),
                        region_names=np.array(ctx.region_names),
                        region_display_ids=np.array(ctx.region_ids))
    month = ctx.month_of_day[y][:nd]
    np.savez_compressed(out / "days.npz", day=np.arange(nd, dtype=np.int16),
                        month=month.astype(np.int8), year=np.full(nd, y, np.int16),
                        doy_sin=np.array([C.doy_sincos(t)[0] for t in range(nd)], np.float32),
                        doy_cos=np.array([C.doy_sincos(t)[1] for t in range(nd)], np.float32))

    t0 = time.time()
    TR = np.zeros((nd, npix), np.int16)
    for t in range(nd):
        TR[t] = q(ctx.truth(y, t)[rows, cols], "truth")
    np.savez_compressed(out / "truth.npz", truth=TR)
    print(f"  truth 导出完成 {time.time() - t0:.0f}s", flush=True)

    checks, srcs = {}, {}
    for lab, d in specs:
        t1 = time.time()
        pn = "ens_mean" if (d / "ens_mean").is_dir() else "prediction"
        arrs = {"ens_mean": np.zeros((nd, npix), np.int16), "crps": np.zeros((nd, npix), np.int16)}
        has_sp, has_rk = (d / "spread").is_dir(), (d / "rank").is_dir()
        if has_sp:
            arrs["spread"] = np.zeros((nd, npix), np.int16)
        if has_rk:
            arrs["rank"] = np.zeros((nd, npix), np.int8)
        for t in range(nd):
            arrs["ens_mean"][t] = q(np.load(d / pn / f"{y}_d{t}.npy")[rows, cols], "ens_mean")
            arrs["crps"][t] = q(np.load(d / "crps" / f"{y}_d{t}.npy")[rows, cols], "crps")
            if has_sp:
                arrs["spread"][t] = q(np.load(d / "spread" / f"{y}_d{t}.npy")[rows, cols], "spread")
            if has_rk:
                arrs["rank"][t] = np.load(d / "rank" / f"{y}_d{t}.npy")[rows, cols].astype(np.int8)
        np.savez_compressed(out / f"{slug(lab)}.npz", **arrs)
        pooled = float(dq(arrs["crps"], "crps").mean())
        mp, ref = d / "metrics.json", None
        if mp.exists():
            mj = json.load(open(mp))
            blk = next((v for v in mj.values() if isinstance(v, dict) and "crps" in v), None)
            ref = float(blk["crps"]) if blk else None
        tol = QUANT["crps"][1] * 2
        if ref is not None and a.stride <= 1 and abs(pooled - ref) > tol:
            raise SystemExit(f"{lab}: 导出后重算 CRPS {pooled:.4f} 与 metrics.json {ref} 差超过量化分辨率")
        checks[lab] = {"pooled_crps_from_bundle": round(pooled, 5), "metrics_json_crps": ref,
                       "matched": None if ref is None else bool(a.stride > 1 or abs(pooled - ref) <= tol),
                       "note": "stride>1 时抽样格点与全域不同, 只作参考不作判据"}
        mj2 = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
        srcs[lab] = {"dir": str(d), "members": mj2.get("members"), "kind": mj2.get("kind"),
                     "fields": sorted(arrs)}
        print(f"  {lab} 导出完成 {time.time() - t1:.0f}s  池化 CRPS {pooled:.4f}", flush=True)

    if a.factors:
        import shutil
        shutil.copy(a.factors, out / "factors.npz")
    (out / "load_example.py").write_text(LOADER, encoding="utf-8")

    meta = {"kind": "calibration_bundle", "target": a.target, "year": y, "n_days": nd,
            "n_pixels": int(npix), "stride": a.stride,
            "domain": f"分区文件的 land(daymet_land ∧ era5_valid), 未腐蚀; 全域 {int(land.sum())} 格, "
                      f"本包按 stride {a.stride} 抽出 {npix} 格",
            "arrays": {"truth.npz": "(n_days, n_pixels) int16",
                       "<label>.npz": "ens_mean/crps int16, spread int16(若有), rank int8(若有)",
                       "pixels.npz": "row/col/lon/lat/region_id/elevation/roughness 与区域名表",
                       "days.npz": "day/month/year/doy_sin/doy_cos"},
            "quant": {k: {"offset": v[0], "scale": v[1], "dequant": "v * scale + offset"}
                      for k, v in QUANT.items()},
            "rank": "真值在 32 个成员中的名次 0..32; 平局按固定种子均匀劈分",
            "crps": "集合 CRPS 的公平(无偏)估计式, 逐像素",
            "sources": srcs, "selfcheck": checks,
            "units": "K" if a.target != C.PRECIP else "见 contract; 降水有两种单位空间",
            "caveat": "μ 是确定性模型, 没有 spread 与 rank; 其 crps 恒等于 MAE"}
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    tot = sum(f.stat().st_size for f in out.glob("*.npz"))
    print(f"\n包大小 {tot/1e6:.0f} MB -> {out}")
    for lab, c in checks.items():
        print(f"  自检 {lab:<6} 池化 CRPS {c['pooled_crps_from_bundle']} vs metrics.json {c['metrics_json_crps']}  {c['matched']}")


if __name__ == "__main__":
    main()
