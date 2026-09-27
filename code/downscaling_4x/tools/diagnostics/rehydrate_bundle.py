#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
rehydrate_bundle.py — 把传输用的量化包还原成标准落场目录
============================================================================
export_calibration_bundle 打出的包是 (天, 有效格点) 的整型数组, 体积只有标准落场的
约四分之一; 本工具按包里记的 row/col 把它摊回 (H, W) 的逐日 .npy, 域外填 NaN,
于是 dump_metrics / render / plot_cross_set_gain / scale_band_multi / region_daily_profile
等全部既有工具都能原样跑, 不必为传输格式改任何一个。

还原后自检: 重算池化 CRPS 与包里 meta.json 记的 metrics_json_crps 对拍, 并写一份
metrics.json 与 meta.json 到落场目录, 使还原出来的目录与原生落场在工具眼里等价。

★只还原包里有的场★: 没有成员, 因此成员级的分析仍需原始 members/。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.rehydrate_bundle \\
      --bundle <收到的包目录> --model DEC --out runs/exp/<还原成的落场目录>
============================================================================
"""
import argparse
import json
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True, help="包内的模型标签, 即 <label>.npz 的 label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", type=int, nargs=2, default=[480, 960])
    a = ap.parse_args()

    B = Path(a.bundle)
    meta = json.load(open(B / "meta.json"))
    if meta.get("stride", 1) != 1:
        raise SystemExit(f"包的 stride 是 {meta['stride']}, 抽样过的包还原不出完整场; 还原需要 stride 1")
    px = np.load(B / "pixels.npz")
    rows, cols = px["row"].astype(np.int64), px["col"].astype(np.int64)
    H, W = a.shape
    z = np.load(B / f"{a.model}.npz")
    y, nd = meta["year"], meta["n_days"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for name in z.files:
        (out / name).mkdir(exist_ok=True)
        qz = meta["quant"].get(name)
        for t in range(nd):
            if name == "rank":
                f = np.full((H, W), -1, np.int8)
                f[rows, cols] = z[name][t]
            else:
                f = np.full((H, W), np.nan, np.float32)
                f[rows, cols] = z[name][t].astype(np.float32) * np.float32(qz["scale"]) + np.float32(qz["offset"])
            np.save(out / name / f"{y}_d{t}.npy", f)
        print(f"  {name} 还原 {nd} 天", flush=True)

    src = meta["sources"][a.model]
    chk = meta["selfcheck"][a.model]
    qz = meta["quant"]["crps"]
    pooled = float((z["crps"].astype(np.float64) * qz["scale"] + qz["offset"]).mean())
    ref = chk.get("metrics_json_crps")
    if ref is not None and abs(pooled - ref) > qz["scale"] * 2:
        raise SystemExit(f"还原后池化 CRPS {pooled:.5f} 与包里记的 {ref} 不符")
    unit = meta.get("units", "K")
    json.dump({unit: {"crps": round(pooled, 4), "crps_n": int(z["crps"].size),
                      "n": int(z["crps"].size)},
               "id": out.name, "target": meta["target"], "years": [y], "n_days": nd,
               "source": "rehydrated from calibration bundle"},
              open(out / "metrics.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"id": out.name, "target": meta["target"], "unit": unit, "years": [y],
               "n_days_expected": nd, "all_days": nd == C.DAYS_PER_YEAR,
               "members": src.get("members"), "kind": src.get("kind"),
               "status": "done", "rehydrated_from": str(B), "original_dir": src.get("dir"),
               "note": "由传输包还原; 只含包内的场, 没有成员与路由"},
              open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n还原完成, 池化 CRPS {pooled:.5f} (包里记 {ref}) -> {out}")


if __name__ == "__main__":
    main()
