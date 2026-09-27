#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
rebuild_det_crps_log.py — 从 ens_mean 与真值重建确定性落场的 crps_log 场
============================================================================
确定性方法只有一个成员, log 空间的 CRPS 恒等于 |log(预测) − log(真值)|, 因此 crps_log
场是 ens_mean 与真值的确定性函数, 不需要重跑网络前向就能重建。log 变换走评测侧唯一的
`metrics.precip_log_mm`(合同的 precip_fwd: <0.1 mm 置零后 log1p), 与指标、统计基线同口径。

只接受单成员的落场目录: 集合落场的 crps_log 是逐成员算的, 不能由均值场恢复, 遇到会拒绝。
重建后把变换口径写进目录的 meta.json, 使产物自述所用口径。

用法:
  python -m downscaling_4x.tools.rebuild_det_crps_log \\
      --fields runs/exp/<id>/fields2020/total_precipitation_24hr [--era5-dir ...]
============================================================================
"""
import argparse
import json
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.metrics import precip_log_mm


def main():
    ap = argparse.ArgumentParser(description="重建确定性落场的 crps_log 场")
    ap.add_argument("--fields", required=True, help="落场目录(含 ens_mean/), 目标须为降水")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    II.add_arg(ap)
    a = ap.parse_args()

    fields = Path(a.fields)
    meta = json.load(open(fields / "meta.json"))
    if meta.get("target") != C.PRECIP:
        raise SystemExit(f"{fields}: 目标是 {meta.get('target')}, crps_log 只对降水有定义")
    if int(meta.get("members", 1)) != 1 or meta.get("kind") != "deterministic_field_dump":
        raise SystemExit(f"{fields}: 不是单成员的确定性落场(members={meta.get('members')}, kind={meta.get('kind')}), "
                         "集合的 crps_log 不能由均值场重建")
    II.check((meta.get("input") or {}).get("era5_dir"), a.era5_dir, f"落场 {fields}",
             allow=getattr(a, II.ALLOW_DEST, False))

    stats = Stats(a.era5_dir, a.daymet_dir)
    dd = DownscaleData(a.era5_dir, a.daymet_dir, list(a.years), stats, mode="baseline_21")
    land = dd.mask
    ti = C.TARGETS.index(C.PRECIP)
    lg = lambda x: precip_log_mm(x, stats.precip_clip)
    (fields / "crps_log").mkdir(exist_ok=True)
    n = 0
    for y in a.years:
        for t in range(C.DAYS_PER_YEAR):
            src = fields / "ens_mean" / f"{y}_d{t}.npy"
            if not src.exists():
                continue
            p = np.load(src)                                   # mm/day, 有效域外 NaN
            truth = dd.target(y, t)[1][ti] * stats.precip_scale   # m/day -> mm/day
            e = np.abs(lg(np.where(land, p, 0.0)) - lg(truth))
            np.save(fields / "crps_log" / f"{y}_d{t}.npy", np.where(land, e, np.nan).astype(np.float32))
            n += 1
    meta["crps_log_transform"] = f"precip_fwd: <{stats.precip_clip:g} mm 置零后 log1p (预测与真值同); 由 ens_mean 与真值重建"
    meta.setdefault("written", {})["crps_log"] = n
    json.dump(meta, open(fields / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(f"[crps_log] 重建 {n} 天 -> {fields / 'crps_log'}")


if __name__ == "__main__":
    main()
