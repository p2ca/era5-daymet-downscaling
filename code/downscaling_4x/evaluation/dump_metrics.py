#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
dump_metrics.py — 从逐日落场目录算全年汇总标量指标
============================================================================
输入是 det_dump(或生成式采样落盘)落下的场目录(ens_mean/ crps/ ...), 输出一份
与其它方法同口径的标量指标, 供核心指标表直接引用。

口径与 eval_common 完全一致, 关键几条:
  * 确定性指标(RMSE/MAE/bias/corr)与 SSIM 都算在★集合均值 ens_mean★上。
  * RMSE 走逐(天,像素)池化, 不是逐日 RMSE 再平均 —— 两者不等价。
  * SSIM 用 ssim_masked: Wang 2004 高斯窗(sigma=1.5), data_range 取当日 truth 在陆地
    上的 max-min, 海洋填当日陆地均值, 只在腐蚀 5px 的陆地掩膜上平均。
  * CRPS 取逐像素 crps 场在陆地上的池化均值(每天陆地像素数相同, 与逐日平均等价)。
  * 降水额外给一份 log1p(mm) 空间的 RMSE/MAE/bias/corr/SSIM/CRPS: 物理空间的量级
    由少数强降水格点主导, 两个空间下方法排名可以相反, 因此两份都要给。
  * "陆地"一律指有效域(daymet_land AND era5_valid), 与训练 loss 同一掩膜。

用法:
    python -m downscaling_4x.evaluation.dump_metrics \
        --fields runs/exp/<id>/fields2020/2m_temperature_max --target 2m_temperature_max
============================================================================
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from downscaling_4x import contract as C
from downscaling_4x.data import downscale_baseline as DB
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation.metrics import precip_log_mm, ssim_masked, SSIM_ERODE


def _mean_field(paths, land):
    """一组逐日场在陆地上的池化均值(跳过 NaN)。"""
    s = 0.0
    n = 0
    for p in paths:
        a = np.load(p)[land].astype(np.float64)
        g = np.isfinite(a)
        s += float(a[g].sum())
        n += int(g.sum())
    return (s / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser(description="落场目录 -> 全年汇总标量指标")
    ap.add_argument("--fields", required=True, help="落场目录(含 ens_mean/ crps/)")
    ap.add_argument("--target", required=True, choices=C.TARGETS)
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--out", default=None, help="缺省写 <fields>/metrics.json")
    ap.add_argument("--max-days", type=int, default=0, help=">0 时只取前 N 天, 冒烟用")
    II.add_arg(ap)
    a = ap.parse_args()

    fields = Path(a.fields)
    # 落场 meta 记录了产生它的输入目录; 与本次 --era5-dir 不一致说明调用弄混了产品
    fm = fields / "meta.json"
    rec = (json.load(open(fm)).get("input") or {}).get("era5_dir") if fm.exists() else None
    II.check(rec, a.era5_dir, f"落场 {fields}", allow=getattr(a, II.ALLOW_DEST, False))
    ti = C.TARGETS.index(a.target)
    is_precip = (a.target == C.PRECIP)

    # 只读真值与掩膜, 用不到条件张量, 因此取无历史的模式免去多载一年
    stats = Stats(a.era5_dir, a.daymet_dir)
    dd = DownscaleData(a.era5_dir, a.daymet_dir, list(a.years), stats, mode="baseline_21")

    days = [(y, t) for y in a.years for t in range(C.DAYS_PER_YEAR)
            if (fields / "ens_mean" / f"{y}_d{t}.npy").exists()]
    if a.max_days:
        days = days[: a.max_days]
    print(f"[metrics] {fields.name} target={a.target} 天数={len(days)}", flush=True)

    land = dd.mask
    mb_er = binary_erosion(land, iterations=SSIM_ERODE)
    acc = DB.Acc()
    acc_log = DB.Acc()
    ssim_sum = ssim_log_sum = 0.0
    ssim_n = 0
    for k, (y, t) in enumerate(days):
        em = np.load(fields / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
        _, hr = dd.target(y, t)
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale          # 落场的 ens_mean 已是 mm/day

        em_f = np.where(land, em, 0.0)                  # 陆外是 NaN, 填掉再进 SSIM 的填充逻辑
        acc.add(em[land], truth[land])
        ssim_sum += ssim_masked(em_f, truth, land, mb_er)
        ssim_n += 1
        if is_precip:
            lg = lambda x: precip_log_mm(x, stats.precip_clip)      # 与训练目标、统计基线同一条变换
            acc_log.add(lg(em[land]), lg(truth[land]))
            ssim_log_sum += ssim_masked(lg(em_f), lg(truth), land, mb_er)
        if (k + 1) % 50 == 0:
            print(f"[metrics] {k+1}/{len(days)}", flush=True)

    res = {"id": fields.name, "target": a.target, "years": list(a.years),
           "n_days": len(days), "source": "ens_mean 场 + crps 场",
           "input": II.describe(a.era5_dir),
           "ssim_note": f"Wang2004 高斯窗 sigma=1.5, data_range=当日 truth 陆地 max-min, "
                        f"陆地掩膜腐蚀 {SSIM_ERODE}px"}

    unit = "mm/day" if is_precip else "K"
    det = acc.result()
    res[unit] = {**det, "ssim": round(ssim_sum / max(ssim_n, 1), 4)}
    for f in ("crps", "spread"):
        if (fields / f).is_dir():
            v, n = _mean_field([fields / f / f"{y}_d{t}.npy" for y, t in days], land)
            res[unit][f] = round(v, 4)
            res[unit][f"{f}_n"] = n

    if is_precip:
        # 表里降水用 m/day: 与 mm/day 差一个常数因子, 一阶量线性缩放, corr/SSIM 不变
        k = 1.0 / stats.precip_scale
        res["m/day"] = {"rmse": round(det["rmse"] * k, 6), "mae": round(det["mae"] * k, 6),
                        "bias": round(det["bias"] * k, 6), "corr": det["corr"],
                        "ssim": res[unit]["ssim"], "n": det["n"]}
        if "crps" in res[unit]:
            res["m/day"]["crps"] = round(res[unit]["crps"] * k, 6)
        if "spread" in res[unit]:
            res["m/day"]["spread"] = round(res[unit]["spread"] * k, 6)

        dl = acc_log.result()
        res["log1p(mm)"] = {**dl, "ssim": round(ssim_log_sum / max(ssim_n, 1), 4),
                            "transform": f"precip_fwd: <{stats.precip_clip:g} mm 置零后 log1p (预测与真值同)"}
        if (fields / "crps_log").is_dir():
            v, n = _mean_field([fields / "crps_log" / f"{y}_d{t}.npy" for y, t in days], land)
            res["log1p(mm)"]["crps"] = round(v, 4)
            res["log1p(mm)"]["crps_n"] = n

    out = Path(a.out) if a.out else fields / "metrics.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[metrics] -> {out}")


if __name__ == "__main__":
    main()
