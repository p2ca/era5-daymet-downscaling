#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_spectrum.py — 落场目录的径向功率谱(预测 vs 真值)
============================================================================
从确定性/生成式落场目录读逐日 ens_mean, 与真值一同在最大的陆地方窗上算径向功率谱,
逐日平均后画成一张 log-log 单图。回答两个问题:

  * 预测在各空间尺度上的能量是否跟得上真值(高波数掉多少 = 细节钝化多少);
  * 有无孤立的谱峰(切块推理的网格伪影会在 box/patch 的整数倍波数处堆出尖峰,
    地图上未必显眼, 谱上一眼可见)。

降水在 log1p(mm) 空间上算(物理空间被少数强降水格点主导), 温度用物理单位。

用法:
    python -m downscaling_4x.tools.plotting.plot_spectrum \
        --fields runs/exp/<id>/fields2020/2m_temperature_max --out <png>
============================================================================
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.evaluation.metrics import pick_land_box, radial_psd, precip_log_mm


def main():
    ap = argparse.ArgumentParser(description="落场目录 -> 径向功率谱(预测 vs 真值)")
    ap.add_argument("--fields", required=True, help="目标级落场目录(含 ens_mean/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--box", type=int, default=256, help="陆地方窗边长(像素)")
    ap.add_argument("--max-days", type=int, default=0, help=">0 时只取前 N 天, 冒烟用")
    a = ap.parse_args()

    fields = Path(a.fields)
    meta = json.load(open(fields / "meta.json"))
    target = meta.get("target") or meta["targets"][0]
    ti = C.TARGETS.index(target)
    is_precip = (target == C.PRECIP)
    tag = ((meta.get("model_sources") or [{}])[0] or {}).get("path", "")
    tag = Path(tag).parent.name if str(tag).endswith(".pt") else (meta.get("method") or "model")

    stats = Stats()
    dd = DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, list(a.years), stats, mode="baseline_21")
    by, bx, bs = pick_land_box(dd.mask, a.box)
    days = [(y, t) for y in a.years for t in range(C.DAYS_PER_YEAR)
            if (fields / "ens_mean" / f"{y}_d{t}.npy").exists()]
    if a.max_days:
        days = days[: a.max_days]
    lg = precip_log_mm                                     # 与指标同一条 log 变换

    psd = None
    for y, t in days:
        em = np.load(fields / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
        _, hr = dd.target(y, t)
        tr = hr[ti].astype(np.float64)
        if is_precip:
            tr = tr * stats.precip_scale               # 与落场同为 mm/day
        em = np.where(dd.mask, em, 0.0)
        tr = np.where(dd.mask, tr, 0.0)
        if is_precip:
            em, tr = lg(em), lg(tr)
        cur = np.stack([radial_psd(f[by:by + bs, bx:bx + bs], bs) for f in (tr, em)], 0)
        psd = cur if psd is None else psd + cur
    psd /= max(len(days), 1)

    space = "log1p(mm) space" if is_precip else "physical (K)"
    fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
    for i, (lab, c, ls) in enumerate([("truth", "k", "-"), ("prediction", "#4878a8", "-")]):
        cur = psd[i][1:]
        ax.loglog(np.arange(1, len(cur) + 1), cur, color=c, ls=ls, lw=1.6, label=lab)
    ax.set_xlabel("radial wavenumber (cycles / box)")
    ax.set_ylabel("power")
    ax.set_title(f"radial power spectrum  {target}  ({space}, {len(days)}-day mean)\n{tag}",
                 fontsize=10)
    ax.legend(frameon=False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    # 高波数段的能量比: <1 = 细节钝化; 谱峰另看曲线本身
    hi = slice(len(psd[0]) // 2, None)
    ratio = float(psd[1][hi].sum() / max(psd[0][hi].sum(), 1e-30))
    print(f"[spectrum] {len(days)} 天, box={bs}px @({by},{bx}), "
          f"高波数(后半段)能量比 pred/truth = {ratio:.3f} -> {out}")


if __name__ == "__main__":
    main()
