#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_gain_by_ssim_set.py — (区域 × 月) 的 CRPS 技能热图, 按 SSIM 集合分组
============================================================================
每个 (区域, 月, 集合) 单元上给出各模型相对★共同的阶段A μ★的技能

    CRPSS = 1 − ΣCRPS_模型 / ΣCRPS_μ        (池化求和后再相除, 不是逐单元比值再平均)

μ 是确定性模型, CRPS 恒等于 MAE; 四个阶段B 共用同一份 μ 缓存与同一个 σ_r, 所以"相对 μ 提升
多少"对它们是同一把尺子。分组是每个模型★自身★逐像素 SSIM 的两个独立二分: 最差 q 对其余,
最优 q 对其余 —— 与 stage_b_ssim_sets 的 global 排序同一批像素(阈值直接读那次的 summary.json,
不重算)。

数据全部取自 stage_b_ssim_sets 落的逐像素数组缓存, 不读落场。首次用到某模型的 SSIM 场时会把
l·c·s 的结果存成 <key>__ssim.npy 回写缓存, 之后所有工具都不必再算一遍。

★域的口径★: SSIM 只在腐蚀 5px 的陆地上有定义, 所以两个"其余"之和是腐蚀域, 海岸带不在任何组里。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_gain_by_ssim_set \\
      --cache <逐像素数组缓存目录> --thresholds runs/exp/<ssimsets 实验>/summary.json \\
      --model DEC=jmb-dec-jda --model TC=jmb-tc-jda --model EC=jmb-ec-jda --model JDB=jdb-jda \\
      --ref MU=jda1-mu --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag>
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
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation.render.context import RenderContext

BASE = ("worst", "best", "mid")                       # 两个二分的共同细化
PARTS = ("worst", "worst_rest", "best", "best_rest")
UNION = {"worst": ("worst",), "worst_rest": ("best", "mid"),
         "best": ("best",), "best_rest": ("worst", "mid")}
PEN = {"worst": "SSIM-worst", "worst_rest": "complement of SSIM-worst",
       "best": "SSIM-best", "best_rest": "complement of SSIM-best"}


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=key 形式, 得到 {s!r}")
    lab, k = s.split("=", 1)
    return lab.strip(), k.strip()


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def ssim_field(cache, key):
    """<key>__ssim.npy 存在就 mmap, 否则由 l·c·s 算出来并回写缓存(下次免算)。"""
    p = cache / f"{key}__ssim.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    t0 = time.time()
    S = (np.load(cache / f"{key}__l.npy", mmap_mode="r")
         * np.load(cache / f"{key}__c.npy", mmap_mode="r")
         * np.load(cache / f"{key}__s.npy", mmap_mode="r")).astype(np.float32)
    np.save(p, S)
    print(f"  [{key}] SSIM 场算好并回写缓存 {time.time() - t0:.0f}s", flush=True)
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--thresholds", required=True, help="stage_b_ssim_sets 的 summary.json, 提供各模型的全年阈值")
    ap.add_argument("--model", action="append", required=True, help="label=缓存里的 key, 可多次")
    ap.add_argument("--ref", required=True, help="label=缓存里的 key, 作分母的参考模型(阶段A μ)")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.01, help="只用于图题标注; 阈值取自 --thresholds")
    ap.add_argument("--year", type=int, default=2020)
    a = ap.parse_args()

    cache = Path(a.cache)
    ident = json.load(open(cache / "_ident.json"))
    TH = json.load(open(a.thresholds))["sets"]
    specs = [parse_spec(s) for s in a.model]
    ref_lab, ref_key = parse_spec(a.ref)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ctx = RenderContext(Path(ident["dirs"].split("|")[0]), a.regions, C.TARGETS[0], out,
                        years=(a.year,), model_tag="gain", scales_path=out / "scales.json")
    land = ctx.land
    er = MT.eroded_land_mask(land)
    nreg = len(ctx.region_names)
    rid = ctx.region_id[er] - 1                        # 0..nreg-1
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    nd = C.DAYS_PER_YEAR
    month = ctx.month_of_day[a.year][:nd]
    months = sorted(set(month.tolist()))
    mlab = [ctx.MONTHS[m - 1] for m in months]

    ref_crps = np.load(cache / f"{ref_key}__crps.npy", mmap_mode="r")
    if ref_crps.shape != (nd, int(er.sum())):
        raise SystemExit(f"参考模型的形状 {ref_crps.shape} 与 (天数, 腐蚀域) 不符")

    res = {"year": a.year, "frac": a.frac, "cache": str(cache), "ref": ref_lab,
           "metric": "CRPSS = 1 − ΣCRPS_model / ΣCRPS_ref (池化求和后相除)",
           "domain": {"land": int(land.sum()), "eroded": int(er.sum())}, "models": {}}
    mats, raws = {}, {}
    for label, key in specs:
        t0 = time.time()
        S = ssim_field(cache, key)
        crps = np.load(cache / f"{key}__crps.npy", mmap_mode="r")
        lo = TH[f"{key}__ssim_worst_global"]["global_threshold"]
        hi = TH[f"{key}__ssim_best_global"]["global_threshold"]
        num = {b: np.zeros((nd, nreg)) for b in BASE}   # Σ CRPS_模型
        den = {b: np.zeros((nd, nreg)) for b in BASE}   # Σ CRPS_μ
        cnt = {b: np.zeros((nd, nreg)) for b in BASE}   # 像素数
        for d in range(nd):
            sd = np.asarray(S[d]); cm = np.asarray(crps[d], np.float64); cr = np.asarray(ref_crps[d], np.float64)
            grp = np.where(sd <= lo, 0, np.where(sd >= hi, 1, 2))
            idx = rid * 3 + grp
            w = np.bincount(idx, minlength=nreg * 3).reshape(nreg, 3)
            sm = np.bincount(idx, weights=cm, minlength=nreg * 3).reshape(nreg, 3)
            sr = np.bincount(idx, weights=cr, minlength=nreg * 3).reshape(nreg, 3)
            for bi, b in enumerate(BASE):
                cnt[b][d] = w[:, bi]; num[b][d] = sm[:, bi]; den[b][d] = sr[:, bi]
        stats = {}
        for g in PARTS:
            ps = UNION[g]
            N = sum(num[b] for b in ps); D = sum(den[b] for b in ps); Cn = sum(cnt[b] for b in ps)
            skill = np.full((nreg, len(months)), np.nan)
            raw = np.full((nreg, len(months)), np.nan)
            for j, mm in enumerate(months):
                sel = month == mm
                n = N[sel].sum(0); dd = D[sel].sum(0); c = Cn[sel].sum(0)
                ok = (dd > 0) & (c > 0)
                skill[:, j] = np.where(ok, 1.0 - n / np.maximum(dd, 1e-12), np.nan)[order]
                raw[:, j] = np.where(ok, n / np.maximum(c, 1), np.nan)[order]
            mats[(label, g)] = skill; raws[(label, g)] = raw
            stats[g] = {"crpss_overall": float(1 - N.sum() / max(D.sum(), 1e-12)),
                        "crps_overall": float(N.sum() / max(Cn.sum(), 1)),
                        "ref_crps_overall": float(D.sum() / max(Cn.sum(), 1)),
                        "pixel_days": int(Cn.sum()),
                        "cells_negative": int(np.nansum(skill < 0))}
        res["models"][label] = {"key": key, "ssim_low": lo, "ssim_high": hi,
                                "seconds": round(time.time() - t0, 1), "by_group": stats}
        print(f"[{label}] {time.time() - t0:.0f}s  " +
              "  ".join(f"{g} CRPSS={stats[g]['crpss_overall']:.4f}" for g in PARTS), flush=True)

    allv = np.concatenate([m[np.isfinite(m)] for m in mats.values()])
    vmin, vmax = float(np.percentile(allv, 0.5)), float(np.percentile(allv, 99.5))
    if vmin < 0:                                        # 有负技能就用围绕 0 对称的发散色标
        lim = max(abs(vmin), abs(vmax)); vmin, vmax, cmap = -lim, lim, "RdBu_r"
    else:
        cmap = "viridis"
    for (label, g), m in mats.items():
        ttl = f"{label} · {PEN[g]} {a.frac*100:g}%" if g in ("worst", "best") \
            else f"{label} · {PEN[g]} {100 - a.frac*100:g}%"
        ctx.heatmap(m, rlab, mlab, f"crpss_region_month_{slug(label)}_{g}",
                    cbar=f"CRPSS vs {ref_lab}", scale_group="crpss_rm", cmap=cmap,
                    vmin=vmin, vmax=vmax, title=f"{ttl} · CRPSS vs {ref_lab} · region × month")
    ctx.write_scales()
    np.savez_compressed(out / "cells.npz", region_display_ids=np.array(rlab), months=np.array(months),
                        **{f"crpss__{slug(l)}__{g}": v for (l, g), v in mats.items()},
                        **{f"crps__{slug(l)}__{g}": v for (l, g), v in raws.items()})
    res["scale"] = {"vmin": vmin, "vmax": vmax, "cmap": cmap}
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"色标 [{vmin:.3f}, {vmax:.3f}] {cmap};  图 {len(mats)} 张")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
