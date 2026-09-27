#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_worst_cells.py — 参考模型最差的一批 (区域, 日) 单元与其余单元上的分方法得分
============================================================================
在 (区域, 日) 单元的配对比较之上再走一步: 先按一条规则挑出参考模型"最差"的一批单元,
再在★最差集★与★其余集★上分别给出全部参与模型的 MAE / RMSE / bias / corr / CRPS / SSIM。
用途是把一个总量上很小的平均差距拆开 —— 增益是普遍的, 还是"在一小撮单元上大输、在其余
单元上小赢"换来的。

挑选规则(--rule, 可多给):
  rel_gap   参考相对最优对照的相对劣势 (ref - min(others)) / min(others) 最大的 q 比例
  loss_all  参考同时劣于全部对照的单元; 集合大小由数据决定, 不受 q 控制
  abs_ref   参考自身分数最高的 q 比例, 即"最难"而非"最劣"的单元

口径:
  * 读场只做一件事: 把每个 (模型, 日, 区域) 的★充分统计量★(像素数与各阶和)落盘。之后
    任何单元子集的指标都由这些和池化算出, 换一条规则或换一个比例是秒级的, 不必重读场。
  * RMSE / MAE / bias / corr 走逐(天,像素)池化, 与 eval_common 的累加器同一套式子。
    逐日 RMSE 再平均与池化不等价, 子集上同样不等价, 因此一律池化。
  * CRPS 取落场的逐像素 crps 场(集合的公平估计式)在子集内的池化均值。
  * SSIM 取 metrics.ssim_field 的 SSIM 图, 在 (子集单元 ∩ 腐蚀 5px 陆地) 上池化平均;
    data_range 仍是当日★全域★ truth 的陆地 max-min, 与子集无关 —— 同一天里所有模型、所有
    子集共用同一把尺子, 子集之间才可比。
  * ★corr 不可跨子集比较★: 子集限制了真值自身的动态范围, 相关系数会被机械地压低。
    同一子集内跨模型比较有效, 最差集与其余集之间比较无效。
  * 自检: 全集上池化出的六个量必须逐项复现各落场目录 metrics.json 里的官方标量, 不一致
    即当场退出。子集口径一旦与官方口径分叉, 数字照样算得出来, 但已经不是同一个量。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.stage_b_worst_cells \\
      --ref jmb-dec-jda=runs/exp/<dec eval2020> \\
      --cmp jmb-tc-jda=runs/exp/<tc eval2020> --cmp jdb-jda=runs/exp/<jdb eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation.render.context import RenderContext

# 单元的充分统计量: 池化任何子集的六个指标只需要这些和
STAT_KEYS = ("n", "ae", "se", "d", "crps", "p", "t", "pp", "tt", "pt", "ssim", "ssim_n")


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def bin_sum(vals, ids, nreg):
    """按区域编号(1..nreg)对一维取值求和 -> 长 nreg 的向量。"""
    return np.bincount(ids, weights=vals, minlength=nreg + 1)[1:]


def accumulate(ctx, dirs, labels, y, nd, land, land_er, nreg):
    """一次遍历落场, 得到每个模型的 (nd, nreg) 充分统计量。"""
    rid = ctx.region_id[land]
    rid_er = ctx.region_id[land_er]
    if rid.min() < 1 or rid.max() > nreg:
        raise SystemExit("region_id 超出 1..nreg, 与 regions_v1.npz 约定不符")
    n_px = bin_sum(np.ones(rid.size), rid, nreg)
    n_er = bin_sum(np.ones(rid_er.size), rid_er, nreg)
    if (n_px <= 0).any():
        raise SystemExit("有区域在有效域内没有格点")

    S = {lab: {k: np.zeros((nd, nreg)) for k in STAT_KEYS} for lab in labels}
    t0 = time.time()
    for t in range(nd):
        tr_full = ctx.truth(y, t)
        tr = tr_full[land]
        if not np.isfinite(tr).all():
            raise SystemExit(f"第 {t} 天真值在有效域内含非有限值")
        for lab in labels:
            d0 = dirs[lab]
            em_full = np.load(d0 / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
            cr = np.load(d0 / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
            em = em_full[land]
            if not (np.isfinite(em).all() and np.isfinite(cr).all()):
                raise SystemExit(f"{lab} 第 {t} 天有效域内含非有限值")
            dd = em - tr
            s = S[lab]
            s["n"][t] = n_px
            s["ae"][t] = bin_sum(np.abs(dd), rid, nreg)
            s["se"][t] = bin_sum(dd * dd, rid, nreg)
            s["d"][t] = bin_sum(dd, rid, nreg)
            s["crps"][t] = bin_sum(cr, rid, nreg)
            s["p"][t] = bin_sum(em, rid, nreg)
            s["t"][t] = bin_sum(tr, rid, nreg)
            s["pp"][t] = bin_sum(em * em, rid, nreg)
            s["tt"][t] = bin_sum(tr * tr, rid, nreg)
            s["pt"][t] = bin_sum(em * tr, rid, nreg)
            sm = MT.ssim_field(em_full, tr_full, land)
            if sm is not None:                      # 退化日(当日陆地是常数场)不进 SSIM 平均
                s["ssim"][t] = bin_sum(sm[land_er], rid_er, nreg)
                s["ssim_n"][t] = n_er
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)
    return S


def pool(stat, sel):
    """充分统计量在单元子集 sel 上的池化指标; 与 eval_common 的累加器同一套式子。"""
    g = lambda k: float(stat[k][sel].sum())
    n = g("n")
    if n <= 0:
        return None
    mp, mt = g("p") / n, g("t") / n
    cov = g("pt") / n - mp * mt
    vp = g("pp") / n - mp * mp
    vt = g("tt") / n - mt * mt
    sn = g("ssim_n")
    return {"rmse": (g("se") / n) ** 0.5,
            "mae": g("ae") / n,
            "bias": g("d") / n,
            "corr": cov / ((vp * vt) ** 0.5) if vp > 0 and vt > 0 else float("nan"),
            "crps": g("crps") / n,
            "ssim": g("ssim") / sn if sn > 0 else float("nan"),
            "n_px": int(n),
            "n_cells": int(sel.sum())}


def selections(ref_cell, others_cell, rule, fracs):
    """(名字 -> 布尔单元掩膜) 的最差集候选。"""
    best = others_cell.min(0)
    worst = others_cell.max(0)
    out = {}
    if rule == "loss_all":
        out["loss_all"] = ref_cell > worst
        return out
    key = (ref_cell - best) / best if rule == "rel_gap" else ref_cell
    flat = key.ravel()
    for q in fracs:
        k = max(1, int(round(q * flat.size)))
        thr = np.partition(flat, flat.size - k)[flat.size - k]
        sel = key >= thr                              # 并列值一并入选, 实际比例可能略大于 q
        out[f"{rule}_q{q:g}"] = sel
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="label=eval_dir, 参考模型")
    ap.add_argument("--cmp", action="append", required=True, help="label=eval_dir, 对照模型, 可多次")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-metric", default="crps", choices=("crps", "mae"),
                    help="按哪个指标定义'最差'")
    ap.add_argument("--rule", action="append", default=None,
                    choices=("rel_gap", "loss_all", "abs_ref"), help="挑选规则, 可多给")
    ap.add_argument("--frac", type=float, nargs="+", default=[0.05, 0.1, 0.25],
                    help="rel_gap / abs_ref 的最差集比例")
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--selfcheck-tol", type=float, default=3e-4,
                    help="全集复现 metrics.json 的容差(官方标量保留 4 位小数)")
    ap.add_argument("--stats", default=None, help="复用已落盘的 cell_stats.npz, 跳过读场")
    a = ap.parse_args()

    lr, dr = parse_spec(a.ref)
    cmps = [parse_spec(s) for s in a.cmp]
    labels = [lr] + [l for l, _ in cmps]
    dirs = {lr: dr, **{l: d for l, d in cmps}}
    rules = a.rule or ["rel_gap", "loss_all", "abs_ref"]
    y = a.year
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(dr, a.regions, a.target, out, years=(y,),
                        model_tag=f"{slug(lr)}-worstset")
    land = ctx.land
    land_er = MT.eroded_land_mask(land)
    nreg = len(ctx.region_names)
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    month = ctx.month_of_day[y][:nd]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]

    if a.stats:
        z = np.load(a.stats)
        S = {lab: {k: z[f"{slug(lab)}__{k}"] for k in STAT_KEYS} for lab in labels}
        print(f"[worst] 复用 {a.stats}")
    else:
        S = accumulate(ctx, dirs, labels, y, nd, land, land_er, nreg)
        np.savez(out / "cell_stats.npz", month=month,
                 region_display_ids=np.array(ctx.region_ids), region_names=np.array(ctx.region_names),
                 **{f"{slug(lab)}__{k}": S[lab][k] for lab in labels for k in STAT_KEYS})

    # ---- 自检: 全集池化必须复现各落场 metrics.json 的官方标量 ----
    full = np.ones((nd, nreg), bool)
    checks = {}
    for lab in labels:
        got = pool(S[lab], full)
        mp = dirs[lab] / "metrics.json"
        if not mp.exists() or nd != C.DAYS_PER_YEAR:
            checks[lab] = "skipped"
            continue
        off = json.load(open(mp))[ctx.unit]
        bad = {k: (round(got[k], 4), off[k]) for k in ("rmse", "mae", "bias", "corr", "crps", "ssim")
               if k in off and abs(got[k] - off[k]) > a.selfcheck_tol}
        if bad:
            raise SystemExit(f"全集自检失败 {lab}: 池化值与 metrics.json 不一致 {bad}")
        checks[lab] = "ok"
    print(f"[worst] 全集自检 {checks}", flush=True)

    # ---- 单元分数与最差集 ----
    cell = {lab: S[lab][a.rank_metric] / S[lab]["n"] for lab in labels}
    ref_cell = cell[lr]
    others_cell = np.stack([cell[l] for l, _ in cmps], 0)

    res = {"ref": lr, "cmp": [l for l, _ in cmps], "target": a.target, "unit": ctx.unit,
           "year": y, "n_days": nd, "n_regions": nreg, "rank_metric": a.rank_metric,
           "n_cells": int(ref_cell.size), "selfcheck_vs_metrics_json": checks,
           "regions": {"display_ids": rlab, "names": rname},
           "caveats": ["corr 与 SSIM 在窄子集上会被真值动态范围压低, 只可在同一子集内跨模型比较",
                       "RMSE/MAE/bias/corr/CRPS 为逐(天,像素)池化, 非逐日平均",
                       "SSIM 的 data_range 取当日全域 truth 陆地 max-min, 与子集无关"],
           "sets": {}}

    for rule in rules:
        for name, sel in selections(ref_cell, others_cell, rule, a.frac).items():
            rest = ~sel
            blk = {"rule": rule, "share_cells": float(sel.mean()),
                   "n_cells_worst": int(sel.sum()), "n_cells_rest": int(rest.sum()),
                   "worst": {lab: pool(S[lab], sel) for lab in labels},
                   "rest": {lab: pool(S[lab], rest) for lab in labels},
                   "composition": {
                       "by_region_share": {rlab[j]: float(sel[:, order[j]].mean()) for j in range(nreg)},
                       "by_month_share": {int(m): float(sel[month == m].mean()) for m in sorted(set(month.tolist()))},
                       "worst_cells_by_region": {rlab[j]: int(sel[:, order[j]].sum()) for j in range(nreg)},
                       "worst_cells_by_month": {int(m): int(sel[month == m].sum()) for m in sorted(set(month.tolist()))}}}
            res["sets"][name] = blk

    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)

    # ---- 控制台表 ----
    mk = ("mae", "rmse", "bias", "corr", "crps", "ssim")
    for name, blk in res["sets"].items():
        print(f"\n=== {name}  最差集 {blk['n_cells_worst']} 单元 ({blk['share_cells']*100:.1f}%) "
              f"| 其余 {blk['n_cells_rest']} 单元  [{a.rank_metric} 排序, 单位 {ctx.unit}] ===")
        for part in ("worst", "rest"):
            print(f"  -- {part} --  " + "".join(f"{k:>9}" for k in mk))
            for lab in labels:
                v = blk[part][lab]
                print(f"  {lab:<14}" + "".join(f"{v[k]:9.4f}" for k in mk))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
