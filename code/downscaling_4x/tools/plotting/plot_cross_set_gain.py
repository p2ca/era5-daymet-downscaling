#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_cross_set_gain.py — (区域 × 月) 的 CRPS 相对差热图
============================================================================
一个参考模型对若干对照模型, 每个 (区域, 月) 单元给出

    百分比 = (对照 − 参考) / 对照 × 100     正 = 参考更好

两种口径二选一, 域与集合都不同:

--fields  ★完整评测集★
    两边是同一批像素与同一批日期: 有效域(分区文件的 land, 不腐蚀)× 全年。逐日 CRPS 场
    按 (区域, 月) 先求和再相除(池化, 不是逐日比值再平均)。两侧完全对齐, 因此没有"去掉了
    谁的哪一段"的问题, 也就不需要方向与中点 —— 每个对照只有一张图。

--cells   ★交叉集合★(SSIM 子集, 域是腐蚀 5px 的陆地, 海岸带不在任何一组)
    两个模型各自站在★自己的★ SSIM 集合上比较, 且两边取的集合是交叉的:

      方向 1  参考模型的「去掉自己最优 q」 对 对照模型的「去掉自己最差 q」   (参考被压)
      方向 2  参考模型的「去掉自己最差 q」 对 对照模型的「去掉自己最优 q」   (参考被抬)

    每个 (区域, 月) 单元里, 两边各按★自己那一侧的像素与日期★取 CRPS 均值(归一化) 再相比。
    ★这两个方向都不是等价比较, 而且偏得很有规律★: 去掉自己最差 q 会让 CRPS 白降约 1.9%,
    去掉自己最优 q 只让它自罚约 0.35%(q=1% 时), 所以方向 1 系统性压参考、方向 2 系统性抬
    参考, 量级都是约 2.2-2.3 个百分点。两者把真实差距夹在中间, 因此另出一张★中点图★, 把
    这个由"去掉了谁的哪一段"造成的偏移抵消掉 —— 中点才是可以拿来读的估计, 两个方向各自
    都不是。输入直接取 plot_gain_by_ssim_set 落的 cells.npz(内含逐 (区域,月) 的原始 CRPS)。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_cross_set_gain \\
      --fields DEC=runs/exp/<dec eval2020> --fields TC=runs/exp/<tc eval2020> \\
      --fields EC=runs/exp/<ec eval2020> --fields JDB=runs/exp/<jdb eval2020> \\
      --regions runs/exp/<regions>/regions_v1.npz \\
      --ref DEC --cmp TC --cmp EC --cmp JDB --out runs/exp/<diag>

  python -m downscaling_4x.tools.plotting.plot_cross_set_gain \\
      --cells runs/exp/<crpss 实验>/cells.npz --regions runs/exp/<regions>/regions_v1.npz \\
      --ref DEC --cmp TC --cmp EC --cmp JDB --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation.render.context import RenderContext


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def parse_labeled_dir(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def crosscheck_metrics(lab, d, meta, pooled, pixel_days, tol=1e-3):
    """全集池化必须复现落场自己的 metrics.json, 否则说明域或日期集合取错了。

    池化是这张图的全部算术基础: 取错域(例如误用腐蚀掩膜)照样出得来一张完整的热图, 只是
    每个数都偏了。落场的标量 crps 是独立算出来的同一个量, 拿它对一次就把这条路封死。
    """
    p = Path(d) / "metrics.json"
    if not p.exists():
        return "跳过: 落场没有 metrics.json"
    unit = meta.get("unit")
    blk = json.load(open(p)).get(unit)
    if not isinstance(blk, dict) or "crps" not in blk:
        return f"跳过: metrics.json 里没有 unit={unit!r} 的 crps"
    n = int(blk.get("crps_n", pixel_days))
    if n != pixel_days:
        raise SystemExit(f"{lab}: 池化的像素-日 {pixel_days:,} 与 metrics.json 的 {n:,} 不符, "
                         "两者不是同一个域")
    if abs(pooled - float(blk["crps"])) > tol:
        raise SystemExit(f"{lab}: 池化 CRPS {pooled:.4f} 与 metrics.json 的 {blk['crps']} 不符")
    return f"ok: 池化 {pooled:.4f} ≈ metrics.json {blk['crps']} ({pixel_days:,} 像素-日)"


def pooled_region_month(a, labels_needed, out):
    """逐 (区域, 月) 的池化 CRPS(完整评测集) -> (每模型矩阵, 行标签, 列标签, 绘图 ctx, 记录)。

    池化 = 先把该单元所有像素-日的 CRPS 求和、再除以像素-日数; 与 plot_gain_by_ssim_set
    的 CRPSS 同一种池化方向。像素-日计数矩阵 K 必须逐元素相同 —— 它同时钉住有效域与落盘
    日期集合, 一旦某个模型少了一天或域内出现非有限值, K 就会不同, 而两边照样能相除出一张
    看不出问题的图。
    """
    specs = [parse_labeled_dir(s) for s in a.fields]
    have = [l for l, _ in specs]
    if len(set(have)) != len(have):
        raise SystemExit(f"--fields 的标签有重复: {have}")
    for need in labels_needed:
        if need not in have:
            raise SystemExit(f"--ref/--cmp 里的 {need!r} 没有对应的 --fields {need}=<落场目录>")

    target, members, K0, plot_ctx = None, {}, None, None
    CR, mass, rows, checks = {}, {}, [], {}
    for lab, d in specs:
        if not (d / "crps").is_dir():
            raise SystemExit(f"{lab}: {d} 下没有 crps/ 逐日场目录")
        meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
        tg = meta.get("target")
        if tg is None:
            raise SystemExit(f"{lab}: {d}/meta.json 没有 target 字段, 无法确认是同一个目标")
        if target is None:
            target = tg
        elif tg != target:
            raise SystemExit(f"{lab} 的 target 是 {tg}, 与 {target} 不一致, 拒绝混比")
        members[lab] = int(meta.get("members", 1) or 1)

        ctx = RenderContext(d, a.regions, target, out, years=(a.year,),
                            model_tag=f"full-{slug(a.ref)}", scales_path=out / "scales.json")
        nday = len(ctx.available_days("crps"))
        if not a.partial_year and (meta.get("all_days") is False or nday != C.DAYS_PER_YEAR):
            raise SystemExit(f"{lab}: crps 只落了 {nday}/{C.DAYS_PER_YEAR} 天, 完整评测集口径要求整年; "
                             "各臂共用同一批日期的干预对照可给 --partial-year")
        M, K = ctx.agg.mass_matrix("crps", "region")
        M, K = M[1:, 1:], K[1:, 1:]                       # 0 行/列弃用
        if K0 is None:
            K0 = K
        elif not np.array_equal(K, K0):
            bad = np.argwhere(K != K0)[:3].tolist()
            raise SystemExit(f"{lab} 的像素-日计数与 {have[0]} 不同(单元 {bad} ...), "
                             "说明两者的有效域或落盘日期不是同一批, 拒绝相比")
        if (K <= 0).any():
            raise SystemExit(f"{lab}: 有 (区域, 月) 单元一个有效像素-日都没有")
        checks[lab] = crosscheck_metrics(lab, d, meta, float(M.sum() / K.sum()), int(K.sum()))
        order = np.argsort([int(i) for i in ctx.region_ids])
        CR[lab] = (M / K)[order]
        mass[lab] = M[order]
        plot_ctx = ctx
        rows.append({"label": lab, "dir": str(d), "members": members[lab],
                     "mode": meta.get("mode"), "days": nday})

    rlab = [plot_ctx.region_ids[i] for i in order]
    rname = [plot_ctx.region_names[i] for i in order]
    mlab = list(plot_ctx.MONTHS)
    info = {"domain": {"valid_cells": int(plot_ctx.land.sum()),
                       "pixel_days": int(K0.sum()), "days": nday,
                       "full_year": bool(nday == C.DAYS_PER_YEAR),
                       "note": "分区文件的 land(未腐蚀); SSIM 子集口径的域是腐蚀 5px 的陆地"},
            "target": target, "sources": rows,
            "metrics_crosscheck": checks,
            "region_names": rname}
    if len(set(members.values())) > 1:
        info["members_note"] = (f"成员数不一致 {members}; 落场的 crps 是公平(无偏)估计式, "
                                "对成员数不敏感, 但仍需知道这件事")
    return CR, mass, rlab, mlab, plot_ctx, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", help="交叉集合模式: plot_gain_by_ssim_set 落的 cells.npz")
    ap.add_argument("--fields", action="append",
                    help="完整评测集模式: label=落场目录, 可多次; 与 --cells 互斥")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--ref", required=True, help="参考模型标签(图上的'参考')")
    ap.add_argument("--cmp", action="append", required=True, help="对照模型标签, 可多次")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.01, help="交叉集合模式的图题标注")
    ap.add_argument("--vlim", type=float, default=0.0,
                    help="色标上下限 ±vlim(%%); 0 = 按 99 百分位自动。要与另一批图共尺子时显式给")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--partial-year", action="store_true",
                    help="允许各模型只落了部分日期; 跨模型的像素-日计数仍须逐元素相同")
    a = ap.parse_args()

    if bool(a.cells) == bool(a.fields):
        raise SystemExit("--fields(完整评测集) 与 --cells(交叉集合) 必须二选一")
    if a.ref in a.cmp:
        raise SystemExit("--ref 不能同时出现在 --cmp 里")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    mats, rows, extra = {}, [], {}
    if a.fields:
        CR, mass, rlab, mlab, ctx, extra = pooled_region_month(a, [a.ref] + a.cmp, out)
        for cm in a.cmp:
            d = (CR[cm] - CR[a.ref]) / CR[cm] * 100
            mats[(cm, "fullset")] = d
            tot_r, tot_c = mass[a.ref].sum(), mass[cm].sum()
            rows.append({"cmp": cm,
                         "pooled_overall": float((tot_c - tot_r) / tot_c * 100),
                         "cell_mean": float(np.nanmean(d)), "cell_median": float(np.nanmedian(d)),
                         "cells_positive": int(np.nansum(d > 0)),
                         "cells_total": int(np.isfinite(d).sum()),
                         "crps_ref": float(tot_r / extra["domain"]["pixel_days"]),
                         "crps_cmp": float(tot_c / extra["domain"]["pixel_days"])})
        extra["by_region"] = {lab: {"pooled_overall": [
            float((mass[lab][i].sum() - mass[a.ref][i].sum()) / mass[lab][i].sum() * 100)
            for i in range(len(rlab))]} for lab in a.cmp}
        extra["by_month"] = {lab: {"pooled_overall": [
            float((mass[lab][:, j].sum() - mass[a.ref][:, j].sum()) / mass[lab][:, j].sum() * 100)
            for j in range(len(mlab))]} for lab in a.cmp}
    else:
        z = np.load(a.cells)
        ctx = RenderContext(Path(a.cells).parent, a.regions, C.TARGETS[0], out, years=(a.year,),
                            model_tag=f"cross-{slug(a.ref)}", scales_path=out / "scales.json")
        rlab = [str(x) for x in z["region_display_ids"]]
        mlab = [ctx.MONTHS[m - 1] for m in z["months"]]

        def cell(model, part):
            k = f"crps__{slug(model)}__{part}"
            if k not in z.files:
                raise SystemExit(f"cells.npz 里没有 {k}")
            return z[k]

        for cm in a.cmp:
            d1 = (cell(cm, "worst_rest") - cell(a.ref, "best_rest")) / cell(cm, "worst_rest") * 100
            d2 = (cell(cm, "best_rest") - cell(a.ref, "worst_rest")) / cell(cm, "best_rest") * 100
            mats[(cm, "dir1_ref_penalised")] = d1
            mats[(cm, "dir2_ref_favoured")] = d2
            mats[(cm, "midpoint")] = 0.5 * (d1 + d2)
            rows.append({"cmp": cm,
                         "dir1_mean": float(np.nanmean(d1)), "dir1_median": float(np.nanmedian(d1)),
                         "dir2_mean": float(np.nanmean(d2)), "dir2_median": float(np.nanmedian(d2)),
                         "mid_mean": float(np.nanmean(0.5 * (d1 + d2))),
                         "mid_median": float(np.nanmedian(0.5 * (d1 + d2))),
                         "mid_cells_positive": int(np.nansum(0.5 * (d1 + d2) > 0)),
                         "cells_total": int(np.isfinite(d1).sum())})

    allv = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()])
    lim = float(a.vlim) if a.vlim > 0 else float(np.percentile(np.abs(allv), 99))
    q = a.frac * 100
    nd = extra.get("domain", {}).get("pixel_days")
    TTL = {"dir1_ref_penalised": f"{a.ref} minus its best {q:g}%  vs  CMP minus its worst {q:g}%  (ref penalised)",
           "dir2_ref_favoured": f"{a.ref} minus its worst {q:g}%  vs  CMP minus its best {q:g}%  (ref favoured)",
           "midpoint": "midpoint of the two directions (removal bias cancelled)",
           "fullset": f"CRPS · full evaluation set · same {nd:,} pixel-days on both sides" if nd
                      else "CRPS · full evaluation set"}
    group = "fullset_gain" if a.fields else "cross_gain"
    for (cm, kind), m in mats.items():
        ctx.heatmap(m, rlab, mlab, f"crps_gain_{slug(a.ref)}_vs_{slug(cm)}_{kind}",
                    cbar=f"({cm} − {a.ref}) / {cm}  [%]", scale_group=group,
                    cmap="RdBu_r", vmin=-lim, vmax=lim,
                    title=f"{a.ref} vs {cm} · {TTL[kind].replace('CMP', cm)}")
    ctx.write_scales()
    stem = "full_cells" if a.fields else "cross_cells"
    save = {f"{slug(c)}__{k}": v for (c, k), v in mats.items()}
    if a.fields:
        save.update({f"crps__{slug(l)}": v for l, v in CR.items()})
    np.savez_compressed(out / f"{stem}.npz", region_display_ids=np.array(rlab),
                        months=np.arange(1, len(mlab) + 1), **save)
    res = {"mode": "fullset" if a.fields else "cross_set", "ref": a.ref, "cmp": a.cmp,
           "unit": "percent", "sign": "正 = 参考更好", "year": a.year,
           "scale": {"vmin": -lim, "vmax": lim, "cmap": "RdBu_r"}, "by_cmp": rows, **extra}
    if not a.fields:
        res["frac"] = a.frac
        res["note"] = ("dir1 系统性压参考、dir2 系统性抬参考(去掉自己最差 q 白赚约 1.9%, "
                       "去掉自己最优 q 自罚约 0.35%); midpoint 抵消该偏移")
    else:
        res["note"] = "两侧同像素同日期, 无子集划分; 域是未腐蚀的有效域"
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    if a.fields:
        for lab, msg in extra["metrics_crosscheck"].items():
            print(f"  自检 {lab:<5} {msg}")
        print(f"{'对照':<6}{'池化总体':>11}{'单元均值':>11}{'单元中位':>11}{'参考 CRPS':>12}{'对照 CRPS':>12}{'为正的单元':>14}")
        for r in rows:
            print(f"{r['cmp']:<6}{r['pooled_overall']:10.2f}%{r['cell_mean']:10.2f}%{r['cell_median']:10.2f}%"
                  f"{r['crps_ref']:12.4f}{r['crps_cmp']:12.4f}{r['cells_positive']:>9}/{r['cells_total']}")
    else:
        print(f"{'对照':<6}{'方向1均值':>11}{'方向2均值':>11}{'中点均值':>11}{'中点中位':>11}{'中点为正的单元':>16}")
        for r in rows:
            print(f"{r['cmp']:<6}{r['dir1_mean']:10.2f}%{r['dir2_mean']:10.2f}%{r['mid_mean']:10.2f}%"
                  f"{r['mid_median']:10.2f}%{r['mid_cells_positive']:>10}/{r['cells_total']}")
    print(f"色标 ±{lim:.2f}%;  图 {len(mats)} 张 -> {out}")


if __name__ == "__main__":
    main()
