#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_pairwise_regions.py — 一个参考模型对若干对照模型的 (区域, 日) 单元配对胜负
============================================================================
把每个模型的逐像素 CRPS 场(集合方法)与 |集合均值 − 真值| 场按 (区域, 日) 取均值, 得到
(区域数 × 天数) 的单元矩阵; 逐单元判定参考模型是否**同时优于全部对照**(赢)或**同时劣于
全部对照**(输), 其余记为混合。输出:

  overall     赢 / 输 / 混合的单元占比, 以及带相对容差(排除采样噪声)的"明显赢/输"占比
  by_region   逐区域(跨天)的赢/输占比, 附区域内高程粗糙度(P×P 块内高程标准差的区域均值)
  by_month    逐月(跨区域)的赢/输占比
  heatmap     区域 × 月 的赢占比与输占比热图
  difficulty  按单元难度(对照模型该单元分数的均值)十分位统计赢/输占比
  magnitude   赢时与输时的相对差幅(参考对最优对照)分布

口径: 有效域 = 分区文件的 land; 区域用展示编号(地理顺序)标注; 逐日分数取该区域有效格的均值,
CRPS 直接用落场的逐像素 crps 场(集合方法为公平估计式), MAE 用 ens_mean 与真值现算。
纯 CPU, 单图, 同组共色标。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.stage_b_pairwise_regions \\
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation.render.context import RenderContext


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def pool_sum(x, P):
    H, W = x.shape
    return x.reshape(H // P, P, W // P, P).sum((1, 3))


def region_means(field, rid, nreg):
    """field: 有效域内的一维取值; rid: 同长的区域编号(1..nreg) -> 长 nreg 的区域均值。"""
    s = np.bincount(rid, weights=field, minlength=nreg + 1)[1:]
    n = np.bincount(rid, minlength=nreg + 1)[1:]
    return s / np.maximum(n, 1)


def grouped_bars(ctx, groups, labels, name, ylabel, title=None, rotate=0, ylim=None):
    k = len(groups)
    x = np.arange(len(labels))
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(labels) + 2.0), 3.8), constrained_layout=True)
    for i, (lab, vals) in enumerate(groups.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="label=eval_dir, 参考模型")
    ap.add_argument("--cmp", action="append", required=True, help="label=eval_dir, 对照模型, 可多次")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=16, help="区域粗糙度用的块边长(px)")
    ap.add_argument("--tol", type=float, default=0.01, help="'明显'赢/输的相对容差")
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()

    lr, dr = parse_spec(a.ref)
    cmps = [parse_spec(s) for s in a.cmp]
    labels = [lr] + [l for l, _ in cmps]
    dirs = {lr: dr, **{l: d for l, d in cmps}}
    y = a.year
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(dr, a.regions, a.target, out, years=(y,), model_tag=f"{slug(lr)}-vs-{'-'.join(slug(l) for l, _ in cmps)}")
    land = ctx.land
    nreg = len(ctx.region_names)
    rid = ctx.region_id[land]
    if rid.min() < 1 or rid.max() > nreg:
        raise SystemExit("region_id 超出 1..nreg, 与 regions_v1.npz 约定不符")
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    month = ctx.month_of_day[y][:nd]
    order = np.argsort([int(i) for i in ctx.region_ids])          # 展示编号顺序
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]

    # 区域粗糙度: P×P 块内高程标准差, 落回像素后取区域均值
    z = np.where(land, ctx.elevation(), 0.0)
    P = a.patch
    n_k = pool_sum(land.astype(np.float64), P)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_k = pool_sum(z * z, P) / np.maximum(n_k, 1) - (pool_sum(z, P) / np.maximum(n_k, 1)) ** 2
    rough_px = np.repeat(np.repeat(np.sqrt(np.maximum(var_k, 0.0)), P, 0), P, 1)[land]
    rough_reg = region_means(rough_px, rid, nreg)

    # (模型, 天, 区域) 的 CRPS 与 MAE
    S = {m: {lab: np.zeros((nd, nreg)) for lab in labels} for m in ("crps", "mae")}
    t0 = time.time()
    for t in range(nd):
        tr = ctx.truth(y, t)[land]
        for lab in labels:
            d = dirs[lab]
            cr = np.load(d / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
            em = np.load(d / "ens_mean" / f"{y}_d{t}.npy")[land].astype(np.float64)
            if not (np.isfinite(cr).all() and np.isfinite(em).all()):
                raise SystemExit(f"{lab} 第 {t} 天有效域内含非有限值")
            S["crps"][lab][t] = region_means(cr, rid, nreg)
            S["mae"][lab][t] = region_means(np.abs(em - tr), rid, nreg)
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    res = {"ref": lr, "cmp": [l for l, _ in cmps], "target": a.target, "year": y, "n_days": nd,
           "n_regions": nreg, "tol": a.tol, "regions": {"display_ids": rlab, "names": rname,
           "roughness_std_m": [float(rough_reg[i]) for i in order]}}
    for metric in ("crps", "mae"):
        ref = S[metric][lr]
        others = np.stack([S[metric][l] for l, _ in cmps], 0)           # (k, nd, nreg)
        best_other = others.min(0)
        worst_other = others.max(0)
        win = ref < best_other                                          # 同时优于全部对照
        loss = ref > worst_other                                        # 同时劣于全部对照
        win_clear = ref < best_other * (1 - a.tol)
        loss_clear = ref > worst_other * (1 + a.tol)
        rel_vs_best = (ref - best_other) / best_other                   # <0 赢
        rel_vs_worst = (ref - worst_other) / worst_other                # >0 输
        diff_level = others.mean(0)                                     # 单元难度: 对照的均值分数
        dec = np.digitize(diff_level, np.quantile(diff_level, np.linspace(0.1, 0.9, 9)))
        R = {"overall": {"win": float(win.mean()), "loss": float(loss.mean()), "mixed": float(1 - win.mean() - loss.mean()),
                         "win_clear": float(win_clear.mean()), "loss_clear": float(loss_clear.mean()),
                         "n_cells": int(win.size)},
             "by_region": {"display_ids": rlab, "win": [float(win[:, i].mean()) for i in order],
                           "loss": [float(loss[:, i].mean()) for i in order],
                           "mean_score_ref": [float(ref[:, i].mean()) for i in order],
                           "mean_score_best_other": [float(best_other[:, i].mean()) for i in order]},
             "by_month": {"months": sorted(set(month.tolist())),
                          "win": [float(win[month == m].mean()) for m in sorted(set(month.tolist()))],
                          "loss": [float(loss[month == m].mean()) for m in sorted(set(month.tolist()))]},
             "by_difficulty_decile": {"decile": list(range(1, 11)),
                                      "win": [float(win[dec == i].mean()) for i in range(10)],
                                      "loss": [float(loss[dec == i].mean()) for i in range(10)],
                                      "score_range": [[float(diff_level[dec == i].min()), float(diff_level[dec == i].max())] for i in range(10)]},
             "magnitude": {"rel_vs_best_when_win_median": float(np.median(rel_vs_best[win])) if win.any() else None,
                           "rel_vs_worst_when_loss_median": float(np.median(rel_vs_worst[loss])) if loss.any() else None,
                           "rel_vs_best_all_median": float(np.median(rel_vs_best)),
                           "rel_vs_best_all_mean": float(rel_vs_best.mean())},
             "pairwise": {l: {"ref_better_share": float((ref < S[metric][l]).mean())} for l, _ in cmps}}
        # 区域 × 月
        months = R["by_month"]["months"]
        hm_w = np.array([[win[month == m][:, i].mean() for m in months] for i in order])
        hm_l = np.array([[loss[month == m][:, i].mean() for m in months] for i in order])
        R["region_month_win"] = hm_w.round(3).tolist()
        R["region_month_loss"] = hm_l.round(3).tolist()
        # 区域胜率与粗糙度的秩相关
        from scipy.stats import spearmanr
        R["spearman_region_win_vs_roughness"] = float(spearmanr(R["by_region"]["win"], res["regions"]["roughness_std_m"]).correlation)
        R["spearman_region_loss_vs_roughness"] = float(spearmanr(R["by_region"]["loss"], res["regions"]["roughness_std_m"]).correlation)
        res[metric] = R

        # ---- 图 ----
        mlab = [ctx.MONTHS[m - 1] for m in months]
        grouped_bars(ctx, {"win (beats all)": R["by_region"]["win"], "loss (worse than all)": R["by_region"]["loss"]},
                     rlab, f"{metric}_by_region", "share of days", title=f"{metric.upper()} · {lr} vs {', '.join(res['cmp'])} · by region", ylim=(0, 1))
        grouped_bars(ctx, {"win (beats all)": R["by_month"]["win"], "loss (worse than all)": R["by_month"]["loss"]},
                     mlab, f"{metric}_by_month", "share of region-days", title=f"{metric.upper()} · by month", ylim=(0, 1))
        grouped_bars(ctx, {"win (beats all)": R["by_difficulty_decile"]["win"], "loss (worse than all)": R["by_difficulty_decile"]["loss"]},
                     [f"D{i}" for i in range(1, 11)], f"{metric}_by_difficulty", "share of region-days",
                     title=f"{metric.upper()} · by difficulty decile (mean score of comparators)", ylim=(0, 1))
        ctx.heatmap(hm_w, rlab, mlab, f"{metric}_region_month_win", cbar="win share", scale_group="win_share_hm",
                    cmap="Blues", vmin=0, vmax=1, title=f"{metric.upper()} · share of days {lr} beats all · region × month")
        ctx.heatmap(hm_l, rlab, mlab, f"{metric}_region_month_loss", cbar="loss share", scale_group="loss_share_hm",
                    cmap="Reds", vmin=0, vmax=1, title=f"{metric.upper()} · share of days {lr} worse than all · region × month")
        # 相对差幅直方图
        fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
        ax.hist(rel_vs_best.ravel() * 100, bins=80, color="#4878a8")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"({lr} − best comparator) / best comparator [%]  (<0: {lr} better)")
        ax.set_ylabel("region-days")
        ax.set_title(f"{metric.upper()} · relative gap to best comparator", fontsize=10)
        ctx.savefig(fig, f"{metric}_rel_gap_hist")

    ctx.write_scales()
    np.savez(out / "cells.npz", month=month, region_display_ids=np.array(ctx.region_ids), region_names=np.array(ctx.region_names),
             **{f"{m}_{slug(l)}": S[m][l] for m in ("crps", "mae") for l in labels})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for metric in ("crps", "mae"):
        o = res[metric]["overall"]
        print(f"{metric}: win {o['win']:.3f} loss {o['loss']:.3f} mixed {o['mixed']:.3f} | clear(tol {a.tol}) win {o['win_clear']:.3f} loss {o['loss_clear']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
