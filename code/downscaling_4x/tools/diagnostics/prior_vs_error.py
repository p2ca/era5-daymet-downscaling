#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
prior_vs_error.py — D-EC 的难度头预测的是什么: 逐 token 对照实际误差
============================================================================
难度头由训练侧用★本次前向的逐 token 损失★做监督, 即它被训练去预测"这个 token 此刻的误差
有多大"。本工具把落盘的逐 token 先验与几个可直接测到的量并排:

  mu_err     阶段A μ 在该 token 上的误差(确定性, CRPS 恒等于 MAE) —— 残差模式下它就是
             阶段B 要去噪的残差幅度, 是先验被训练去预测的那个量最直接的代理
  dec_err    D-EC 集合在该 token 上的 CRPS
  reduction  mu_err − dec_err, 阶段B 实际降下来的量
  rel_red    reduction / mu_err, 同一个量的相对形式
  k          该 token 每层平均路由到的专家数

先落一张逐 (天, token) 的表, 之后所有统计与图都从表里出, 不再重读落场。表里每个 token 记
它覆盖的有效格点数与占比最大的区域; 误差是该 token 内有效格点上的均值。

相关系数分两种, 分开出图, 因为它们答的不是同一个问题:
  pooled   区域-月内所有 (token, 天) 一起算 —— 空间与时间的变化混在一起
  spatial  先对该区域-月内每个 token 跨天取平均再算 —— 只看"先验在空间上排得对不对"

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.prior_vs_error \\
      --routing runs/exp/<dec eval2020> --mu runs/exp/<jda run>/fields2020/<target> \\
      --fields DEC=runs/exp/<dec eval2020> --fields EC=... --fields TC=... \\
      --regions runs/exp/<regions>/regions_v1.npz --members 0 \\
      --show-region 1 --show-region 4 --show-region 10 --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import TokenRegionWeights, parse_spec, slug


def pool_sum(x, P):
    H, W = x.shape
    return x.reshape(H // P, P, W // P, P).sum((1, 3))


def token_roughness(ctx, land, gh, gw, patch, offset):
    """该 offset 下每个 token 覆盖块内的高程标准差(只用有效格点)。"""
    z = np.where(land, ctx.elevation(), 0.0)
    Y, X = np.nonzero(land)
    dy, dx = int(offset[0]), int(offset[1])
    tok = ((Y + dy) // patch) * gw + ((X + dx) // patch)
    T = gh * gw
    n = np.bincount(tok, minlength=T).astype(np.float64)
    s = np.bincount(tok, weights=z[land], minlength=T)
    s2 = np.bincount(tok, weights=z[land] ** 2, minlength=T)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / np.maximum(n, 1) - (s / np.maximum(n, 1)) ** 2
    return np.sqrt(np.maximum(var, 0.0)), n


def build_table(a, ctx, out):
    """逐 (天, token) 的表: 先验、k、各模型误差、区域、有效格点数、粗糙度。"""
    info = RD.load_routing_meta(a.routing)
    if info is None:
        raise SystemExit(f"{a.routing} 下没有 routing/meta.json")
    gh, gw = info["token_grid"]
    patch, L = info["patch"], info["n_moe_layers"]
    items = [(y, d, m) for (y, d, m) in RD.available_routing(a.routing)
             if y == a.year and m in a.members and (a.days is None or d in a.days)]
    if not items:
        raise SystemExit("没有符合条件的路由文件")
    land = ctx.land
    nreg = len(ctx.region_names)
    wmap = TokenRegionWeights(land, ctx.region_id[land], gh, gw, patch, nreg)
    Y, X = np.nonzero(land)
    T = gh * gw
    specs = dict(parse_spec(s) for s in a.fields)
    if "DEC" not in specs:
        raise SystemExit("--fields 里必须有 DEC=<落场目录>")
    err_names = ["MU"] + list(specs)
    fdirs = {"MU": Path(a.mu), **specs}

    rows = {k: [] for k in ("day", "tok", "region", "npix", "rough", "prior", "k")}
    rows.update({f"err_{n}": [] for n in err_names})
    t0 = time.time()
    for n_i, (y, d, m) in enumerate(items):
        r = RD.load_routing(a.routing, y, d, m)
        if "prior_sum_t" not in r:
            raise SystemExit("该落场的路由里没有 prior_sum_t, 不是 D-EC 或先验成分没开")
        w = wmap(r["offset"])                                  # (T, nreg) 有效格点数
        tin = np.asarray(r["tok_in"], bool)
        if not np.array_equal(tin, w.sum(1) > 0):
            raise SystemExit(f"{y}-d{d}-m{m}: tok_in 与按 offset 算出的域内 token 不一致")
        dy, dx = int(r["offset"][0]), int(r["offset"][1])
        tok = ((Y + dy) // patch) * gw + ((X + dx) // patch)   # 每个有效格点属于哪个 token
        npix = np.bincount(tok, minlength=T).astype(np.float64)
        sel = np.nonzero(npix > 0)[0]
        nf = float(r["nfwd"])
        prior = r["prior_sum_t"].sum(0) / nf                   # (T,) 对全部前向取平均
        kk = r["k_sum"].sum(0) / (nf * L)
        rough, _ = token_roughness(ctx, land, gh, gw, patch, r["offset"])
        rows["day"].append(np.full(sel.size, d, np.int16))
        rows["tok"].append(sel.astype(np.int32))
        rows["region"].append((np.argmax(w[sel], axis=1) + 1).astype(np.int16))
        rows["npix"].append(npix[sel].astype(np.float32))
        rows["rough"].append(rough[sel].astype(np.float32))
        rows["prior"].append(prior[sel].astype(np.float32))
        rows["k"].append(kk[sel].astype(np.float32))
        for nm in err_names:
            f = np.load(fdirs[nm] / "crps" / f"{y}_d{d}.npy")[land].astype(np.float64)
            if not np.isfinite(f).all():
                raise SystemExit(f"{nm} 第 {d} 天有效域内含非有限值")
            s = np.bincount(tok, weights=f, minlength=T)
            rows[f"err_{nm}"].append((s[sel] / npix[sel]).astype(np.float32))
        if (n_i + 1) % 60 == 0 or n_i + 1 == len(items):
            print(f"  表 {n_i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    tab = {k: np.concatenate(v) for k, v in rows.items()}
    tab["_err_names"] = np.array(err_names)
    np.savez_compressed(out / "token_table.npz", **tab)
    print(f"  表 {tab['day'].size:,} 行 -> token_table.npz  {time.time() - t0:.0f}s")
    return tab, err_names, len(items)


def safe_spearman(x, y, nmin=30):
    if x.size < nmin or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).correlation)


def decile_curve(ctx, x, series, name, xlabel, ylabel, title, nq=10):
    q = np.quantile(x, np.linspace(0, 1, nq + 1))
    q[-1] += 1e-9
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nq - 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    xs = np.arange(1, nq + 1)
    for lab, v in series.items():
        ax.plot(xs, [v[idx == i].mean() if (idx == i).any() else np.nan for i in range(nq)],
                marker="o", ms=3.5, label=lab)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", required=True, help="带 prior 的 D-EC 落场目录")
    ap.add_argument("--mu", required=True, help="阶段A μ 的逐日场目录(含 crps/)")
    ap.add_argument("--fields", action="append", required=True, help="label=落场目录, 必须含 DEC=")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--show-region", type=int, action="append", default=None,
                    help="要出分位曲线的区域展示编号, 可多次")
    ap.add_argument("--table", default=None, help="已有的 token_table.npz, 给了就不重建")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(Path(a.routing), a.regions, C.TARGETS[0], out, years=(a.year,),
                        model_tag="prior", scales_path=out / "scales.json")
    if a.table:
        z = np.load(a.table)
        tab = {k: z[k] for k in z.files}
        err_names = [str(s) for s in tab.pop("_err_names")]
        nfiles = None
    else:
        tab, err_names, nfiles = build_table(a, ctx, out)
        tab.pop("_err_names")

    nreg = len(ctx.region_names)
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    disp_of_raw = {i + 1: int(ctx.region_ids[i]) for i in range(nreg)}   # 原始 region_id -> 展示编号
    month = ctx.month_of_day[a.year]
    mo = month[tab["day"]]
    disp = np.array([disp_of_raw[int(r)] for r in tab["region"]])

    err_names = [k[4:] for k in tab if k.startswith("err_")]
    mu, dec = tab["err_MU"], tab["err_DEC"]
    red = mu - dec
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(mu > 1e-9, red / np.maximum(mu, 1e-9), np.nan)
    quant = {"mu_err": mu, "dec_err": dec, "reduction": red, "rel_reduction": rel, "k": tab["k"]}
    prior = tab["prior"]

    months = list(range(1, 13))
    mlab = [ctx.MONTHS[m - 1] for m in months]
    stats = {}
    for qn, qv in quant.items():
        for kind in ("pooled", "spatial"):
            M = np.full((nreg, 12), np.nan)
            for i, rd in enumerate(rlab):
                for j, mm in enumerate(months):
                    s = (disp == int(rd)) & (mo == mm)
                    if s.sum() < 30:
                        continue
                    if kind == "pooled":
                        x, y = prior[s], qv[s]
                    else:
                        tk = tab["tok"][s]
                        u, inv = np.unique(tk, return_inverse=True)
                        cnt = np.bincount(inv)
                        x = np.bincount(inv, weights=prior[s]) / cnt
                        y = np.bincount(inv, weights=np.nan_to_num(qv[s])) / cnt
                    g = np.isfinite(x) & np.isfinite(y)
                    M[i, j] = safe_spearman(x[g], y[g])
            stats[f"{qn}__{kind}"] = M
            ctx.heatmap(M, rlab, mlab, f"prior_vs_{qn}_{kind}",
                        cbar=f"Spearman ρ (prior, {qn})", scale_group="rho_rm",
                        cmap="RdBu_r", vmin=-1, vmax=1,
                        title=f"difficulty prior vs {qn} · {kind} · region × month")

    for rd in (a.show_region or []):
        s = disp == int(rd)
        if s.sum() < 100:
            print(f"  区域 {rd} 样本太少, 跳过分位曲线")
            continue
        nm = ctx.region_names[order[[int(x) for x in rlab].index(int(rd))]]
        decile_curve(ctx, prior[s], {"μ error": mu[s], "DEC CRPS": dec[s], "reduction (μ − DEC)": red[s]},
                     f"prior_decile_region{rd}", "difficulty prior decile (within region)",
                     f"error [{ctx.unit}]", f"region {rd} {nm} · binned by difficulty prior")
        decile_curve(ctx, prior[s], {"relative reduction": rel[s]},
                     f"prior_decile_relred_region{rd}", "difficulty prior decile (within region)",
                     "(μ − DEC) / μ", f"region {rd} {nm} · relative reduction by prior decile")
        decile_curve(ctx, tab["k"][s], {"μ error": mu[s], "DEC CRPS": dec[s], "reduction (μ − DEC)": red[s]},
                     f"k_decile_region{rd}", "routed experts per token per layer (decile)",
                     f"error [{ctx.unit}]", f"region {rd} {nm} · binned by routed expert count")

    # 全域: 先验与 k 各自跟粗糙度、跟误差的关系
    decile_curve(ctx, tab["rough"], {"difficulty prior": prior, "routed experts k": tab["k"]},
                 "prior_and_k_by_roughness", "token terrain roughness decile",
                 "prior / k", "difficulty prior and routed expert count by terrain roughness")
    decile_curve(ctx, prior, {"μ error": mu, "DEC CRPS": dec, "reduction (μ − DEC)": red},
                 "prior_decile_all", "difficulty prior decile (whole domain)",
                 f"error [{ctx.unit}]", "whole domain · binned by difficulty prior")

    res = {"year": a.year, "members": a.members, "routing": str(a.routing), "mu": str(a.mu),
           "fields": {l: str(d) for l, d in (parse_spec(s) for s in a.fields)},
           "n_rows": int(tab["day"].size), "n_files": nfiles,
           "note_prior": "难度头的监督是本次前向的逐 token 损失; 表里的先验对该轨迹全部前向取平均",
           "note_member": "路由与先验取自 member 0 的轨迹; 误差场是 32 成员集合量",
           "note_token_region": "每个 token 归入其有效格点最多的区域",
           "overall": {
               "spearman_prior_vs": {qn: safe_spearman(prior[np.isfinite(qv)], qv[np.isfinite(qv)])
                                     for qn, qv in quant.items()},
               "spearman_prior_vs_roughness": safe_spearman(prior, tab["rough"]),
               "spearman_k_vs_prior": safe_spearman(tab["k"], prior),
               "spearman_k_vs_roughness": safe_spearman(tab["k"], tab["rough"])},
           "by_region": {}}
    for i, rd in enumerate(rlab):
        s = disp == int(rd)
        res["by_region"][rd] = {
            "name": ctx.region_names[order[i]], "n_rows": int(s.sum()),
            "mean_prior": float(prior[s].mean()), "mean_k": float(tab["k"][s].mean()),
            "mean_mu_err": float(mu[s].mean()), "mean_dec_err": float(dec[s].mean()),
            "mean_reduction": float(red[s].mean()),
            "mean_rel_reduction": float(np.nanmean(rel[s])),
            "rho_prior_mu": safe_spearman(prior[s], mu[s]),
            "rho_prior_reduction": safe_spearman(prior[s], red[s]),
            "rho_prior_rel_reduction": safe_spearman(prior[s][np.isfinite(rel[s])], rel[s][np.isfinite(rel[s])]),
            "rho_k_reduction": safe_spearman(tab["k"][s], red[s]),
            "mean_err": {nm: float(tab[f"err_{nm}"][s].mean()) for nm in err_names},
            "mean_reduction_by_model": {nm: float((mu[s] - tab[f"err_{nm}"][s]).mean())
                                        for nm in err_names if nm != "MU"}}
    # 区域尺度: 先验/容量 与 误差/降低量 的对照(19 个点)
    def region_vec(key, fn):
        return np.array([fn(disp == int(rd)) for rd in rlab])
    rv = {"prior": region_vec("p", lambda s: prior[s].mean()),
          "k": region_vec("k", lambda s: tab["k"][s].mean()),
          "rough": region_vec("r", lambda s: tab["rough"][s].mean()),
          "mu": region_vec("mu", lambda s: mu[s].mean()),
          "red": region_vec("red", lambda s: red[s].mean())}
    for xn, yn, fname, xl, yl in (
            ("prior", "mu", "region_prior_vs_mu_error", "region mean difficulty prior", f"region mean μ error [{ctx.unit}]"),
            ("prior", "rough", "region_prior_vs_roughness", "region mean difficulty prior", "region mean terrain roughness [m]"),
            ("k", "red", "region_k_vs_reduction", "region mean routed experts per token per layer",
             f"region mean reduction μ − DEC [{ctx.unit}]")):
        fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
        ax.scatter(rv[xn], rv[yn], s=26, color="#4878a8")
        for xi, yi, nm in zip(rv[xn], rv[yn], rlab):
            ax.annotate(nm, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho = safe_spearman(rv[xn], rv[yn], nmin=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}  ·  Spearman ρ = {rho:.2f}", fontsize=9)
        ctx.savefig(fig, fname)
        res["overall"][f"region_rho_{xn}_vs_{yn}"] = rho

    ctx.write_scales()
    np.savez_compressed(out / "rho_cells.npz", region_display_ids=np.array(rlab),
                        months=np.array(months), **stats)
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("\n全域 token 级秩相关(先验 vs):", json.dumps(res["overall"]["spearman_prior_vs"], ensure_ascii=False))
    print("先验 vs 粗糙度:", round(res["overall"]["spearman_prior_vs_roughness"], 3),
          "| k vs 粗糙度:", round(res["overall"]["spearman_k_vs_roughness"], 3))
    print("区域尺度(19 点):", {k: round(v, 3) for k, v in res["overall"].items() if k.startswith("region_rho")})
    print(f"{'区':<15}{'先验':>8}{'k':>7}{'μ误差':>8}{'DEC':>8}{'降低':>8}{'ρ(先验,μ)':>11}{'ρ(先验,降低)':>13}{'ρ(先验,相对降低)':>17}")
    for rd in rlab:
        b = res["by_region"][rd]
        print(f"{rd:>2} {b['name']:<12}{b['mean_prior']:8.3f}{b['mean_k']:7.2f}{b['mean_mu_err']:8.3f}"
              f"{b['mean_dec_err']:8.3f}{b['mean_reduction']:8.3f}{b['rho_prior_mu']:11.3f}"
              f"{b['rho_prior_reduction']:13.3f}{b['rho_prior_rel_reduction']:17.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
