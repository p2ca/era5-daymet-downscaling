#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
region_daily_profile.py — 逐区域的集合成绩拆解与逐日序列
============================================================================
一次扫描逐日场, 对每个 (区域, 天) 累出 CRPS、|集合均值 − 真值|、(集合均值 − 真值)²、
集合离散度, 以及当日该区域真值的空间标准差与平均梯度幅度; rank 场按区域累成名次直方图。

据此分两件事:

  赢在哪   同时给 CRPS 与 |集合均值 − 真值| 相对参考模型的差。两者差不多 -> 差异来自
           集合均值本身; CRPS 的差明显大于均值的差 -> 多出来的部分来自集合离散度与名次
           分布(校准), 与均值准不准是两回事。
  什么时候输 指定区域的逐日相对差序列, 并与当日该区域真值的空间标准差、平均梯度并排,
           看输的日子是不是集中在某一类天气。

口径: 有效域取分区文件的 land; 逐区域池化 = 该区域全部有效格点-日先求和再相除。
名次直方图用落场的 rank 场(0..成员数, 域外为 -1), 平坦表示集合离散度与误差匹配。
真值的梯度在整幅上算再取区域内均值, 区域边界处的梯度含区域外的邻格。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.region_daily_profile \\
      --model DEC=runs/exp/<dec eval2020> --model EC=... --model TC=... --model JDB=... \\
      --ref DEC --regions runs/exp/<regions>/regions_v1.npz \\
      --show-region 1 --show-region 4 --show-region 10 --out runs/exp/<diag>
============================================================================
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x import contract as C
from downscaling_4x.evaluation.render.context import RenderContext
from downscaling_4x.tools.plotting.plot_expert_region_month import parse_spec, slug


def grouped_bars(ctx, groups, labels, name, ylabel, title, rotate=45, hline=None):
    k = len(groups)
    x = np.arange(len(labels))
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(labels) + 2.2), 3.9), constrained_layout=True)
    for i, (lab, vals) in enumerate(groups.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=lab)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    return ctx.savefig(fig, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--show-region", type=int, action="append", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    dirs = dict(specs)
    if a.ref not in labels:
        raise SystemExit(f"--ref {a.ref} 不在 {labels} 里")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y = a.year
    ctx = RenderContext(dirs[a.ref], a.regions, C.TARGETS[0], out, years=(y,),
                        model_tag=slug(a.ref), scales_path=out / "scales.json")
    land = ctx.land
    nreg = len(ctx.region_names)
    rid = ctx.region_id[land]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    npx = np.bincount(rid, minlength=nreg + 1)[1:].astype(np.float64)

    members = {}
    for lab in labels:
        mj = json.load(open(dirs[lab] / "meta.json"))
        members[lab] = int(mj.get("members", 1) or 1)
        if mj.get("target") != C.TARGETS[0]:
            raise SystemExit(f"{lab} 的 target 是 {mj.get('target')}, 与 {C.TARGETS[0]} 不符")

    S = {k: {l: np.zeros((nd, nreg)) for l in labels} for k in ("crps", "abserr", "sqerr", "err", "spread")}
    RH = {l: np.zeros((nreg, 34)) for l in labels}          # 名次 0..成员数
    tstd = np.zeros((nd, nreg)); tgrad = np.zeros((nd, nreg))
    t0 = time.time()
    for t in range(nd):
        tr2 = ctx.truth(y, t)
        tr = tr2[land]
        gy, gx = np.gradient(np.nan_to_num(tr2, nan=0.0))
        gmag = np.sqrt(gy ** 2 + gx ** 2)[land]
        s1 = np.bincount(rid, weights=tr, minlength=nreg + 1)[1:]
        s2 = np.bincount(rid, weights=tr ** 2, minlength=nreg + 1)[1:]
        tstd[t] = np.sqrt(np.maximum(s2 / npx - (s1 / npx) ** 2, 0.0))
        tgrad[t] = np.bincount(rid, weights=gmag, minlength=nreg + 1)[1:] / npx
        for lab in labels:
            d = dirs[lab]
            cr = np.load(d / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
            em = np.load(d / "ens_mean" / f"{y}_d{t}.npy")[land].astype(np.float64)
            sp = np.load(d / "spread" / f"{y}_d{t}.npy")[land].astype(np.float64) \
                if (d / "spread").is_dir() else np.zeros_like(cr)
            e = em - tr
            for k, v in (("crps", cr), ("abserr", np.abs(e)), ("sqerr", e ** 2), ("err", e), ("spread", sp)):
                S[k][lab][t] = np.bincount(rid, weights=v, minlength=nreg + 1)[1:]
            rk = np.load(d / "rank" / f"{y}_d{t}.npy")[land].astype(np.int64) \
                if (d / "rank").is_dir() else None
            if rk is not None:
                ok = rk >= 0
                RH[lab] += np.bincount(rid[ok] * 34 + rk[ok], minlength=nreg * 34 + 34)[34:].reshape(nreg, 34)
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)

    tot = npx * nd
    P = {lab: {"crps": S["crps"][lab].sum(0) / tot, "mae": S["abserr"][lab].sum(0) / tot,
               "rmse": np.sqrt(S["sqerr"][lab].sum(0) / tot), "bias": S["err"][lab].sum(0) / tot,
               "spread": S["spread"][lab].sum(0) / tot} for lab in labels}
    for lab in labels:
        P[lab]["spread_skill"] = P[lab]["spread"] / np.maximum(P[lab]["rmse"], 1e-12)

    res = {"year": y, "n_days": nd, "ref": a.ref, "members": members,
           "regions": {"display_ids": rlab, "names": rname, "n_pixels": npx[order].astype(int).tolist()},
           "note_rank": "名次直方图用落场 rank 场, 平坦 = 集合离散度与误差匹配",
           "pooled_overall": {lab: {"crps": float(S["crps"][lab].sum() / tot.sum()),
                                    "mae": float(S["abserr"][lab].sum() / tot.sum()),
                                    "rmse": float(np.sqrt(S["sqerr"][lab].sum() / tot.sum())),
                                    "spread": float(S["spread"][lab].sum() / tot.sum())} for lab in labels},
           "by_region": {}}
    for i, rd in enumerate(rlab):
        j = order[i]
        res["by_region"][rd] = {"name": rname[i],
                                **{lab: {k: float(P[lab][k][j]) for k in
                                         ("crps", "mae", "rmse", "bias", "spread", "spread_skill")}
                                   for lab in labels}}

    # ---- 图: 赢在均值还是赢在校准 ----
    cmps = [l for l in labels if l != a.ref]
    rel_crps = {l: (P[l]["crps"][order] - P[a.ref]["crps"][order]) / P[l]["crps"][order] * 100 for l in cmps}
    rel_mae = {l: (P[l]["mae"][order] - P[a.ref]["mae"][order]) / P[l]["mae"][order] * 100 for l in cmps}
    rlab2 = [f"{d} {n}" for d, n in zip(rlab, rname)]
    grouped_bars(ctx, rel_crps, rlab2, "rel_crps_by_region", f"({{cmp}} − {a.ref}) / {{cmp}}  [%]".replace("{cmp}", "cmp"),
                 f"CRPS relative difference vs {a.ref} · by region", hline=0.0)
    grouped_bars(ctx, rel_mae, rlab2, "rel_mae_ens_mean_by_region", "(cmp − ref) / cmp  [%]",
                 f"|ens mean − truth| relative difference vs {a.ref} · by region", hline=0.0)
    for l in cmps:
        grouped_bars(ctx, {"CRPS": rel_crps[l], "|ens mean − truth|": rel_mae[l]}, rlab2,
                     f"rel_crps_vs_mae_{slug(l)}", "(cmp − ref) / cmp  [%]",
                     f"{a.ref} vs {l} · CRPS and ensemble-mean error side by side", hline=0.0)
    grouped_bars(ctx, {l: P[l]["spread_skill"][order] for l in labels}, rlab2,
                 "spread_skill_by_region", "spread / RMSE(ens mean)",
                 "ensemble spread over ensemble-mean RMSE · by region", hline=1.0)
    grouped_bars(ctx, {l: P[l]["spread"][order] for l in labels}, rlab2,
                 "spread_by_region", f"ensemble spread [{ctx.unit}]", "ensemble spread · by region")

    # ---- 图: 指定区域的名次直方图与逐日序列 ----
    show = a.show_region or []
    month = ctx.month_of_day[y][:nd]
    daily = {}
    for rd in show:
        i = rlab.index(str(rd)) if str(rd) in rlab else None
        if i is None:
            print(f"  区域 {rd} 不在分区里, 跳过"); continue
        j = order[i]
        nm = rname[i]
        nb = max(members[l] for l in labels) + 1
        fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
        for lab in labels:
            h = RH[lab][j, :nb]
            ax.plot(np.arange(nb), h / h.sum(), marker="o", ms=2.5, lw=1.0, label=lab)
        ax.axhline(1.0 / nb, color="k", lw=0.8, ls="--")
        ax.set_xlabel("rank of truth among ensemble members")
        ax.set_ylabel("frequency")
        ax.set_title(f"region {rd} {nm} · rank histogram", fontsize=10)
        ctx.savefig(fig, f"rank_hist_region{rd}")

        dref = S["crps"][a.ref][:, j] / npx[j]
        fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
        for l in cmps:
            dl = S["crps"][l][:, j] / npx[j]
            ax.plot(np.arange(nd), (dl - dref) / dl * 100, lw=0.9, label=f"vs {l}")
            daily[f"{rd}__{l}"] = (dl - dref) / dl * 100
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("day index (2020)")
        ax.set_ylabel(f"(cmp − {a.ref}) / cmp  [%]")
        ax.set_title(f"region {rd} {nm} · daily CRPS relative difference", fontsize=10)
        ax.legend(fontsize=8)
        ctx.savefig(fig, f"daily_rel_crps_region{rd}")

        for xn, xv, xl in (("truth_std", tstd[:, j], f"within-region std of truth [{ctx.unit}]"),
                           ("truth_grad", tgrad[:, j], f"within-region mean |grad truth| [{ctx.unit}/px]")):
            fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
            from scipy.stats import spearmanr
            txt = []
            for l in cmps:
                v = daily[f"{rd}__{l}"]
                ax.scatter(xv, v, s=12, alpha=0.65, label=l)
                txt.append(f"{l} ρ={spearmanr(xv, v).correlation:.2f}")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel(xl); ax.set_ylabel(f"(cmp − {a.ref}) / cmp  [%]")
            ax.set_title(f"region {rd} {nm} · daily difference vs {xn}  ·  " + ", ".join(txt), fontsize=9)
            ax.legend(fontsize=8)
            ctx.savefig(fig, f"daily_rel_crps_vs_{xn}_region{rd}")

        from scipy.stats import spearmanr
        res["by_region"][str(rd)]["daily"] = {
            l: {"mean_rel_diff": float(daily[f"{rd}__{l}"].mean()),
                "days_ref_better": int((daily[f"{rd}__{l}"] > 0).sum()),
                "worst_days_for_ref": np.argsort(daily[f"{rd}__{l}"])[:10].tolist(),
                "best_days_for_ref": np.argsort(daily[f"{rd}__{l}"])[-10:][::-1].tolist(),
                "rho_vs_truth_std": float(spearmanr(tstd[:, j], daily[f"{rd}__{l}"]).correlation),
                "rho_vs_truth_grad": float(spearmanr(tgrad[:, j], daily[f"{rd}__{l}"]).correlation),
                "by_month_mean": [float(daily[f"{rd}__{l}"][month == m].mean()) for m in range(1, 13)]}
            for l in cmps}

    ctx.write_scales()
    np.savez_compressed(out / "region_daily.npz", region_display_ids=np.array(rlab),
                        region_names=np.array(rname), month=month, npix=npx[order],
                        truth_std=tstd[:, order], truth_grad=tgrad[:, order],
                        **{f"{k}__{slug(l)}": S[k][l][:, order] for k in S for l in labels},
                        **{f"rankhist__{slug(l)}": RH[l][order] for l in labels})
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n{'模型':<6}{'CRPS':>9}{'MAE(均值)':>11}{'RMSE':>9}{'spread':>9}{'spread/RMSE':>13}")
    for lab in labels:
        p = res["pooled_overall"][lab]
        print(f"{lab:<6}{p['crps']:9.4f}{p['mae']:11.4f}{p['rmse']:9.4f}{p['spread']:9.4f}"
              f"{p['spread']/p['rmse']:13.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
