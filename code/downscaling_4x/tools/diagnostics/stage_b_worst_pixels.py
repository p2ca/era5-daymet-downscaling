#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_worst_pixels.py — 参考模型最差的一批 (像素, 日) 与其余像素上的分方法得分
============================================================================
与 stage_b_worst_cells 同一件事, 但单元细到★单个格点★: 6,935 个 (区域, 日) 单元换成
79,960,185 个 (像素, 日) 单元。挑出参考模型相对劣势最大的一批, 在最差集与其余集上分别给出
全部模型的 MAE / RMSE / bias / corr / CRPS / SSIM。

挑选规则(--rule, 可多给):
  rel_gap   (ref − min(对照)) / min(对照) 最大的 q 比例
  abs_gap   (ref − min(对照)) 最大的 q 比例
  abs_own   ref ★自身★ CRPS 最大的 q 比例; 不看对照, 所以每个模型的集合各不相同 ——
            它回答的是"这个模型自己最差的那批格点长什么样", 不是"它相对谁更差"

★细到像素后必须一起看的两件事★

1) 逐像素 CRPS 是 32 个成员对一个格点的单样本估计, 噪声远大于区域日均值; 按"参考减最优
   对照"排序取头部, 选出的主要是★对照恰好蒙对★的格点, 而不是参考系统性做得差的地方。
   因此本工具默认对每个模型各做一次同样的挑选(--null-refs): 若三个模型作为参考时的最差集
   落差量级相同, 说明落差来自挑选程序本身。
2) rel_gap 的分母是逐像素 CRPS, 可以任意接近 0, 相对劣势会被小分母放大到无意义。输出里
   给了最差集内 min(对照) 的分位数与"分母 < 0.05 K 的占比", 用来判断这一档有多严重;
   abs_gap 没有这个问题, 两者并列给出。

口径与 stage_b_worst_cells 一致: RMSE/MAE/bias/corr/CRPS 走逐(天,像素)池化; SSIM 取
metrics.ssim_field 的 SSIM 图在 (子集 ∩ 腐蚀 5px 陆地) 上平均, data_range 仍是当日全域
truth 的陆地 max-min。SSIM 在散点式子集上只是"这些格点所在窗口的结构相似度"的平均, 窗口
本身仍覆盖未入选的邻域, 解释力弱于连片子集。corr 与 SSIM 一样不可跨子集比较。
全集自检: 池化出的六个量必须复现各落场目录 metrics.json 的官方标量。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.stage_b_worst_pixels \\
      --ref jmb-dec-jda=<dec eval2020> --cmp jmb-tc-jda=<tc eval2020> --cmp jdb-jda=<jdb eval2020> \\
      --regions <regions_v1.npz> --out runs/exp/<diag>
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

REL_FLOOR = 1e-12          # rel_gap 分母的数值下限; 只防除零, 不改变量级


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def load_all(ctx, dirs, labels, y, nd, land):
    """整年逐像素的 crps / 误差 / SSIM 读进内存(有效域内展平)。"""
    nland = int(land.sum())
    truth = np.empty((nd, nland), np.float32)
    A = {lab: {k: np.empty((nd, nland), np.float32) for k in ("crps", "err", "ssim")} for lab in labels}
    t0 = time.time()
    for t in range(nd):
        tf = ctx.truth(y, t)
        tr = tf[land]
        if not np.isfinite(tr).all():
            raise SystemExit(f"第 {t} 天真值在有效域内含非有限值")
        truth[t] = tr
        for lab in labels:
            d0 = dirs[lab]
            em_full = np.load(d0 / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
            cr = np.load(d0 / "crps" / f"{y}_d{t}.npy")[land].astype(np.float64)
            em = em_full[land]
            if not (np.isfinite(em).all() and np.isfinite(cr).all()):
                raise SystemExit(f"{lab} 第 {t} 天有效域内含非有限值")
            A[lab]["crps"][t] = cr
            A[lab]["err"][t] = em - tr
            sm = MT.ssim_field(em_full, tf, land)
            A[lab]["ssim"][t] = sm[land] if sm is not None else np.nan
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)
    return truth, A


def pool(truth, a, sel, er):
    """子集 sel 上的池化指标; er 是有效域内的腐蚀陆地布尔, 只用于 SSIM。"""
    n = 0
    se = ae = be = sp = st = spp = stt = spt = cs = ss = 0.0
    sn = 0
    for t in range(truth.shape[0]):
        m = sel[t]
        if not m.any():
            continue
        e = a["err"][t][m].astype(np.float64)
        tt = truth[t][m].astype(np.float64)
        p = tt + e
        n += e.size
        se += float((e * e).sum()); ae += float(np.abs(e).sum()); be += float(e.sum())
        sp += float(p.sum()); st += float(tt.sum())
        spp += float((p * p).sum()); stt += float((tt * tt).sum()); spt += float((p * tt).sum())
        cs += float(a["crps"][t][m].astype(np.float64).sum())
        m2 = m & er
        if m2.any():
            v = a["ssim"][t][m2].astype(np.float64)
            g = np.isfinite(v)
            ss += float(v[g].sum()); sn += int(g.sum())
    if n == 0:
        return None
    mp, mt = sp / n, st / n
    cov = spt / n - mp * mt
    vp = spp / n - mp * mp
    vt = stt / n - mt * mt
    return {"mae": ae / n, "rmse": (se / n) ** 0.5, "bias": be / n,
            "corr": cov / ((vp * vt) ** 0.5) if vp > 0 and vt > 0 else float("nan"),
            "crps": cs / n, "ssim": ss / sn if sn else float("nan"),
            "n_px": int(n), "n_px_ssim": int(sn)}


def top_frac(key, frac):
    """key 最大的 frac 比例的布尔掩膜与阈值(并列值一并入选)。"""
    flat = key.reshape(-1)
    k = max(1, int(round(frac * flat.size)))
    thr = float(np.partition(flat, flat.size - k)[flat.size - k])
    return key >= thr, thr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="label=eval_dir, 参考模型")
    ap.add_argument("--cmp", action="append", required=True, help="label=eval_dir, 对照模型, 可多次")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", action="append", default=None, choices=("rel_gap", "abs_gap", "abs_own"))
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--null-refs", type=int, default=1,
                    help="1=对每个模型各做一次同样挑选, 用来分离挑选程序自身造成的落差")
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--selfcheck-tol", type=float, default=3e-4)
    ap.add_argument("--figs", type=int, default=1, help="1=出最差集入选频次地图")
    a = ap.parse_args()

    lr, dr = parse_spec(a.ref)
    cmps = [parse_spec(s) for s in a.cmp]
    labels = [lr] + [l for l, _ in cmps]
    dirs = {lr: dr, **{l: d for l, d in cmps}}
    rules = a.rule or ["rel_gap", "abs_gap"]
    y = a.year
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(dr, a.regions, a.target, out, years=(y,), model_tag=f"{slug(lr)}-worstpx")
    land = ctx.land
    er = MT.eroded_land_mask(land)[land]
    rid = ctx.region_id[land]
    nreg = len(ctx.region_names)
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    month = ctx.month_of_day[y][:nd]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    rname = [ctx.region_names[i] for i in order]
    reg_px = np.bincount(rid, minlength=nreg + 1)[1:].astype(np.float64)

    truth, A = load_all(ctx, dirs, labels, y, nd, land)

    # ---- 自检: 全集池化必须复现各落场 metrics.json 的官方标量 ----
    full = np.ones((nd, rid.size), bool)
    checks = {}
    for lab in labels:
        got = pool(truth, A[lab], full, er)
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
    print(f"[worstpx] 全集自检 {checks}", flush=True)

    res = {"ref": lr, "cmp": [l for l, _ in cmps], "target": a.target, "unit": ctx.unit,
           "year": y, "n_days": nd, "frac": a.frac, "n_pixel_days": int(nd * rid.size),
           "selfcheck_vs_metrics_json": checks,
           "regions": {"display_ids": rlab, "names": rname},
           "caveats": ["逐像素 CRPS 是单样本估计, 按'参考减对照'取头部会大量选中对照蒙对的格点;"
                       " 三个模型各作一次参考的对照结果在 sets 里, 用来分离挑选程序本身的落差",
                       "rel_gap 的分母可任意接近 0, 见每个集合的 denom_quantiles 与 denom_lt_0.05_share",
                       "corr 与 SSIM 不可跨子集比较; SSIM 在散点子集上窗口仍覆盖未入选邻域"],
           "sets": {}}

    refs = labels if a.null_refs else [lr]
    freq_saved = {}
    for rf in refs:
        oth = np.stack([A[l]["crps"] for l in labels if l != rf], 0)
        best = oth.min(0)
        gap = A[rf]["crps"] - best
        for rule in rules:
            key = {"rel_gap": lambda: gap / np.maximum(best, REL_FLOOR),
                   "abs_gap": lambda: gap,
                   "abs_own": lambda: A[rf]["crps"]}[rule]()
            sel, thr = top_frac(key, a.frac)
            del key
            rest = ~sel
            bsel = best[sel]
            blk = {"ref": rf, "rule": rule, "threshold": thr,
                   "share_px": float(sel.mean()),
                   "denom_quantiles": {str(q): float(np.quantile(bsel, q)) for q in (0.1, 0.5, 0.9)},
                   "denom_quantiles_all": {str(q): float(np.quantile(best, q)) for q in (0.1, 0.5, 0.9)},
                   "denom_lt_0.05_share": float((bsel < 0.05).mean()),
                   "worst": {lab: pool(truth, A[lab], sel, er) for lab in labels},
                   "rest": {lab: pool(truth, A[lab], rest, er) for lab in labels},
                   "composition": {
                       "by_region_share": {rlab[j]: float(np.bincount(rid, weights=sel.sum(0), minlength=nreg + 1)[1:][order[j]]
                                                          / (reg_px[order[j]] * nd)) for j in range(nreg)},
                       "by_month_share": {int(m): float(sel[month == m].mean()) for m in sorted(set(month.tolist()))}}}
            res["sets"][f"{slug(rf)}__{rule}"] = blk
            freq_saved[f"{slug(rf)}__{rule}"] = sel.sum(0).astype(np.int16)
            del sel, rest
            print(f"[worstpx] {rf} {rule} 完成", flush=True)
        del oth, best, gap

    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(out / "select_freq.npz", land=land, month=month,
                        region_names=np.array(ctx.region_names), **freq_saved)

    if a.figs:
        for name, f in freq_saved.items():
            g = np.full(land.shape, np.nan)
            g[land] = f
            ctx.map(g, f"select_freq__{name}", cmap="magma", vmin=0, vmax=float(np.quantile(f, 0.999)),
                    scale_group="select_freq", cbar="days in worst set",
                    title=f"days in worst {a.frac*100:.0f}% · {name.replace('__', ' · ')}")
        ctx.write_scales()

    mk = ("mae", "rmse", "bias", "corr", "crps", "ssim")
    for name, blk in res["sets"].items():
        print(f"\n=== {name}  最差 {blk['share_px']*100:.1f}% 像素日  阈值 {blk['threshold']:.4g} "
              f"| 分母中位数 {blk['denom_quantiles']['0.5']:.4f} (全集 {blk['denom_quantiles_all']['0.5']:.4f}), "
              f"分母<0.05K 占 {blk['denom_lt_0.05_share']*100:.1f}% ===")
        for part in ("worst", "rest"):
            print(f"  -- {part} --  " + "".join(f"{k:>9}" for k in mk))
            for lab in labels:
                v = blk[part][lab]
                print(f"  {lab:<14}" + "".join(f"{v[k]:9.4f}" for k in mk))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
