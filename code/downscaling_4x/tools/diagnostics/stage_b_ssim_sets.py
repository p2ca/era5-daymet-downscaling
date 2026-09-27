#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
stage_b_ssim_sets.py — 按逐像素 SSIM 排序的最差集 / 最优集与其余集
============================================================================
与按 CRPS 排序的 stage_b_worst_pixels 是同一类分析, 但排序变量换成逐像素 SSIM。四条规则:

  ssim_worst_global   全年所有 (像素, 日) 里 SSIM 最低的 q 比例
  ssim_best_global    全年所有 (像素, 日) 里 SSIM 最高的 q 比例
  ssim_worst_daily    每天各取该日 SSIM 最低的 q 比例(每天贡献相同份额)
  ssim_best_daily     每天各取该日 SSIM 最高的 q 比例

★换成 SSIM 排序后与 CRPS 排序的四点不同, 读数前必须知道:

1) 定义域小一圈。SSIM 只在★腐蚀 5px 的陆地★上有定义(高斯窗不得跨海岸), 因此本工具的全部
   选取与池化都在腐蚀域上做 —— 最差集与其余集之和是腐蚀域, 不是有效域。海岸带整条被排除在外,
   这类分析在结构上看不见海岸失败。输出里给了腐蚀域上的全集参考行, 子集数字要和它比, 不要和
   落场 metrics.json 的官方标量比(后者在完整有效域上)。
2) 它是窗口量不是点量。每个 SSIM 值是 11x11 高斯加权统计, 相邻像素共享大部分窗口, 所以选出的
   是"窗口中心", 集合在空间上高度成块, 一个坏点会把包含它的上百个像素一起拉低。
3) 跨天的尺子不同。C1/C2 正比于当日 data_range 的平方, 而 data_range 是当日全域真值的极差。
   极差小的日子稳定常数相对更大, SSIM 被推向 1 —— 所以 global 排序里含有"哪天对比度大"的成分;
   daily 排序每天各取同样份额, 把日级差异剥掉, 只留空间结构。两种都给, 结论要分开说。
4) 在按 SSIM 挑出的集合上再报 SSIM 是同义反复(被排序变量自身的截尾均值)。有信息量的是同一集合
   上的 MAE / RMSE / bias / corr / CRPS, 以及 SSIM 的三个分量 l(均值) / c(振幅) / s(结构):
   它们回答"结构最坏的地方是不是逐点误差也大""坏在哪一项"。

自检: 载入时同时在★完整有效域★上池化 MAE/RMSE/bias/corr/CRPS 并与各落场 metrics.json 对拍;
SSIM 取腐蚀域上 l*c*s 的均值与官方标量对拍。两项都不一致即当场退出。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.stage_b_ssim_sets \\
      --model jmb-dec-jda=<dec eval2020> --model jmb-tc-jda=<tc eval2020> --model jdb-jda=<jdb eval2020> \\
      --regions <regions_v1.npz> --out runs/exp/<diag> --frac 0.05
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

RULES = ("ssim_worst_global", "ssim_best_global", "ssim_worst_daily", "ssim_best_daily")


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


class FullAcc:
    """完整有效域上的逐(天,像素)池化累计, 只为自检, 与 eval_common 同式。"""
    def __init__(self):
        self.n = self.se = self.ae = self.be = 0.0
        self.sp = self.st = self.spp = self.stt = self.spt = self.cs = 0.0

    def add(self, pred, tgt, crps):
        d = pred - tgt
        self.n += d.size
        self.se += float((d * d).sum()); self.ae += float(np.abs(d).sum()); self.be += float(d.sum())
        self.sp += float(pred.sum()); self.st += float(tgt.sum())
        self.spp += float((pred * pred).sum()); self.stt += float((tgt * tgt).sum())
        self.spt += float((pred * tgt).sum()); self.cs += float(crps.sum())

    def result(self):
        n = self.n
        mp, mt = self.sp / n, self.st / n
        cov = self.spt / n - mp * mt
        vp, vt = self.spp / n - mp * mp, self.stt / n - mt * mt
        return {"rmse": (self.se / n) ** 0.5, "mae": self.ae / n, "bias": self.be / n,
                "corr": cov / ((vp * vt) ** 0.5), "crps": self.cs / n}


def load_all(ctx, dirs, labels, y, nd, land, er):
    """整年读场: 腐蚀域上的 crps / 误差 / SSIM 三分量; 同时在完整有效域上累计自检量。"""
    n_er = int(er.sum())
    truth = np.empty((nd, n_er), np.float32)
    A = {l: {k: np.empty((nd, n_er), np.float32) for k in ("crps", "err", "l", "c", "s")} for l in labels}
    acc = {l: FullAcc() for l in labels}
    t0 = time.time()
    for t in range(nd):
        tf = ctx.truth(y, t)
        if not np.isfinite(tf[land]).all():
            raise SystemExit(f"第 {t} 天真值在有效域内含非有限值")
        truth[t] = tf[er]
        for l in labels:
            d0 = dirs[l]
            em = np.load(d0 / "ens_mean" / f"{y}_d{t}.npy").astype(np.float64)
            cr = np.load(d0 / "crps" / f"{y}_d{t}.npy").astype(np.float64)
            if not (np.isfinite(em[land]).all() and np.isfinite(cr[land]).all()):
                raise SystemExit(f"{l} 第 {t} 天有效域内含非有限值")
            acc[l].add(em[land], tf[land], cr[land])
            lm, cm, sm = MT.ssim_components(em, tf, land)
            a = A[l]
            a["crps"][t] = cr[er]; a["err"][t] = (em - tf)[er]
            a["l"][t] = lm[er]; a["c"][t] = cm[er]; a["s"][t] = sm[er]
        if t % 60 == 0 or t == nd - 1:
            print(f"  day {t + 1}/{nd}  {time.time() - t0:.0f}s", flush=True)
    return truth, A, acc


def pool(truth, a, S, sel):
    """腐蚀域子集 sel 上的池化指标与 SSIM 三分量均值。"""
    n = 0
    se = ae = be = sp = st = spp = stt = spt = cs = ss = sl = sc = sst = 0.0
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
        ss += float(S[t][m].astype(np.float64).sum())
        sl += float(a["l"][t][m].astype(np.float64).sum())
        sc += float(a["c"][t][m].astype(np.float64).sum())
        sst += float(a["s"][t][m].astype(np.float64).sum())
    if n == 0:
        return None
    mp, mt = sp / n, st / n
    cov = spt / n - mp * mt
    vp, vt = spp / n - mp * mp, stt / n - mt * mt
    return {"mae": ae / n, "rmse": (se / n) ** 0.5, "bias": be / n,
            "corr": cov / ((vp * vt) ** 0.5) if vp > 0 and vt > 0 else float("nan"),
            "crps": cs / n, "ssim": ss / n,
            "ssim_l": sl / n, "ssim_c": sc / n, "ssim_s": sst / n, "n_px": int(n)}


def select(S, rule, frac):
    """按规则返回 (布尔掩膜, 阈值描述)。"""
    worst = "worst" in rule
    q = frac if worst else 1.0 - frac
    if rule.endswith("global"):
        thr = float(np.quantile(S.reshape(-1), q))
        return (S <= thr if worst else S >= thr), {"global_threshold": thr}
    thr = np.quantile(S, q, axis=1)                     # 逐日阈值
    sel = (S <= thr[:, None]) if worst else (S >= thr[:, None])
    return sel, {"daily_threshold_median": float(np.median(thr)),
                 "daily_threshold_p5": float(np.quantile(thr, 0.05)),
                 "daily_threshold_p95": float(np.quantile(thr, 0.95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--target", default=C.TARGETS[0], choices=C.TARGETS)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--rule", action="append", default=None, choices=RULES)
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--selfcheck-tol", type=float, default=3e-4)
    ap.add_argument("--figs", type=int, default=1)
    ap.add_argument("--cache", default=None,
                    help="逐像素数组的缓存★目录★; 每个数组一个裸 .npy, 读回时 mmap 不搬运数据。"
                         "存在则直接用(跳过读场), 否则算完写一份。"
                         "缓存记录模型标签、落场目录、年份、天数与腐蚀掩膜校验和, 任一不符即拒绝 —— "
                         "拿错缓存不会报错但结果全错")
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    labels = [l for l, _ in specs]
    dirs = dict(specs)
    rules = a.rule or list(RULES)
    y = a.year
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(specs[0][1], a.regions, a.target, out, years=(y,), model_tag="ssim-sets")
    land = ctx.land
    er = MT.eroded_land_mask(land)
    rid = ctx.region_id[er]
    nreg = len(ctx.region_names)
    nd = C.DAYS_PER_YEAR if not a.limit_days else min(a.limit_days, C.DAYS_PER_YEAR)
    month = ctx.month_of_day[y][:nd]
    order = np.argsort([int(i) for i in ctx.region_ids])
    rlab = [ctx.region_ids[i] for i in order]
    reg_px = np.bincount(rid, minlength=nreg + 1)[1:].astype(np.float64)
    print(f"[ssim] 有效域 {int(land.sum()):,} 格; 腐蚀 {MT.SSIM_ERODE}px 后 {int(er.sum()):,} 格 "
          f"(SSIM 的定义域); 像素日 {int(er.sum()) * nd:,}", flush=True)

    def stamp(d):
        """落场的内容指纹: 首末帧的字节数与修改时间。落场被原地重采样时目录名不变, 只有这个会变。"""
        parts = []
        for f in (d / "ens_mean" / f"{y}_d0.npy", d / "ens_mean" / f"{y}_d{nd - 1}.npy"):
            st = f.stat()
            parts.append(f"{st.st_size}@{int(st.st_mtime)}")
        return ",".join(parts)

    ident = {"labels": "|".join(labels), "dirs": "|".join(str(dirs[l].resolve()) for l in labels),
             "year": str(y), "nd": str(nd), "target": a.target,
             "er": f"{int(er.sum())}:{int(np.frombuffer(er.tobytes(), np.uint8).sum())}",
             "fields": "|".join(stamp(dirs[l]) for l in labels)}
    KEYS = ("crps", "err", "l", "c", "s")
    cache = Path(a.cache) if a.cache else None
    if cache is not None and (cache / "_ident.json").exists():
        got = json.load(open(cache / "_ident.json"))
        if got != ident:
            raise SystemExit(f"缓存身份不符, 拒绝使用:\n  缓存 {got}\n  本次 {ident}")
        # mmap: 打开时不搬运数据, 池化按需缺页; 缓存在内存文件系统上时等同直接访存
        truth = np.load(cache / "truth.npy", mmap_mode="r")
        A = {l: {k: np.load(cache / f"{slug(l)}__{k}.npy", mmap_mode="r") for k in KEYS} for l in labels}
        acc = None
        print(f"[ssim] 复用缓存目录 {cache} (mmap; 跳过读场与全集自检, 二者在写缓存时已做)", flush=True)
    else:
        truth, A, acc = load_all(ctx, dirs, labels, y, nd, land, er)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "truth.npy", truth)
            for l in labels:
                for k in KEYS:
                    np.save(cache / f"{slug(l)}__{k}.npy", A[l][k])
            json.dump(ident, open(cache / "_ident.json", "w"))       # 最后写, 半截的缓存不会被当成有效
            sz = sum(f.stat().st_size for f in cache.glob("*.npy"))
            print(f"[ssim] 缓存已写 {cache} ({sz/1e9:.1f} GB)", flush=True)

    Sf = {l: (np.asarray(A[l]["l"]) * np.asarray(A[l]["c"]) * np.asarray(A[l]["s"])).astype(np.float32)
          for l in labels}

    # ---- 自检: 完整有效域上的五个量 + 腐蚀域上的 SSIM, 对拍各落场 metrics.json ----
    checks = {}
    for l in labels:
        mp = dirs[l] / "metrics.json"
        if acc is None:
            checks[l] = "cached"; continue
        if not mp.exists() or nd != C.DAYS_PER_YEAR:
            checks[l] = "skipped"; continue
        off = json.load(open(mp))[ctx.unit]
        got = acc[l].result()
        got["ssim"] = float(Sf[l].mean(dtype=np.float64))
        bad = {k: (round(got[k], 4), off[k]) for k in ("rmse", "mae", "bias", "corr", "crps", "ssim")
               if k in off and abs(got[k] - off[k]) > a.selfcheck_tol}
        if bad:
            raise SystemExit(f"全集自检失败 {l}: {bad}")
        checks[l] = "ok"
    print(f"[ssim] 全集自检 {checks}", flush=True)

    full_er = np.ones((nd, rid.size), bool)
    res = {"models": labels, "target": a.target, "unit": ctx.unit, "year": y, "n_days": nd,
           "frac": a.frac, "domain": {"land": int(land.sum()), "land_eroded": int(er.sum()),
                                      "pixel_days": int(er.sum()) * nd},
           "selfcheck_vs_metrics_json": checks,
           "reference_full_eroded": {l: pool(truth, A[l], Sf[l], full_er) for l in labels},
           "caveats": ["选取与池化都在腐蚀 5px 的陆地上, 与落场 metrics.json 的完整有效域不同域, 不要混比",
                       "SSIM 是 11x11 窗口量, 集合空间上成块; 在按 SSIM 挑的集合上再报 SSIM 是同义反复",
                       "global 排序含日级 data_range 差异, daily 排序每天取同样份额剥掉它",
                       "corr 与 SSIM 不可跨子集比较; 最优集落在低方差区域时 corr 会被机械压低"],
           "sets": {}}

    freq = {}
    for own in labels:
        for rule in rules:
            sel, thr = select(Sf[own], rule, a.frac)
            rest = ~sel
            key = f"{slug(own)}__{rule}"
            res["sets"][key] = {
                "owner": own, "rule": rule, "share_px": float(sel.mean()), **thr,
                "selected": {l: pool(truth, A[l], Sf[l], sel) for l in labels},
                "rest": {l: pool(truth, A[l], Sf[l], rest) for l in labels},
                "composition": {
                    "by_region_share": {rlab[j]: float(np.bincount(rid, weights=sel.sum(0), minlength=nreg + 1)[1:][order[j]]
                                                       / (reg_px[order[j]] * nd)) for j in range(nreg)},
                    "by_month_share": {int(m): float(sel[month == m].mean()) for m in sorted(set(month.tolist()))}}}
            freq[key] = sel.sum(0).astype(np.int16)
            print(f"[ssim] {key} 完成", flush=True)
            del sel, rest

    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(out / "select_freq.npz", land=land, eroded=er, month=month,
                        region_names=np.array(ctx.region_names), **freq)

    if a.figs:
        for key, f in freq.items():
            g = np.full(land.shape, np.nan)
            g[er] = f
            grp = "select_freq_best" if "best" in key else "select_freq_worst"
            ctx.map(g, f"select_freq__{key}", cmap="magma", vmin=0,
                    vmax=float(np.quantile(f, 0.999)), scale_group=grp,
                    cbar="days in set", title=f"days in {a.frac*100:.0f}% set · {key.replace('__', ' · ')}")
        ctx.write_scales()

    mk = ("mae", "rmse", "bias", "corr", "crps", "ssim", "ssim_l", "ssim_c", "ssim_s")
    print("\n参考: 腐蚀域全集")
    print(f"  {'模型':<14}" + "".join(f"{k:>9}" for k in mk))
    for l in labels:
        v = res["reference_full_eroded"][l]
        print(f"  {l:<14}" + "".join(f"{v[k]:9.4f}" for k in mk))
    for key, b in res["sets"].items():
        print(f"\n=== {key}  选中 {b['share_px']*100:.1f}% 像素日 ===")
        for part in ("selected", "rest"):
            print(f"  -- {part} --  " + "".join(f"{k:>9}" for k in mk))
            for l in labels:
                v = b[part][l]
                print(f"  {l:<14}" + "".join(f"{v[k]:9.4f}" for k in mk))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
