#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_expert_by_ssim_set.py — 把专家 (区域 × 月) 热图拆到 SSIM 最差/最优/其余三个集合上
============================================================================
在 plot_expert_region_month 的 token 级聚合之上再加一维分组: 每个 (像素, 日) 先按该模型
自身的逐像素 SSIM 归入★最差 q★、★最优 q★或★其余★, 然后分别出 (区域 × 月) 热图。回答的是
"专家在结构最坏的地方是怎么开的, 与在结构最好的地方有什么不同"。

集合定义与 stage_b_ssim_sets 的 global 排序完全一致: 用同一份逐像素数组缓存重算 SSIM = l·c·s,
取全年统一分位阈值; 因此本工具的分组与那批 SSIM 集合实验逐格同一批像素。

★域的口径★: SSIM 只在腐蚀 5px 的陆地上有定义, 所以本工具的三个组之和是腐蚀域(200,535 格),
比有效域(219,069 格)少 8.5% —— 海岸带整条不在任何一组里。

每组各出: mean k / share k=0 / 参与专家数 1/Σp² / 门控权重总量 / 专家输出占比 / [D-EC]难度先验。
三组与三个模型共用同一把色标, 便于横向对读。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_expert_by_ssim_set \\
      --model DEC=runs/exp/<eval2020> --model TC=... --model EC=... \\
      --regions runs/exp/<regions>/regions_v1.npz --cache <逐像素数组缓存目录> \\
      --frac 0.01 --members 0 --out runs/exp/<diag>
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
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext

# 两个互不相干的二分: 最差 q 对其余 (1−q), 最优 q 对其余 (1−q)。
# "其余" 各自是对应集合的补集 —— 最差集的其余里含最优 q, 最优集的其余里含最差 q。
PARTS = ("worst", "worst_rest", "best", "best_rest")
# 三分区(最差 / 最优 / 中间)是两个二分的共同细化, 按它累加未归一的和后再合并即可。
BASE = ("worst", "best", "mid")
UNION = {"worst": ("worst",), "worst_rest": ("best", "mid"),
         "best": ("best",), "best_rest": ("worst", "mid")}
# 图题用英文: matplotlib 默认字体没有 CJK 字形, 中文会渲染成方块
PZH = {"worst": "SSIM-worst", "worst_rest": "complement of SSIM-worst",
       "best": "SSIM-best", "best_rest": "complement of SSIM-best"}


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


class TokenIndex:
    """按 offset 缓存的 "腐蚀域像素 -> token 下标"; 与 tok_to_pixel 的块映射同一套下标。"""

    def __init__(self, er, gh, gw, patch):
        self.Y, self.X = np.nonzero(er)
        self.gh, self.gw, self.patch, self.T = gh, gw, patch, gh * gw
        self._cache = {}

    def __call__(self, offset):
        key = (int(offset[0]), int(offset[1]))
        if key not in self._cache:
            dy, dx = key
            tok = ((self.Y + dy) // self.patch) * self.gw + ((self.X + dx) // self.patch)
            if tok.min() < 0 or tok.max() >= self.T:
                raise SystemExit(f"offset {key} 下 token 下标越界, 与落盘的 token_grid 不符")
            self._cache[key] = tok.astype(np.int64)
        return self._cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--cache", required=True, help="stage_b_ssim_sets 落的逐像素数组缓存目录")
    ap.add_argument("--cache-label", action="append", default=None,
                    help="模型标签=缓存里的标签(缺省同名); 用于两边命名不一致时")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--min-k", type=float, default=0.05)
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    cmap_lab = dict(t.split("=", 1) for t in (a.cache_label or []))   # 标签映射, 两边都是字符串
    cache = Path(a.cache)
    ident = json.load(open(cache / "_ident.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = out / "scales.json"
    res = {"year": a.year, "frac": a.frac, "members": a.members,
           "cache": str(cache), "cache_ident": ident, "models": {}}

    for label, dump in specs:
        info = RD.load_routing_meta(dump)
        if info is None:
            raise SystemExit(f"{dump} 下没有 routing/meta.json")
        gh, gw = info["token_grid"]
        patch, L, E = info["patch"], info["n_moe_layers"], info["n_experts"]
        items = [(y, d, m) for (y, d, m) in RD.available_routing(dump) if y == a.year
                 and m in a.members and (a.days is None or d in a.days)]
        if not items:
            raise SystemExit(f"{label}: 没有符合条件的路由文件")
        sub = out / slug(label); sub.mkdir(parents=True, exist_ok=True)
        ctx = RenderContext(dump, a.regions, C.TARGETS[0], sub, years=(a.year,),
                            model_tag=slug(label), scales_path=scales)
        land = ctx.land
        er = MT.eroded_land_mask(land)
        nreg = len(ctx.region_names)
        rid_er = ctx.region_id[er]
        order = np.argsort([int(i) for i in ctx.region_ids])
        rlab = [ctx.region_ids[i] for i in order]
        tokidx = TokenIndex(er, gh, gw, patch)

        # ---- SSIM 分组: 与 stage_b_ssim_sets 的 global 排序同一份数据、同一条阈值 ----
        cl = slug(cmap_lab.get(label, label))
        need = [cache / f"{cl}__{k}.npy" for k in ("l", "c", "s")]
        if not all(p.exists() for p in need):
            raise SystemExit(f"缓存里没有 {cl} 的 l/c/s: {need[0]}")
        Sf = (np.load(need[0], mmap_mode="r") * np.load(need[1], mmap_mode="r")
              * np.load(need[2], mmap_mode="r")).astype(np.float32)
        if Sf.shape[1] != int(er.sum()):
            raise SystemExit(f"缓存的像素数 {Sf.shape[1]} 与腐蚀域 {int(er.sum())} 不符")
        q_lo = float(np.quantile(Sf.reshape(-1), a.frac))
        q_hi = float(np.quantile(Sf.reshape(-1), 1.0 - a.frac))

        nd = C.DAYS_PER_YEAR
        # 累加★未归一的加权和★与★分母★, 分开存: 任意几个基础分区的并集 = 分子和 / 分母和
        num = {g: {k: np.zeros((nd, nreg)) for k in ("k", "k0", "gate", "norm", "prior")} for g in BASE}
        den = {g: np.zeros((nd, nreg)) for g in BASE}          # 像素数 × 层数(k/k0/gate/norm 按层与像素平均)
        den_p = {g: np.zeros((nd, nreg)) for g in BASE}        # 先验不含层维, 分母只是像素数
        sel = {g: np.zeros((nd, nreg, E)) for g in BASE}
        npx_tot = {g: 0 for g in BASE}
        t0 = time.time()
        for n, (y, d, m) in enumerate(items):
            r = RD.load_routing(dump, y, d, m)
            tok = tokidx(r["offset"])
            s_day = Sf[d]
            grp = np.where(s_day <= q_lo, 0, np.where(s_day >= q_hi, 1, 2))   # 0 worst 1 best 2 rest
            nf = float(r["nfwd"])
            k_t = r["k_sum"].sum(0) / nf
            k0_t = r["k0_cnt"].sum(0) / nf
            g_t = r["gate_sum"].sum((0, 2)) / nf
            nrm_t = r["norm_sum"].sum((0, 2)) / nf if "norm_sum" in r else None
            pr_t = r["prior_sum_t"].sum(0) / nf if "prior_sum_t" in r else None
            freq = r["sel_cnt"] / nf                                          # (L, T, E)
            fsum = freq.sum(0)                                                # (T, E)
            flat = (tok * nreg + (rid_er - 1)) * 3 + grp
            w = np.bincount(flat, minlength=tokidx.T * nreg * 3).reshape(tokidx.T, nreg, 3).astype(np.float64)
            for gi, g in enumerate(BASE):
                wg = w[:, :, gi]                                              # (T, nreg)
                npx = wg.sum(0)
                npx_tot[g] += int(npx.sum())
                num[g]["k"][d] += wg.T @ k_t
                num[g]["k0"][d] += wg.T @ k0_t
                num[g]["gate"][d] += wg.T @ g_t
                if nrm_t is not None:
                    num[g]["norm"][d] += wg.T @ nrm_t
                if pr_t is not None:
                    num[g]["prior"][d] += wg.T @ pr_t
                    den_p[g][d] += npx
                sel[g][d] += wg.T @ fsum
                den[g][d] += npx * L
            if (n + 1) % 60 == 0 or n + 1 == len(items):
                print(f"  [{label}] {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

        month = ctx.month_of_day[a.year][:nd]
        months = sorted(set(month.tolist()))
        mlab = [ctx.MONTHS[mm - 1] for mm in months]
        stats = {}
        save = {}
        for g in PARTS:
            parts = UNION[g]
            dsum = sum(den[b] for b in parts)
            dpsum = sum(den_p[b] for b in parts)
            ssum = sum(sel[b] for b in parts)
            cells = {k: np.where(dsum > 0, sum(num[b][k] for b in parts) / np.maximum(dsum, 1), np.nan)
                     for k in ("k", "k0", "gate", "norm")}
            cells["prior"] = np.where(dpsum > 0, sum(num[b]["prior"] for b in parts) / np.maximum(dpsum, 1), np.nan)
            share = ssum / np.maximum(ssum.sum(-1, keepdims=True), 1e-12)
            eff = np.where(cells["k"] >= a.min_k, 1.0 / np.maximum((share ** 2).sum(-1), 1e-12), np.nan)

            def rm(cell):
                o = np.full((nreg, len(months)), np.nan)
                for j, mm in enumerate(months):
                    v = cell[month == mm]
                    for i, ri in enumerate(order):
                        col = v[:, ri][np.isfinite(v[:, ri])]
                        if col.size:
                            o[i, j] = col.mean()
                return o

            sl = f"{slug(label)}_{g}"
            ttl = (f"{label} · {PZH[g]} {a.frac*100:g}%" if g in ("worst", "best")
                   else f"{label} · {PZH[g]} {100 - a.frac*100:g}%")
            ctx.heatmap(rm(cells["k"]), rlab, mlab, f"k_region_month_{sl}", cbar="mean routed experts",
                        scale_group="k_rm", cmap="magma", title=f"{ttl} · mean routed experts · region × month")
            ctx.heatmap(rm(cells["k0"]), rlab, mlab, f"k0_region_month_{sl}", cbar="share k=0",
                        scale_group="k0_rm", cmap="magma_r", vmin=0, vmax=1, title=f"{ttl} · share of k=0")
            ctx.heatmap(rm(eff), rlab, mlab, f"participating_experts_region_month_{sl}",
                        cbar="participating experts (1/Σp²)", scale_group="eff_rm", cmap="viridis",
                        title=f"{ttl} · participating experts · region × month")
            ctx.heatmap(rm(cells["gate"]), rlab, mlab, f"gate_total_region_month_{sl}",
                        cbar="gate weight per forward", scale_group="gate_rm", cmap="viridis",
                        title=f"{ttl} · total gate weight")
            if np.isfinite(cells["norm"]).any():
                ctx.heatmap(rm(cells["norm"]), rlab, mlab, f"expert_output_ratio_region_month_{sl}",
                            cbar="Σ‖w·f_e(x)‖/‖x‖", scale_group="norm_rm", cmap="viridis",
                            title=f"{ttl} · expert output / token norm")
            if np.isfinite(cells["prior"]).any():
                ctx.heatmap(rm(cells["prior"]), rlab, mlab, f"prior_region_month_{sl}", cbar="difficulty prior",
                            scale_group="prior_rm", cmap="viridis", title=f"{ttl} · difficulty prior")
            stats[g] = {"mean_k": float(np.nanmean(cells["k"])), "mean_k0": float(np.nanmean(cells["k0"])),
                        "mean_participating": float(np.nanmean(eff)), "mean_gate": float(np.nanmean(cells["gate"])),
                        "mean_norm": float(np.nanmean(cells["norm"])), "mean_prior": float(np.nanmean(cells["prior"])),
                        "pixel_days": sum(npx_tot[b] for b in parts)}
            for k, v in cells.items():
                save[f"{g}_cells_{k}"] = v
            save[f"{g}_eff"] = eff
            save[f"{g}_share"] = share

        np.savez_compressed(sub / "cells_by_ssim_set.npz", month=month,
                            region_display_ids=np.array(ctx.region_ids),
                            region_names=np.array(ctx.region_names), **save)
        ctx.write_scales()          # 每个模型画完就落盘: 下一个模型的 RenderContext 才能读到同一把色标
        res["models"][label] = {"dump": str(dump), "n_files": len(items), "layers": L, "experts": E,
                                "ssim_threshold_low": q_lo, "ssim_threshold_high": q_hi,
                                "seconds": round(time.time() - t0, 1), "by_group": stats}
        print(f"[{label}] 阈值 low {q_lo:.4f} high {q_hi:.4f} | "
              + " | ".join(f"{g} k={stats[g]['mean_k']:.3f} eff={stats[g]['mean_participating']:.2f}"
                           for g in PARTS), flush=True)

    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
