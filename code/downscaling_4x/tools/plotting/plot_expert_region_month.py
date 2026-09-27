#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_expert_region_month.py — 路由落盘 -> (区域 × 月) 专家热图, token 级聚合
============================================================================
与 plot_routing 的 (区域 × 月) 部分算同一批量, 但★不把 token 展开成像素★。

plot_routing 每读一个路由文件都要把 (层, 专家, token) 铺成 (层, 专家, 像素), 中间数组
372 MB、每文件调五次; 而区域求和根本不需要这一步: 每个 token 覆盖一个 patch×patch 的块,
它在各区域内占多少个有效格点只取决于该轨迹的切块起点 offset, 而 offset 至多 patch² 种。
把"token → 区域"的格点计数矩阵按 offset 缓存一次, 区域求和就是一次 (T × nreg) 的矩阵乘。
两者数学上等价(同一组求和换了结合顺序), 不是近似; tests 之外另有 --verify 直接对拍。

出图(全部单图, 同组共色标):
  mean k              (区域 × 月) 每 token 平均路由到的专家数
  share k=0           (区域 × 月) 一次前向里 0 个路由专家的占比
  participating       (区域 × 月) 参与专家数 1/Σp² —— 只说明多少专家在分摊流量, 不说明有没有用
  gate total          (区域 × 月) 每次前向施加的门控权重总量
  expert/token norm   (区域 × 月) 专家输出占 token 隐状态的比例
  prior               [D-EC] (区域 × 月) 难度头先验

逐像素地图不在本工具范围内(那才真的需要展开), 要地图仍用 plot_routing。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.plotting.plot_expert_region_month \\
      --model DEC=runs/exp/<eval2020> --model TC=... --model EC=... \\
      --regions runs/exp/<regions>/regions_v1.npz --out runs/exp/<diag> [--members 0]
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
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.evaluation.render.context import RenderContext


def parse_spec(s):
    if "=" not in s:
        raise SystemExit(f"需要 label=dir 形式, 得到 {s!r}")
    lab, d = s.split("=", 1)
    return lab.strip(), Path(d)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


class TokenRegionWeights:
    """按 offset 缓存的 (token, 区域) 有效格点计数; 与 tok_to_pixel 的块映射同一套下标。"""

    def __init__(self, land, rid_land, gh, gw, patch, nreg):
        self.land, self.rid, self.gh, self.gw = land, rid_land, gh, gw
        self.patch, self.nreg, self.T = patch, nreg, gh * gw
        H, W = land.shape
        self.Y, self.X = np.nonzero(land)          # 有效域像素的行列
        self._cache = {}

    def __call__(self, offset):
        key = (int(offset[0]), int(offset[1]))
        if key not in self._cache:
            dy, dx = key
            # tok_to_pixel: 像素(y,x) 取自 padded 网格的 (y+dy, x+dx), 属于 token ((y+dy)//P, (x+dx)//P)
            tok = ((self.Y + dy) // self.patch) * self.gw + ((self.X + dx) // self.patch)
            if tok.min() < 0 or tok.max() >= self.T:
                raise SystemExit(f"offset {key} 下 token 下标越界, 与落盘的 token_grid 不符")
            flat = tok * self.nreg + (self.rid - 1)
            w = np.bincount(flat, minlength=self.T * self.nreg).reshape(self.T, self.nreg)
            self._cache[key] = w.astype(np.float64)
        return self._cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=eval_dir, 可多次")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, nargs="+", default=None, help="只用这些成员(缺省全部已落盘成员)")
    ap.add_argument("--days", type=int, nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--min-k", type=float, default=0.05, help="参与专家数的门槛: 格内平均专家数低于此值置空")
    ap.add_argument("--verify", type=int, default=0, help=">0 时同时用 plot_routing 的像素路径算前 N 天并逐格对拍")
    a = ap.parse_args()

    specs = [parse_spec(s) for s in a.model]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = out / "scales.json"
    res = {"year": a.year, "members": a.members, "models": {}}

    for label, dump in specs:
        info = RD.load_routing_meta(dump)
        if info is None:
            raise SystemExit(f"{dump} 下没有 routing/meta.json")
        gh, gw = info["token_grid"]
        patch, L, E = info["patch"], info["n_moe_layers"], info["n_experts"]
        items = [(y, d, m) for (y, d, m) in RD.available_routing(dump) if y == a.year
                 and (a.members is None or m in a.members) and (a.days is None or d in a.days)]
        if not items:
            raise SystemExit(f"{label}: 没有符合条件的路由文件")
        sub = out / slug(label); sub.mkdir(parents=True, exist_ok=True)
        ctx = RenderContext(dump, a.regions, C.TARGETS[0], sub, years=(a.year,),
                            model_tag=slug(label), scales_path=scales)
        land = ctx.land
        nreg = len(ctx.region_names)
        rid_land = ctx.region_id[land]
        order = np.argsort([int(i) for i in ctx.region_ids])
        rlab = [ctx.region_ids[i] for i in order]
        wmap = TokenRegionWeights(land, rid_land, gh, gw, patch, nreg)

        nd = C.DAYS_PER_YEAR
        acc = {k: np.zeros((nd, nreg)) for k in ("k", "k0", "gate", "norm", "prior")}
        cnt = np.zeros((nd, nreg))
        cnt_prior = np.zeros((nd, nreg))
        sel = np.zeros((nd, nreg, E))
        reg_exp = np.zeros((L, nreg, E))
        t0 = time.time()
        for n, (y, d, m) in enumerate(items):
            r = RD.load_routing(dump, y, d, m)
            w = wmap(r["offset"])                       # (T, nreg) 有效格点计数
            npx = w.sum(0)                              # 每区域的有效格点数
            good = npx > 0
            nf = float(r["nfwd"])
            # 逐层求和后一次投影: 与 "先铺成像素、再对层与像素取平均" 逐项相同
            k_t = (r["k_sum"].sum(0) / nf)              # (T,)
            k0_t = (r["k0_cnt"].sum(0) / nf)
            g_t = (r["gate_sum"].sum((0, 2)) / nf)
            acc["k"][d] += np.where(good, (w.T @ k_t) / np.maximum(npx * L, 1), 0.0)
            acc["k0"][d] += np.where(good, (w.T @ k0_t) / np.maximum(npx * L, 1), 0.0)
            acc["gate"][d] += np.where(good, (w.T @ g_t) / np.maximum(npx * L, 1), 0.0)
            if "norm_sum" in r:
                nrm_t = (r["norm_sum"].sum((0, 2)) / nf)
                acc["norm"][d] += np.where(good, (w.T @ nrm_t) / np.maximum(npx * L, 1), 0.0)
            if "prior_sum_t" in r:
                pr_t = r["prior_sum_t"].sum(0) / nf
                acc["prior"][d] += np.where(good, (w.T @ pr_t) / np.maximum(npx, 1), 0.0)
                cnt_prior[d] += good
            freq = r["sel_cnt"] / nf                    # (L, T, E)
            sel[d] += w.T @ freq.sum(0)                 # (nreg, E)
            reg_exp += np.einsum("tr,lte->lre", w, freq)
            cnt[d] += good
            if (n + 1) % 100 == 0 or n + 1 == len(items):
                print(f"  [{label}] {n + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

        cells = {k: np.where(cnt > 0, v / np.maximum(cnt, 1), np.nan) for k, v in acc.items()}
        cells["prior"] = np.where(cnt_prior > 0, acc["prior"] / np.maximum(cnt_prior, 1), np.nan)
        share = sel / np.maximum(sel.sum(-1, keepdims=True), 1e-12)
        eff = 1.0 / np.maximum((share ** 2).sum(-1), 1e-12)
        eff = np.where(cells["k"] >= a.min_k, eff, np.nan)

        month = ctx.month_of_day[a.year][:nd]
        months = sorted(set(month.tolist()))
        mlab = [ctx.MONTHS[mm - 1] for mm in months]

        def rm(cell):
            out_ = np.full((nreg, len(months)), np.nan)
            for j, mm in enumerate(months):
                v = cell[month == mm]
                for i, ri in enumerate(order):
                    col = v[:, ri]
                    col = col[np.isfinite(col)]
                    if col.size:
                        out_[i, j] = col.mean()
            return out_

        sl = slug(label)
        ctx.heatmap(rm(cells["k"]), rlab, mlab, f"k_region_month_{sl}", cbar="mean routed experts",
                    scale_group="k_rm", cmap="magma", title=f"{label} · mean routed experts per token · region × month")
        ctx.heatmap(rm(cells["k0"]), rlab, mlab, f"k0_region_month_{sl}", cbar="share k=0",
                    scale_group="k0_rm", cmap="magma_r", vmin=0, vmax=1, title=f"{label} · share of k=0 · region × month")
        ctx.heatmap(rm(eff), rlab, mlab, f"participating_experts_region_month_{sl}",
                    cbar="participating experts (1/Σp²)", scale_group="eff_rm", cmap="viridis",
                    title=f"{label} · participating experts · region × month")
        ctx.heatmap(rm(cells["gate"]), rlab, mlab, f"gate_total_region_month_{sl}", cbar="gate weight per forward",
                    scale_group="gate_rm", cmap="viridis", title=f"{label} · total gate weight · region × month")
        if np.isfinite(cells["norm"]).any():
            ctx.heatmap(rm(cells["norm"]), rlab, mlab, f"expert_output_ratio_region_month_{sl}",
                        cbar="Σ‖w·f_e(x)‖/‖x‖ per forward", scale_group="norm_rm", cmap="viridis",
                        title=f"{label} · expert output / token norm · region × month")
        if np.isfinite(cells["prior"]).any():
            ctx.heatmap(rm(cells["prior"]), rlab, mlab, f"prior_region_month_{sl}", cbar="difficulty prior",
                        scale_group="prior_rm", cmap="viridis", title=f"{label} · difficulty prior · region × month")

        np.savez_compressed(sub / "cells.npz", month=month, region_display_ids=np.array(ctx.region_ids),
                            region_names=np.array(ctx.region_names), share=share, reg_exp=reg_exp,
                            eff=eff, **{f"cells_{k}": v for k, v in cells.items()})
        res["models"][label] = {"dump": str(dump), "n_files": len(items), "n_days": len(set(d for _, d, _ in items)),
                                "layers": L, "experts": E, "token_grid": [gh, gw], "patch": patch,
                                "seconds": round(time.time() - t0, 1),
                                "mean_k_overall": float(np.nanmean(cells["k"])),
                                "mean_participating_overall": float(np.nanmean(eff))}
        print(f"[{label}] 完成 {len(items)} 文件 {time.time() - t0:.0f}s  平均 k {np.nanmean(cells['k']):.3f}  "
              f"参与专家数 {np.nanmean(eff):.2f}", flush=True)

        if a.verify:
            days = sorted(set(d for _, d, _ in items))[:a.verify]
            bad = verify_against_pixel_path(dump, items, days, land, rid_land, nreg, gh, gw, patch, L, cells)
            res["models"][label]["verify"] = bad if bad else f"前 {a.verify} 天逐格与像素路径一致"
            print(f"[{label}] 对拍: {res['models'][label]['verify']}", flush=True)

    ctx.write_scales()
    json.dump(res, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
    print(f"-> {out}")


def verify_against_pixel_path(dump, items, days, land, rid_land, nreg, gh, gw, patch, L, cells):
    """对同样几天用 plot_routing 的像素展开路径重算 (日, 区域) 的 k, 逐格比较。"""
    H, W = land.shape
    ref = {}
    for (y, d, m) in items:
        if d not in days:
            continue
        r = RD.load_routing(dump, y, d, m)
        off = tuple(int(v) for v in r["offset"])
        k_px = RD.tok_to_pixel(r["k_sum"] / float(r["nfwd"]), gh, gw, patch, off, H, W)[..., land]
        row = ref.setdefault(d, [np.zeros(nreg), 0])
        for rr in range(nreg):
            mm = rid_land == rr + 1
            if mm.any():
                row[0][rr] += k_px[:, mm].mean()
        row[1] += 1
    bad = []
    for d, (s, c) in ref.items():
        got, want = cells["k"][d], s / c
        e = np.nanmax(np.abs(got - want))
        if e > 1e-9:
            bad.append(f"day {d}: 最大差 {e:.3e}")
    return bad


if __name__ == "__main__":
    main()
