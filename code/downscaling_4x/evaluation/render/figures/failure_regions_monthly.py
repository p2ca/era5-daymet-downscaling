# -*- coding: utf-8 -*-
"""各失败地区的逐月误差 —— 每个失败区一张单图, 12 个月的 mean CRPS; 叠阶段A MAE 作对照。

失败区名单由 --failure-regions 给定(与 failure_regions_map 同一份)。看每个区的误差集中在哪些月。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from downscaling_4x.evaluation.render.registry import figure


@figure("failure_regions_monthly", needs=("crps",), full_year=True)
def render(ctx):
    fr = ctx.failure_regions
    if not fr:
        print("[failure_regions_monthly] 未给 --failure-regions, 跳过")
        return []
    Mb, Kb = ctx.agg.mass_matrix("crps", "region")
    have_a = False
    try:
        ctx.ensure_mae_a()
        Ma, Ka = ctx.agg.mass_matrix("mae_a", "region")
        have_a = True
    except RuntimeError:
        pass
    idx = {n: i for i, n in enumerate(ctx.region_names)}

    arts = []
    for n in fr:
        if n not in idx:
            print(f"[failure_regions_monthly] 未知区名 {n}, 跳过")
            continue
        r = idx[n] + 1
        crps_m = Mb[r, 1:] / np.maximum(Kb[r, 1:], 1)
        fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
        x = np.arange(12)
        if have_a:
            mae_m = Ma[r, 1:] / np.maximum(Ka[r, 1:], 1)
            ax.bar(x - 0.2, crps_m, 0.4, color="#4878a8", label="Stage-B CRPS")
            ax.bar(x + 0.2, mae_m, 0.4, color="#9aa0a6", label="Stage-A MAE (ref)")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.bar(x, crps_m, 0.7, color="#4878a8")
        ax.set_xticks(x)
        ax.set_xticklabels(ctx.MONTHS, fontsize=8)
        ax.set_ylabel(f"mean CRPS [{ctx.unit}]")
        ax.set_xlabel("month")
        ax.set_title(f"monthly error — #{ctx.display_id(n)} {n}  ·  {ctx.target}", fontsize=10)
        arts.append(ctx.savefig(fig, f"failure_monthly_{ctx.display_id(n)}_{n}"))
    return arts
