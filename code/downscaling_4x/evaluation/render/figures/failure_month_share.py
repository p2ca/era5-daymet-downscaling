# -*- coding: utf-8 -*-
"""失败区的逐月错误贡献率 —— 每区一张单图。

贡献率 = 该月的误差质量 ÷ 该区全年误差质量, 12 个月加起来是 100%, 回答"这个区的错
主要发生在什么时候"。因为是质量占比而不是均值, 月长自然计入其中。

条上标注该月的 mean CRPS(误差质量 ÷ 格点数), 便于同时读出"错得多"与"错得频"——
一个月份可以贡献率高只因为它错得频, 也可以只因为单日错得狠。

横轴按日历月排列而不是降序, 这样季节结构可以直接看出来。降水另有一份 log1p(mm)
空间的同构图: 物理空间的贡献率被强降水月主导, 两个空间的季节结构可以不同。
"""
import numpy as np

from downscaling_4x.evaluation.render.registry import figure


def _render(ctx, field, unit, sfx):
    fr = ctx.failure_regions
    if not fr:
        print(f"[failure_month_share{sfx}] 未给 --failure-regions, 跳过")
        return []
    M, K = ctx.agg.mass_matrix(field, "region")
    idx = {n: i for i, n in enumerate(ctx.region_names)}
    sp = f"  [{unit}]" if sfx else ""

    arts = []
    for n in fr:
        if n not in idx:
            print(f"[failure_month_share{sfx}] 未知区名 {n}, 跳过")
            continue
        r = idx[n] + 1
        mass = M[r, 1:].astype(float)
        cnt = K[r, 1:].astype(float)
        share = mass / max(mass.sum(), 1e-12) * 100.0
        mean = mass / np.maximum(cnt, 1)
        did = ctx.display_id(n)
        arts.append(ctx.bar(
            share, ctx.MONTHS, name=f"failure_month_share{sfx}_{did}_{n}",
            ylabel="share of annual error mass [%]", ref=100.0 / 12,
            annotations=[f"{v:.3f}" for v in mean],
            title=f"monthly error share — #{did} {n}  ·  {ctx.target}{sp}"
                  f"   (bar label = mean CRPS [{unit}])"))
    return arts


@figure("failure_month_share", needs=("crps",), full_year=True)
def render(ctx):
    """物理单位下失败区的逐月错误贡献率。"""
    return _render(ctx, "crps", ctx.unit, "")


@figure("failure_month_share_log", needs=("crps_log",), full_year=True)
def render_log(ctx):
    """log1p(mm) 空间下失败区的逐月错误贡献率(仅降水)。"""
    return _render(ctx, "crps_log", "log1p(mm)", "_log")
