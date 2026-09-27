# -*- coding: utf-8 -*-
"""错误贡献率排名(区域 / 月份)。

排名量 = mean CRPS = 误差质量 ÷ 格点数(区按像素数、月按天数归一), 故大面积区/长月不会
仅因体量靠前, 排的是真实的"每像素每天错多少"。每条标注: mean CRPS 值 + 该区/月占全域总
误差质量的份额%(供参考总量占比)。
降水另有一份 log1p(mm) 空间的排名: 物理空间的 CRPS 被强降水区主导, 排名可以与 log 空间
相反, 两份都给出才能判断"错在哪"是量级问题还是结构问题。
"""
import numpy as np

from downscaling_4x.evaluation.render.registry import figure


def _render(ctx, field, unit, sfx):
    M, K = ctx.agg.mass_matrix(field, level="region")
    Mr, Kr = M[1:, 1:], K[1:, 1:]
    tot = max(Mr.sum(), 1e-12)
    sp = f"  [{unit}]" if sfx else ""

    reg_mean = Mr.sum(1) / np.maximum(Kr.sum(1), 1)
    reg_share = Mr.sum(1) / tot * 100.0
    reg_ann = [f"{m:.3f} ({s:.1f}%)" for m, s in zip(reg_mean, reg_share)]

    mon_mean = Mr.sum(0) / np.maximum(Kr.sum(0), 1)
    mon_share = Mr.sum(0) / tot * 100.0
    mon_ann = [f"{m:.3f} ({s:.1f}%)" for m, s in zip(mon_mean, mon_share)]

    return [
        ctx.barh(reg_mean, ctx.region_ids, name=f"error_rank_region{sfx}",
                 xlabel=f"mean CRPS [{unit}]  (area-normalized)", annotations=reg_ann,
                 title=f"error rate ranking by region-id  ·  {ctx.target}{sp}"),
        ctx.barh(mon_mean, ctx.MONTHS, name=f"error_rank_month{sfx}",
                 xlabel=f"mean CRPS [{unit}]  (day-normalized)", annotations=mon_ann,
                 title=f"error rate ranking by month  ·  {ctx.target}{sp}"),
    ]


@figure("error_contrib_rank", needs=("crps",), full_year=True)
def render(ctx):
    """物理单位 CRPS 的区域/月份错误率排名。"""
    return _render(ctx, "crps", ctx.unit, "")


@figure("error_contrib_rank_log", needs=("crps_log",), full_year=True)
def render_log(ctx):
    """log1p(mm) 空间 CRPS 的区域/月份错误率排名(仅降水)。"""
    return _render(ctx, "crps_log", "log1p(mm)", "_log")
