#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单日真值场(--days 指定): 与各模型的单日预测图共用同一把色标尺子。

真值与模型无关, 但要和预测图落在同一色标上才能并排看, 所以走同一个 scale_group。
"""
from downscaling_4x.evaluation.render.registry import figure


def _render(ctx, log):
    days = list(getattr(ctx, "days", ()) or ())
    if not days:
        print("[day_truth] 跳过: 未指定 --days")
        return []
    sfx = "_log" if log else ""
    unit = "log1p(mm)" if log else ctx.unit
    lg = ctx.to_log if log else (lambda x: x)
    arts = []
    for y in ctx.years:
        for day in days:
            tag = f"{y}_d{day}"
            try:
                tr = lg(ctx.truth(y, day))
            except Exception as e:
                print(f"[day_truth{sfx}] 跳过 {tag}: 取不到真值 ({e})")
                continue
            arts.append(ctx.map(tr, name=f"day{sfx}_{tag}_truth", cmap="turbo",
                                scale_group=f"day_field{sfx}_{ctx.target}", cbar=unit,
                                title=f"Daymet truth  ·  {tag}  ·  {ctx.target}"
                                      f"{'  [log1p(mm)]' if log else ''}"))
    return arts


@figure("day_truth", needs=())
def render(ctx):
    """物理单位的单日真值场(--days 指定); 与单日预测图同一色标。"""
    return _render(ctx, log=False)


@figure("day_truth_log", needs=())
def render_log(ctx):
    """log1p(mm) 空间的单日真值场(仅降水)。"""
    if not ctx.is_precip:
        return []
    return _render(ctx, log=True)
