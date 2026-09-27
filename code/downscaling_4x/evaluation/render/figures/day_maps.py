# -*- coding: utf-8 -*-
"""指定日期的预测场与 bias 场(单日图)。

日期由 --days 显式给出, 不设缺省: 缺省日会让不同模型的单日图落在不同天气上, 而图面看
不出这件事。年均图回答"系统性偏差在哪", 单日图回答"某一天的天气结构还原得怎样", 两者
互不替代。

同一天的预测与真值共用一套色标(scale_group 含日期), bias 另成一组; 跨模型共色标由
--scales 指向同一份 scales.json 实现。降水另出一份 log1p(mm) 空间的同构图。
"""
import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.evaluation.metrics import precip_log_mm
from downscaling_4x.evaluation.render.registry import figure


def _one_day(ctx, y, day, log):
    unit = "log1p(mm)" if log else ctx.unit
    sfx = "_log" if log else ""
    sp = "  [log1p(mm)]" if log else ""
    # 落场已是 mm/day, 这里只为展示做 log1p 压缩, 不再乘 precip_scale
    lg = precip_log_mm if log else (lambda a: a)
    tag = f"{y}_d{day}"
    arts = []

    pred = ctx.load_field("ens_mean", y, day)
    if pred is None:
        print(f"[day_maps{sfx}] 跳过 {tag}: 缺 ens_mean")
        return arts
    pred = lg(pred)
    sg = f"day_field{sfx}_{ctx.target}"          # 同一目标的各日各方法共一把尺子
    arts.append(ctx.map(pred, name=f"day{sfx}_{tag}_{ctx.pred_name}", cmap="turbo",
                        scale_group=sg, cbar=unit,
                        title=f"{ctx.model_label}  ·  {tag}  {ctx.pred_name}  ·  {ctx.target}{sp}"))
    try:
        tr = lg(ctx.truth(y, day))
    except Exception as e:
        print(f"[day_maps{sfx}] {tag}: 缺真值, 只出预测场 ({e})")
        return arts
    arts.append(ctx.map(pred - tr, name=f"day{sfx}_{tag}_bias", diverging=True, cmap="RdBu_r",
                        scale_group=f"day_bias{sfx}_{ctx.target}", cbar=f"bias [{unit}]",
                        title=f"{ctx.model_label}  ·  {tag}  bias ({ctx.pred_name} − truth)  ·  {ctx.target}{sp}"))
    return arts


def _render(ctx, log):
    days = list(getattr(ctx, "days", ()) or ())
    if not days:
        print("[day_maps] 跳过: 未指定 --days")
        return []
    arts = []
    for y in ctx.years:
        for day in days:
            arts += _one_day(ctx, y, day, log)
    return arts


@figure("day_maps", needs=("ens_mean",))
def render(ctx):
    """物理单位的单日预测场与 bias 场(--days 指定)。"""
    return _render(ctx, log=False)


@figure("day_maps_log", needs=("ens_mean",))
def render_log(ctx):
    """log1p(mm) 空间的单日预测场与 bias 场(仅降水)。"""
    if not ctx.is_precip:
        print("[day_maps_log] 跳过: 只对降水有意义")
        return []
    return _render(ctx, log=True)
