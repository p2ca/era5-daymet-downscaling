# -*- coding: utf-8 -*-
"""全年平均 ens_mean 地图 + bias 地图(ens_mean − truth)。

ens_mean 用顺序色标; bias 用发散色标围绕 0。bias 需真值(--stats-dir), 缺则只出 ens_mean。
降水另有一份 log1p(mm) 空间的同构图: 物理空间的年均被少数强降水格点主导, log 空间才看得
清干湿分布的结构差异; 两个空间数值不可换算, 因此各自独立成图、独立色标。
"""
from downscaling_4x.evaluation.render.registry import figure


def _render(ctx, log):
    unit = "log1p(mm)" if log else ctx.unit
    sfx = "_log" if log else ""
    sp = "  [log1p(mm)]" if log else ""
    em = ctx.agg.annual_mean_log("ens_mean") if log else ctx.agg.annual_mean("ens_mean")
    sg = f"annual_field{sfx}_{ctx.target}"    # truth 与 ens_mean 共色标, 便于直接对比
    arts = [ctx.map(em, name=f"annual_{ctx.pred_name}{sfx}", cmap="turbo",
                    scale_group=sg, cbar=unit,
                    title=f"{ctx.model_label}  ·  annual mean {ctx.pred_name}  ·  {ctx.target}{sp}")]
    try:
        tr = ctx.annual_truth_log() if log else ctx.annual_truth()
        arts.append(ctx.map(tr, name=f"annual_truth{sfx}", cmap="turbo",
                            scale_group=sg, cbar=unit,
                            title=f"Daymet truth  ·  annual mean  ·  {ctx.target}{sp}"))
        arts.append(ctx.map(em - tr, name=f"annual_bias{sfx}", diverging=True, cmap="RdBu_r",
                            scale_group=f"annual_bias{sfx}_{ctx.target}",
                            cbar=f"bias [{unit}]",
                            title=f"{ctx.model_label}  ·  annual bias ({ctx.pred_name} − truth)  ·  {ctx.target}{sp}"))
    except RuntimeError as e:
        print(f"[annual_maps{sfx}] 跳过 truth/bias(缺真值): {e}")
    return arts


@figure("annual_maps", needs=("ens_mean",), full_year=True)
def render(ctx):
    """物理单位的年均场与 bias 地图。"""
    return _render(ctx, log=False)


@figure("annual_maps_log", needs=("ens_mean",), full_year=True)
def render_log(ctx):
    """log1p(mm) 空间的年均场与 bias 地图(仅降水)。"""
    if not ctx.is_precip:
        print("[annual_maps_log] 跳过: 只对降水有意义")
        return []
    return _render(ctx, log=True)
