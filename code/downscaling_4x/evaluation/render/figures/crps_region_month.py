# -*- coding: utf-8 -*-
"""CRPS 的 (分区 × 月) 热图 —— 每格为该区该月的平均 CRPS。"""
import numpy as np

from downscaling_4x.evaluation.render.registry import figure


@figure("crps_region_month", needs=("crps",), full_year=True)
def render(ctx):
    M, K = ctx.agg.mass_matrix("crps", level="region")
    mean = np.where(K[1:, 1:] > 0, M[1:, 1:] / np.maximum(K[1:, 1:], 1), np.nan)
    return ctx.heatmap(
        mean, ctx.region_ids, ctx.MONTHS, name="crps_region_month",
        cbar=f"CRPS [{ctx.unit}]", scale_group=f"crps_region_month_{ctx.target}", cmap="magma",
        title=f"CRPS  region-id × month  ·  {ctx.target}")
