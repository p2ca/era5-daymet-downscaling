#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_ablation.py — 通道敏感性实验出图(单图, 不拼图)
============================================================================
输入是 stage_a_ablation 落的 mae_<组>.npz(确定性 MAE)或 jit_ablation 落的
crps_<组>.npz(集合 CRPS), 两者都是逐月逐像素, 出三种图:

  groups   某区的分组敏感度横向条形 —— ΔMAE 降序, 条端标"相对该区基线 MAE 的放大倍数"
  months   某区的 组 x 月 热图 —— 值为 ΔMAE/当月基线 MAE, 归一后各月可比,
           季节结构一眼可读; 超出色标上限的格子标出真值
  map      某组的逐像素年均 ΔMAE 地图 —— 看效应落在哪里, 失败区勾边

ΔMAE = mae(组) − mae(none), 阶段 A 确定性, 无采样噪声, 因此不需要噪声地板。
年均按各月实际天数加权, 不是 12 个月简单平均。

★ 读图注意: 动态通道用 zero 模式置换时, 填的是全域全年的标量均值, 夏季离这个常数更远,
  因此动态组的 ΔMAE 天然在夏季更大 —— 那部分季节性是置换值造成的, 不是模型依赖的变化。
  静态通道(STA-Z/STA-S)没有这个问题, 它们的季节结构可以直接解读。
============================================================================
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import grid as G
from downscaling_4x.evaluation.ablation_groups import fname
from downscaling_4x.tools.plotting.mpl_style import use_cjk
from downscaling_4x.tools.preprocessing.region_display import display_id_map

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BASIC = ["SFC-T", "SFC-W", "SFC-V", "UPR-T", "UPR-Q", "UPR-D", "STA-Z", "STA-S"]
# 图上用中文名; 机理组代号与 ablation_groups 的对应关系见报告的"通道划分"一节。
# 逐通道消融的组名就是合同通道名, 同样映射为中文 —— 凡以通道/组为轴的图默认出中文标签,
# 数据文件与 npz 键仍用英文原名。
ZH = {"SFC-T": "近地面温度", "SFC-W": "近地面水分", "SFC-V": "近地面风",
      "UPR-T": "高空热力", "UPR-Q": "高空湿度", "UPR-D": "高空环流",
      "STA-Z": "亚网格地形", "STA-S": "地表属性",
      "ALL-UPR": "全部高空", "ALL-STA": "全部静态", "ALL": "全部通道",
      "2m_temperature": "2 米气温",
      "2m_temperature_max": "2 米最高气温",
      "2m_temperature_min": "2 米最低气温",
      "total_precipitation_24hr": "24 小时降水",
      "volumetric_soil_water_layer_1": "表层土壤湿度",
      "geopotential_500": "500 hPa 位势",
      "geopotential_850": "850 hPa 位势",
      "specific_humidity_500": "500 hPa 比湿",
      "specific_humidity_850": "850 hPa 比湿",
      "temperature_500": "500 hPa 气温",
      "temperature_850": "850 hPa 气温",
      "u_component_of_wind_500": "500 hPa 纬向风",
      "u_component_of_wind_850": "850 hPa 纬向风",
      "v_component_of_wind_500": "500 hPa 经向风",
      "v_component_of_wind_850": "850 hPa 经向风",
      "dz": "亚网格地形 Δz",
      "elevation": "绝对高程",
      "landcover": "地表覆盖",
      "land_sea_mask": "海陆掩膜",
      "doy_sin": "年内相位 sin",
      "doy_cos": "年内相位 cos"}
# 因果历史通道: "t_minus_<L>:<var>" -> "前 L 日·<var 中文名>"
ZH.update({C.history_name(lag, v): f"前 {lag} 日·{ZH[v]}"
           for lag in C.HISTORY_LAGS for v in C.ERA5_IN})


def load(d, metric="mae", space="phys"):
    meta = json.load(open(Path(d) / "meta.json"))
    groups = list(meta["groups"])
    key = "crps_log_month" if (metric == "crps" and space == "log") else f"{metric}_month"
    M = {g: np.load(Path(d) / f"{metric}_{fname(g)}.npz")[key].astype(np.float64)
         for g in groups}
    nd = np.load(Path(d) / f"{metric}_none.npz")["n_days"].astype(float)
    return meta, groups, M, nd


def main():
    ap = argparse.ArgumentParser(description="通道敏感性出图")
    ap.add_argument("--ablation", required=True, help="stage_a_ablation 输出目录")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--region", type=int, required=True, help="展示编号 1-19")
    ap.add_argument("--kind", nargs="+", default=["groups", "months", "map"],
                    choices=["groups", "months", "map"])
    ap.add_argument("--map-group", default="dz", help="出空间图的组/通道")
    ap.add_argument("--vmax-pct", type=float, default=100.0, help="月份热图色标上限(%%)")
    ap.add_argument("--metric", choices=["mae", "crps"], default="mae",
                    help="mae=确定性(mae_<组>.npz) / crps=集合方法(crps_<组>.npz)")
    ap.add_argument("--space", choices=["phys", "log"], default="phys",
                    help="log 仅对降水的 crps 有效: 取 crps_log_month")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    font = use_cjk()
    zh = (lambda g: ZH.get(g, g)) if font else (lambda g: g)
    meta, groups, M, nd = load(a.ablation, a.metric, a.space)
    unit = "log1p(mm)" if a.space == "log" else meta["unit"]
    # 阶段A 消融看回归均值的 MAE, 集合方法看 CRPS; 轴标签随之变, 免得两种图混着读
    MET = "ΔMAE(μ)" if a.metric == "mae" else "ΔCRPS"
    BASE = "基线 MAE" if a.metric == "mae" else "基线 CRPS"
    HEAD = "阶段 A 通道敏感度" if a.metric == "mae" else "通道敏感度"
    z = np.load(a.regions, allow_pickle=False)
    rid = z["region_id"].astype(int)
    names = [str(s) for s in z["region_names"]]
    disp = display_id_map(names)
    inv = {v: k for k, v in disp.items()}
    rname = inv[a.region]
    sel = (rid == names.index(rname) + 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = nd / nd.sum()
    tag = f"#{a.region} {rname}"

    def rmean(arr3, month=None):
        """该区的均值; month=None 时按天数加权成年均。"""
        if month is None:
            v = np.nansum(arr3 * w[:, None, None], axis=0)
        else:
            v = arr3[month]
        m = sel & np.isfinite(v)
        return float(v[m].mean())

    base_a = rmean(M["none"])
    base_m = np.array([rmean(M["none"], k) for k in range(12)])
    arts = []

    if "groups" in a.kind:
        gs = [g for g in groups if g != "none"]
        d = np.array([rmean(M[g]) - base_a for g in gs])
        o = np.argsort(d)[::-1]
        fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(gs) + 1.6), constrained_layout=True)
        ax.barh(range(len(gs)), d[o], color=["#b0453a" if gs[i] in BASIC else "#7b8794" for i in o])
        ax.set_yticks(range(len(gs)))
        ax.set_yticklabels([zh(gs[i]) for i in o], fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(d[o]):
            ax.text(v + 0.01 * d.max(), i, f"{v:.3f}  （基线的 {100*v/base_a:.0f}%）",
                    va="center", fontsize=7.6)
        ax.set_xlim(min(0, d.min() * 1.1), d.max() * 1.30)
        ax.set_xlabel(f"{MET} [{unit}]　（{BASE} = {base_a:.3f}）")
        ax.set_title(f"{HEAD} — {tag}  ·  {meta['target']}", fontsize=10)
        p = out / f"abl_groups_{a.region}_{rname}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); arts.append(p)

    if "months" in a.kind:
        gs = [g for g in BASIC if g in groups]
        if not gs:
            # 逐通道消融没有机理组名: 用全部通道, 按该区年均 Δ 降序排行
            gs = sorted((g for g in groups if g != "none"),
                        key=lambda g: rmean(M[g]) - base_a, reverse=True)
        Z = np.array([[(rmean(M[g], k) - base_m[k]) / base_m[k] * 100 for k in range(12)]
                      for g in gs])
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(gs) + 2.0), constrained_layout=True)
        im = ax.imshow(Z, aspect="auto", cmap="magma", vmin=0, vmax=a.vmax_pct)
        ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS, fontsize=8)
        ax.set_yticks(range(len(gs))); ax.set_yticklabels([zh(g) for g in gs], fontsize=9)
        for i in range(len(gs)):
            for j in range(12):
                v = Z[i, j]
                if v > a.vmax_pct or v < 0:            # 超出色标或为负的格子标真值
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                            color="#00e5ff" if v > a.vmax_pct else "#7CFC00")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
        cb.set_label(f"{MET} / 当月{BASE}  [%]")
        ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(range(12))
        ax2.set_xticklabels([f"{v:.2f}" for v in base_m], fontsize=6.2)
        ax2.set_xlabel(f"当月{BASE} [{unit}]", fontsize=8)
        ax.set_title(f"通道敏感度逐月分布 — {tag}  ·  {meta['target']}", fontsize=10)
        p = out / f"abl_months_{a.region}_{rname}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); arts.append(p)

    if "map" in a.kind:
        g = a.map_group
        dm = np.nansum((M[g] - M["none"]) * w[:, None, None], axis=0)
        ext = G.extent()
        asp = G.aspect()
        fin = dm[np.isfinite(dm)]
        vmax = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0
        fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
        im = ax.imshow(dm, origin="lower", extent=ext, aspect=asp, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        H, W = rid.shape
        lonc = np.linspace(ext[0], ext[1], W, endpoint=False) + (ext[1] - ext[0]) / W / 2
        latc = np.linspace(ext[2], ext[3], H, endpoint=False) + (ext[3] - ext[2]) / H / 2
        ax.contour(lonc, latc, sel.astype(float), levels=[0.5], colors="#101418", linewidths=1.4)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label(f"{MET} [{unit}]")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"年均 {MET} — 置换 {zh(g)}（{g}）  ·  {meta['target']}　（黑线 = {tag}）", fontsize=10)
        p = out / f"abl_map_{g}_{a.region}_{rname}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); arts.append(p)

    for p in arts:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
