#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
plot_progress_summary.py — 给微信汇报用的一张方法对比图

设计要点(见 dataviz 规范):
  * 三个变量单位不同(K / K / m·day⁻¹) -> ★绝不共用坐标轴★, 用小倍数三面板, 各自标度
  * 颜色只承担"方法族"的身份: 插值=chrome 灰(参照物, 非系列), BCSD=黄, UNet=蓝, ViT=紫
    三个主角色经 validate_palette.js 验证: 最差相邻色盲 ΔE 16.6 (>=12 门槛), ALL PASS
    黄色对比度 2.11 触发 relief 规则 -> 每根柱子都有可见直标(方法名 + 数值), 已满足
  * 每根柱子直标"相对 bilinear 的改善率" -> 这是让"温度赢/降水败"反差一眼可见的关键
  * 降水面板 BCSD 缺席: 它在 log1p(mm) 空间评测, 与其余方法(m/day)★不可通约★,
    强行并排 = 假结论。图中显式标注缺口, 不伪造数据。

数据来源: runs/LEDGER.md (由 runs/exp/*/meta.json 自动汇总)
用法: python code/plot_progress_summary.py [--out <path.png>]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from era5_daymet.paths import PROJECT_ROOT

FONT = os.path.expanduser("~/.fonts/NotoSansSC-Regular.otf")
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# --- 调色 (dataviz reference palette, light mode) ---
GREY   = "#898781"   # chrome/muted: 插值基线 = 参照物, 不是分类系列
YELLOW = "#eda100"   # BCSD  (统计)
BLUE   = "#2a78d6"   # UNet  (深度学习)
VIOLET = "#4a3aa7"   # ViT   (深度学习, 主角)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

# 方法顺序固定(自上而下), 颜色跟随实体, 三个面板一致
METHODS = ["bilinear", "bicubic", "BCSD", "UNet", "ViT"]
COLOR   = {"bilinear": GREY, "bicubic": GREY, "BCSD": YELLOW, "UNet": BLUE, "ViT": VIOLET}

# RMSE, 全部来自 runs/LEDGER.md (2020 全年 365 天, 陆地掩膜, 整帧评测)
# None = 该方法在此单位空间下无可比数据
PANELS = [
    dict(title="日最高气温 tmax", unit="RMSE  (K)",
         vals={"bilinear": 2.142, "bicubic": 2.122, "BCSD": 1.555, "UNet": 1.703, "ViT": 1.564}),
    dict(title="日最低气温 tmin", unit="RMSE  (K)",
         vals={"bilinear": 3.017, "bicubic": 3.049, "BCSD": 2.291, "UNet": 2.243, "ViT": 2.004}),
    dict(title="日降水", unit="RMSE  (mm/day)",
         # 原始单位 m/day, ×1000 换成 mm/day 便于阅读(线性换算, 不改变任何相对关系)
         vals={"bilinear": 4.6, "bicubic": 4.6, "BCSD": None, "UNet": 4.5, "ViT": 4.7}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(PROJECT_ROOT / "runs/progress_summary.png"))
    a = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.16, wspace=0.42)

    for ax, P in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        base = P["vals"]["bilinear"]                       # 改善率的参照
        ys = list(range(len(METHODS)))[::-1]               # 自上而下

        for y, m in zip(ys, METHODS):
            v = P["vals"][m]
            if v is None:                                  # BCSD 在降水上不可通约 -> 显式留白
                ax.text(0.02, y, "单位空间不同，不可比",
                        va="center", ha="left", fontsize=10.5, color=MUTED, style="italic",
                        transform=ax.get_yaxis_transform())
                continue
            ax.barh(y, v, height=0.62, color=COLOR[m], zorder=3,
                    edgecolor=SURFACE, linewidth=2)        # 2px surface gap between fills

            imp = (base - v) / base * 100                  # 相对 bilinear 的改善(正=更好)
            if m == "bilinear":
                lab = f"{v:g}    (基线)"
                col = INK2
            else:
                sign = "+" if imp > 0 else ""
                lab = f"{v:g}    {sign}{imp:.0f}%"
                col = "#006300" if imp > 5 else ("#c0392b" if imp < -0.5 else INK2)
            ax.text(v * 1.02, y, lab, va="center", ha="left",
                    fontsize=11.5, color=col, fontweight="bold" if abs(imp) > 5 else "normal")

        ax.set_yticks(ys)
        ax.set_yticklabels(METHODS, fontsize=12, color=INK)
        ax.set_xlim(0, max(v for v in P["vals"].values() if v) * 1.55)   # 留够直标空间, 防右缘裁切
        ax.set_title(P["title"], fontsize=14, color=INK, pad=11, fontweight="bold", loc="left")
        ax.set_xlabel(P["unit"] + "   ← 越低越好", fontsize=10.5, color=MUTED, labelpad=7)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")

    fig.text(0.075, 0.945, "ERA5 → Daymet 降尺度 (6×, CONUS)：温度上 ML 赢了，降水上全线撞墙",
             fontsize=17.5, color=INK, fontweight="bold", ha="left")
    fig.text(0.075, 0.885,
             "ViT 在 tmin 上做到 2.004 K，是唯一击穿 2.1 的方法，比统计基线 BCSD 好 12.5%；"
             "但同一个 ViT 在降水上反而不如 UNet，甚至不如双线性插值。",
             fontsize=12, color=INK2, ha="left")
    fig.text(0.075, 0.835,
             "百分比 = 相对 bilinear 的 RMSE 改善。测试集 = 2020 全年 365 天，陆地掩膜，整帧评测。"
             "括注：ViT 15.5M 参数 vs UNet 1.6M，规模差 10×，故不足以论证「架构更优」。",
             fontsize=10.5, color=MUTED, ha="left")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
