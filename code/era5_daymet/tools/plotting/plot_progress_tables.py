#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
plot_progress_tables.py — 评估指标表格 PNG (分享进展用)

列位置显式锚定 -> 比例字体下数字也严格对齐(微信里用空格对齐必然错位)。
降水给出两个空间, ★同一份 BCSD 在两个空间里的排名是反的★ —— 见 AGENTS.md。

数据: runs/LEDGER.md + runs/exp/20260713-bcsd-precip-both-spaces/metrics_both_spaces.json
用法: python code/plot_progress_tables.py [--out runs/progress_tables.png]
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

INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#8a8880", "#d8d7d0", "#ffffff"
HL = "#e8f2fd"          # 该表最优行

HEAD = ["method", "RMSE", "MAE", "bias", "corr"]

TABLES = [
    ("tmax  (K)", None,
     [["nearest",  "2.290", "1.501", "-0.395", "0.981"],
      ["bilinear", "2.142", "1.453", "-0.388", "0.984"],
      ["bicubic",  "2.122", "1.442", "-0.384", "0.984"],
      ["BCSD",     "1.555", "1.117", "+0.350", "0.991"],
      ["UNet",     "1.703", "1.255", "-0.069", "0.989"],
      ["ViT",      "1.564", "1.156", "+0.334", "0.992"]], [3]),

    ("tmin  (K)", None,
     [["nearest",  "3.145", "2.379", "+1.348", "0.963"],
      ["bilinear", "3.017", "2.319", "+1.344", "0.967"],
      ["bicubic",  "3.049", "2.335", "+1.336", "0.966"],
      ["BCSD",     "2.291", "1.649", "-0.109", "0.975"],
      ["UNet",     "2.243", "1.682", "-0.211", "0.976"],
      ["ViT",      "2.004", "1.483", "-0.047", "0.980"]], [5]),

    ("precipitation  (m/day, physical space)", None,
     [["bilinear", "0.0046", "0.0016", "-0.0001", "0.750"],
      ["bicubic",  "0.0046", "0.0016", "-0.0001", "0.745"],
      ["BCSD",     "0.0049", "0.0015", "-0.0009", "0.743"],
      ["UNet",     "0.0045", "0.0014", "-0.0005", "0.763"],
      ["ViT",      "0.0047", "0.0015", "-0.0008", "0.755"]], [3]),

    ("precipitation  (log1p(mm), log space)", "UNet / ViT not evaluated in this space",
     [["nearest",  "0.583", "0.307", "+0.087", "0.794"],
      ["bilinear", "0.572", "0.305", "+0.087", "0.800"],
      ["bicubic",  "0.579", "0.308", "+0.088", "0.797"],
      ["BCSD",     "0.545", "0.295", "-0.014", "0.808"]], [3]),
]

NOTE = [
    "Test: 2020, all 365 days, land mask, full-frame. Train: 1980-2017.",
    "The two precipitation spaces are NOT convertible - compare only within a table.",
    "Same BCSD: rank 1 in log space, rank last in physical space - the ranking flips.",
    "ViT has 15.5M params, UNet 1.6M.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(PROJECT_ROOT / "runs/progress_tables.png"))
    a = p.parse_args()

    fig = plt.figure(figsize=(8.0, 11.2), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    L, R = 0.06, 0.95
    COLS = [L, 0.36, 0.53, 0.71, 0.92]      # 列 x 锚点(显式 -> 严格对齐)
    RH = 0.0268                              # 行高
    y = 0.972

    for title, sub, rows, hi in TABLES:
        ax.text(L, y, title, fontsize=14.5, color=INK, fontweight="bold", va="top")
        y -= 0.026
        if sub:
            ax.text(L, y, sub, fontsize=10, color=MUTED, va="top")
            y -= 0.021

        for j, h in enumerate(HEAD):
            ax.text(COLS[j], y, h, fontsize=11, color=MUTED, va="top",
                    ha="left" if j == 0 else "right")
        y -= 0.019
        ax.plot([L, R], [y, y], color=INK2, lw=1.1)
        y -= 0.008

        for i, row in enumerate(rows):
            if i in hi:
                ax.add_patch(plt.Rectangle((L - 0.014, y - RH + 0.006), R - L + 0.028, RH,
                                           color=HL, zorder=0))
            bold = "bold" if i in hi else "normal"
            for j in range(len(HEAD)):
                ax.text(COLS[j], y - 0.005, row[j], fontsize=12,
                        color=INK if i in hi else INK2, fontweight=bold, va="top",
                        ha="left" if j == 0 else "right")
            y -= RH
            if i < len(rows) - 1:
                ax.plot([L, R], [y + 0.006, y + 0.006], color=RULE, lw=0.6, zorder=0)
        ax.plot([L, R], [y + 0.006, y + 0.006], color=INK2, lw=1.0)
        y -= 0.042

    y -= 0.006
    for line in NOTE:
        ax.text(L, y, line, fontsize=10.3, color=MUTED, va="top")
        y -= 0.019

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
