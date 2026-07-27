#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
plot_table_corrdiff.py — 六方法多指标对比表 (含 CorrDiff, 含 SSIM/CRPS)

★同一 13 天口径★: CorrDiff 只在 2020 的 13 天(stride 30)评过, 若塞进确定性方法的
365 天表就是跨口径假比。故本表统一用 13 天: 5 个确定性方法取自 baseline-13day,
CorrDiff 取自 corrdiff-b64, 两者是★同样 13 天★, 可并排。(365 天的确定性排名与此
基本一致, 见 20260713-eval-all-ssim。)

★每列各自加粗最优值★(不是高亮某一行): 整个故事就是"不同指标赢家不同" ——
CorrDiff 赢 CRPS(概率技巧)、输 RMSE(逐点), 一眼可见。

数据: runs/exp/20260715-baseline-13day/metrics_all.json (bilinear/bicubic/bcsd/unet/vit)
      runs/exp/20260714-corrdiff-b64/metrics_corrdiff.json (corrdiff, 16 成员)
用法: python code/plot_table_corrdiff.py [--out runs/exp/20260715-meeting-maps/table_6methods.png]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT = os.path.expanduser("~/.fonts/NotoSansSC-Regular.otf")
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#8a8880", "#d8d7d0", "#ffffff"
HL = "#e8f2fd"

from era5_daymet.paths import PROJECT_ROOT

ROOT = os.fspath(PROJECT_ROOT)
DET = os.path.join(ROOT, "runs/exp/20260715-baseline-13day/metrics_all.json")
CD = os.path.join(ROOT, "runs/exp/20260714-corrdiff-b64/metrics_corrdiff.json")

METHODS = ["bilinear", "bicubic", "bcsd", "unet", "vit", "corrdiff"]
PRETTY = {"bilinear": "bilinear", "bicubic": "bicubic", "bcsd": "BCSD",
          "unet": "UNet", "vit": "ViT", "corrdiff": "CorrDiff (16)"}

# (变量, 标题, 结构列: 'ssim' 或 'ssim_log', 数值格式)
TEMP_FMT = dict(rmse="{:.3f}", mae="{:.3f}", bias="{:+.3f}", corr="{:.3f}",
                crps="{:.3f}", struct="{:.3f}")
PREC_FMT = dict(rmse="{:.4f}", mae="{:.4f}", bias="{:+.4f}", corr="{:.3f}",
                crps="{:.4f}", struct="{:.3f}")

SPECS = [
    ("2m_temperature_max", "tmax  (K)", "ssim", "SSIM", TEMP_FMT),
    ("2m_temperature_min", "tmin  (K)", "ssim", "SSIM", TEMP_FMT),
    ("total_precipitation_24hr", "precipitation  (m/day, physical space)", "ssim_log", "SSIMlog", PREC_FMT),
]
# 各指标"越优"方向: True=越小越好; bias 特判(绝对值越小越好); corr/struct 越大越好
LOWER_BETTER = dict(rmse=True, mae=True, crps=True, corr=False, struct=False)


def load(det_path=DET, cd_path=CD):
    det = json.load(open(det_path)); cd = json.load(open(cd_path))
    def get(m, v):
        d = (cd if m == "corrdiff" else det)[m][v]
        return dict(rmse=d["rmse"], mae=d["mae"], bias=d["bias"], corr=d["corr"],
                    crps=d["crps"], ssim=d.get("ssim"), ssim_log=d.get("ssim_log"))
    return {v: {m: get(m, v) for m in METHODS} for v, *_ in SPECS}


def best_idx(rows, key):
    """返回该指标最优的行号(bias 取绝对值最小)。"""
    vals = [r[key] for r in rows]
    if key == "bias":
        vals = [abs(x) for x in vals]
        return vals.index(min(vals))
    if LOWER_BETTER.get(key, True):
        return vals.index(min(vals))
    return vals.index(max(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "runs/exp/20260715-meeting-maps/table_6methods.png"))
    p.add_argument("--det", default=DET, help="确定性方法指标 json(bilinear/.../unet/vit)")
    p.add_argument("--cd", default=CD, help="CorrDiff 指标 json")
    a = p.parse_args()
    data = load(a.det, a.cd)

    HEAD = ["method", "RMSE", "MAE", "bias", "corr", "CRPS", None]   # 末列结构名按表填
    KEYS = ["rmse", "mae", "bias", "corr", "crps", "struct"]

    fig = plt.figure(figsize=(9.6, 8.4), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    L, R = 0.05, 0.97
    COLS = [L, 0.36, 0.49, 0.63, 0.75, 0.86, 0.97]
    RH = 0.031
    y = 0.975

    for vkey, title, struct_key, struct_name, fmt in SPECS:
        rows = [data[vkey][m] for m in METHODS]
        # 结构列取值
        for r in rows:
            r["struct"] = r[struct_key]
        winners = {k: best_idx(rows, k) for k in KEYS}

        ax.text(L, y, title, fontsize=14, color=INK, fontweight="bold", va="top")
        y -= 0.030
        head = HEAD[:-1] + [struct_name]
        for j, h in enumerate(head):
            ax.text(COLS[j], y, h, fontsize=10.5, color=MUTED, va="top",
                    ha="left" if j == 0 else "right")
        y -= 0.020
        ax.plot([L, R], [y, y], color=INK2, lw=1.1)
        y -= 0.009

        for i, m in enumerate(METHODS):
            r = rows[i]
            hl = (m == "corrdiff")
            if hl:                                   # CorrDiff 行淡底, 标出它是唯一集合方法
                ax.add_patch(plt.Rectangle((L - 0.012, y - RH + 0.007), R - L + 0.024, RH,
                                           color=HL, zorder=0))
            ax.text(COLS[0], y - 0.005, PRETTY[m], fontsize=11.5,
                    color=INK if hl else INK2, va="top", ha="left",
                    fontweight="bold" if hl else "normal")
            for j, k in enumerate(KEYS, start=1):
                txt = fmt[k].format(r[k])
                win = (i == winners[k])
                ax.text(COLS[j], y - 0.005, txt, fontsize=11.5,
                        color=INK if win else INK2,
                        fontweight="bold" if win else "normal", va="top", ha="right")
            y -= RH
            if i < len(METHODS) - 1:
                ax.plot([L, R], [y + 0.007, y + 0.007], color=RULE, lw=0.6, zorder=0)
        ax.plot([L, R], [y + 0.007, y + 0.007], color=INK2, lw=1.0)
        y -= 0.032

    NOTE = [
        "同一 13 天口径 (2020, stride 30, land-masked)。所有方法在★同样 13 天★上评测 —— 唯一能与 CorrDiff 并排的基准。",
        "★每列各自加粗最优值 (bias 取 |·| 最小)。CorrDiff 三变量 CRPS 全部最优、RMSE 全部垫底 —— 概率技巧 vs 逐点精度的分野。",
        "CorrDiff = 16 成员集合; 其余为单一预测 (N=1)。RMSE/MAE/bias/corr/SSIM 算在集合均值上, CRPS 用整个集合 (确定性方法 CRPS≡MAE)。",
        "降水物理空间 SSIM 被共同干区虚高 -> 报 SSIMlog (log1p 空间); 五方法仍挤在 0.54-0.56 噪声级, 结构上无人突破。",
        "降水两个单位空间不可通约; 365 天确定性排名与本 13 天基本一致 (20260713-eval-all-ssim)。",
    ]
    y -= 0.004
    for line in NOTE:
        ax.text(L, y, line, fontsize=9.2, color=MUTED, va="top")
        y -= 0.0185

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
