#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一次 JiT/JiTMoE 训练的 train/val loss 画成单图。

横轴是累计样本数(百万), 不是 epoch —— JiT 走的是洗牌无放回的连续帧流, 没有 epoch 边界。
纵轴对数刻度: loss 头两百万样本内掉两个数量级, 线性轴会把整条尾巴压成一条平线。

横轴固定画到 --duration(默认 8M 训练预算)。跑满的曲线正好铺满, 没跑满的在中途断掉,
一眼能看出训练量差异 —— 这正是并排比较不同 run 时最容易被忽略的一项。

用法:
  python -m era5_daymet.tools.plotting.plot_jit_loss_curve \
      --run runs/exp/<id> --out <path>.png [--duration 8000000]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

C_TRAIN = "#c1561e"
C_VAL = "#3a6fa8"
YTICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def main():
    p = argparse.ArgumentParser(description="JiT 训练 loss 曲线(单图)")
    p.add_argument("--run", required=True, help="实验目录, 需含 loss_history.json")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=8e6, help="训练预算, 决定横轴右端")
    p.add_argument("--dpi", type=int, default=140)
    a = p.parse_args()

    hist = json.loads((Path(a.run) / "loss_history.json").read_text())
    x = [r["samples"] / 1e6 for r in hist]
    tr = [r["train"] for r in hist]
    va = [r["val"] for r in hist]
    ib = min(range(len(va)), key=lambda i: va[i])

    fig, ax = plt.subplots(figsize=(10.0, 4.3))
    ax.plot(x, tr, "-", color=C_TRAIN, linewidth=1.7, label="train")
    ax.plot(x, va, "-", color=C_VAL, linewidth=1.7, label="val")
    ax.plot([x[ib]], [va[ib]], "o", color=C_VAL, markersize=5)
    ax.annotate(f"best {va[ib]:.4f} @ {x[ib]:.2f}M",
                xy=(x[ib], va[ib]), xytext=(x[ib] - 1.2, va[ib] * 2.6),
                color="#555555", fontsize=11,
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.9))

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    # 对数轴上刻度值跨几个数量级, 定点小数会写成 "2.00"/"0.05" 两种精度; 去掉尾零统一成最短写法
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".")))
    lo, hi = min(min(tr), min(va)), max(max(tr), max(va))
    ax.set_ylim(lo * 0.75, hi * 1.35)
    ax.set_xlim(0, a.duration / 1e6)
    ax.set_xlabel("samples (M)")
    ax.set_ylabel("loss  (flow-matching, land)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=12)

    # 线尾贴标签: 两条线在尾部常常挨得很近, 光靠图例不好认哪条是哪条
    for y, c, t in ((va[-1], C_VAL, "val"), (tr[-1], C_TRAIN, "train")):
        ax.annotate(t, xy=(x[-1], y), xytext=(4, 0), textcoords="offset points",
                    color=c, fontsize=11, fontweight="bold", va="center")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[loss] {Path(a.run).name}: {len(hist)} 点, 末 {x[-1]:.2f}M/"
          f"{a.duration / 1e6:.0f}M, best val {va[ib]:.6f} @ {x[ib]:.2f}M -> {a.out}")


if __name__ == "__main__":
    main()
