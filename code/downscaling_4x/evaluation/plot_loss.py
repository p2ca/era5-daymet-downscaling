#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
plot_loss.py — 训练损失曲线的唯一出图入口
============================================================================
输入是训练进程写下的 loss_history.json。三个训练入口的横轴单位本就不同 —— 有 epoch
边界的按 epoch, 洗牌无放回连续帧流的按累计样本数 —— 因此横轴由数据自身的键决定, 不由
调用方指定:

    含 "epoch"   -> 横轴 epoch      (确定性: UNet / ViT / CorrDiff 阶段A)
    含 "samples" -> 横轴 samples(M) (CorrDiff 阶段B、JiT / JiT-MoE)

一张图里只允许同一种横轴: epoch 与 samples 之间没有换算关系, 混在一起画出来的图看着
正常, 但两条曲线的横向位置根本不可比。

纵轴默认对数: loss 在头部会掉一到两个数量级, 线性轴把整条尾巴压成一条平线, 而尾部恰是
判断收敛与过拟合的地方。有 lr 的 run 另在右轴以对数虚线画出学习率。

★ 纵轴数值不可跨方法比较: 各训练入口的 val 口径不同(逐 epoch 全 730 帧 / 固定 1024 个
patch / 固定验证帧与噪声), 且目标不同、归一化空间不同。本工具把能读到的 val 口径写进
图注, 但读图人仍需自行确认两条曲线是否可比。

用法:
    python -m downscaling_4x.evaluation.plot_loss --run CA1 runs/exp/<id> --out fig.png
    python -m downscaling_4x.evaluation.plot_loss --run dense runs/exp/<a> \
        --run moe runs/exp/<b> --out cmp.png
============================================================================
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling_4x.tools.plotting.mpl_style import use_cjk

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def load(run_dir):
    """读一次训练的损失历史 -> (横轴键, 点列表)。横轴由数据的键决定, 不由调用方指定。"""
    d = Path(run_dir)
    data = json.loads((d / "loss_history.json").read_text())
    hist = data["history"] if isinstance(data, dict) and "history" in data else data
    if not isinstance(hist, list) or not hist:
        raise SystemExit(f"{d}: loss_history.json 不是非空列表")
    keys = set(hist[0])
    xkey = "epoch" if "epoch" in keys else ("samples" if "samples" in keys else None)
    if xkey is None:
        raise SystemExit(f"{d}: loss_history 既无 epoch 也无 samples, 无法确定横轴")
    return xkey, hist


def val_note(run_dir):
    """从 meta.json 取该 run 的 val 口径; 取不到就不写 —— 宁可留白也不臆造口径。"""
    p = Path(run_dir) / "meta.json"
    if not p.exists():
        return ""
    try:
        m = json.loads(p.read_text())
    except Exception:
        return ""
    sp = m.get("sampling_protocol") or {}
    if isinstance(sp, dict) and sp.get("val"):
        return str(sp["val"])
    cfg = m.get("config") or {}
    if cfg.get("val_every"):
        return f"每 {int(cfg['val_every']):,} samples 一点"
    return ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", action="append", nargs=2, metavar=("标签", "目录"), required=True,
                   help="可重复; 同一张图里各 run 的横轴种类必须一致")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=0.0,
                   help="samples 横轴的右端(样本数); 缺省取各 run 的最大值。"
                        "给训练预算可让'没跑满'一眼可见")
    p.add_argument("--ymax", type=float, default=0.0,
                   help="纵轴上限; 缺省自动。头部的一两个数量级会把尾段压成一条线, "
                        "比较接近收敛的曲线时切掉头部才看得出差别")
    p.add_argument("--linear", action="store_true", help="纵轴改线性(缺省对数)")
    p.add_argument("--no-lr", action="store_true", help="不画学习率右轴")
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args()
    use_cjk()                    # 图注的 val 口径是中文, 缺字只警告不报错

    runs = []
    for label, d in a.run:
        xkey, hist = load(d)
        runs.append((label, Path(d), xkey, hist))
    kinds = {r[2] for r in runs}
    if len(kinds) > 1:
        detail = "; ".join(f"{r[0]}={r[2]}" for r in runs)
        raise SystemExit(f"横轴种类不一致, 不能画进同一张图: {detail}。"
                         f" epoch 与 samples 之间没有换算关系。")
    xkey = kinds.pop()
    xdiv = 1e6 if xkey == "samples" else 1.0
    xlabel = "samples (M)" if xkey == "samples" else "epoch"

    fig, ax = plt.subplots(figsize=(9.6, 5.0), constrained_layout=True)
    ax_lr = None
    for i, (label, d, _k, hist) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        x = [r[xkey] / xdiv for r in hist]
        tr = [r.get("train") for r in hist]
        va = [r.get("val") for r in hist]
        if any(v is not None for v in tr):
            ax.plot(x, tr, "-", color=c, lw=1.3, alpha=0.55, label=f"{label} train")
        # best 写进图例条目而非点旁标注: 多个 run 的 best 常落在相近位置, 点旁标注必然互相压住
        fin = [(xi, v) for xi, v in zip(x, va) if v is not None]
        note = ""
        if fin:
            bx, bv = min(fin, key=lambda t: t[1])
            at = f"{bx:.4g}M" if xkey == "samples" else f"{bx:g}"
            note = f"  · best {bv:.4g} @ {at}"
        ax.plot(x, va, "-", color=c, lw=1.9, label=f"{label} val{note}")
        if fin:
            ax.plot([bx], [bv], "o", color=c, ms=5)
        lrs = [r.get("lr") for r in hist]
        if not a.no_lr and any(v for v in lrs):
            ax_lr = ax_lr or ax.twinx()
            ax_lr.plot(x, lrs, "--", color=c, lw=1.1, alpha=0.7)

    if not a.linear:
        ax.set_yscale("log")
    if a.ymax:
        ys = [v for _l, _d, _k, h in runs for r in h
              for v in (r.get("train"), r.get("val")) if v is not None and v <= a.ymax]
        ax.set_ylim(bottom=min(ys) * 0.97 if ys else None, top=a.ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_axisbelow(True)
    if xkey == "samples":
        right = (a.duration / xdiv) if a.duration else max(
            max(r[xkey] for r in h) / xdiv for _l, _d, _k, h in runs)
        ax.set_xlim(0, right)           # 右端固定 -> 没跑满的 run 在中途断掉, 一眼可见
    if ax_lr is not None:
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("learning rate (虚线)")
    ax.legend(fontsize=8, loc="upper right")

    notes = [f"{l}: {val_note(d)}" for l, d, _k, _h in runs if val_note(d)]
    if notes:
        fig.suptitle("val 口径  " + " | ".join(notes), fontsize=8, color="#555555")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi)          # constrained_layout 已管边距, 再加 tight 会冲突
    plt.close(fig)
    for l, d, _k, h in runs:
        fin = [(r[xkey], r["val"]) for r in h if r.get("val") is not None]
        if fin:
            bx, bv = min(fin, key=lambda t: t[1])
            print(f"[loss] {l}: {len(h)} 点, 横轴={xkey}, 末 {h[-1][xkey]:,}, "
                  f"best val={bv:.6g}@{bx:,}")
        else:
            print(f"[loss] {l}: {len(h)} 点")
    print(f"[loss] -> {out}")


if __name__ == "__main__":
    main()
