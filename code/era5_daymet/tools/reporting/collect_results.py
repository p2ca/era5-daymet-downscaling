#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
collect_results.py — 扫描 runs/exp/*/meta.json, 生成 runs/LEDGER.md 实验台账
============================================================================
台账是"生成物", 不手改。改了 meta.json 就重跑本脚本, 台账永远与产物一致。

  python code/collect_results.py            # 打印到屏幕 + 写 runs/LEDGER.md
  python code/collect_results.py --print    # 只打印, 不写文件

核心约束: 指标按 units 分组输出。降水在两种空间评测过
(train_statistical=log1p(mm), eval_all_methods=m/day), 两者 RMSE 不通约,
所以本脚本绝不把它们并进同一张表 —— 不同单位 = 不同表。
============================================================================
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from era5_daymet.paths import PROJECT_ROOT

ROOT = os.fspath(PROJECT_ROOT)
EXP = os.path.join(ROOT, "runs", "exp")
LEDGER = os.path.join(ROOT, "runs", "LEDGER.md")

COLS = ("rmse", "mae", "bias", "corr")


def load_metas():
    metas = []
    if not os.path.isdir(EXP):
        return metas
    for d in sorted(os.listdir(EXP)):
        p = os.path.join(EXP, d, "meta.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            m["_dir"] = d
            metas.append(m)
        else:
            print(f"  [warn] {d}/ 缺 meta.json — 不会进台账", file=sys.stderr)
    return metas


def flatten(m):
    """meta.key_metrics 有两种形状: {var: metrics} 或 {var: {method: metrics}}。
    统一摊平成 (var, method, metrics, unit) 四元组。"""
    out = []
    km = m.get("key_metrics", {}) or {}
    units = m.get("units", {}) or {}
    for var, val in km.items():
        if not isinstance(val, dict):
            continue
        unit = units.get(var, "?")
        if any(k in val for k in COLS):                 # {var: metrics}
            out.append((var, m["method"], val, unit))
        else:                                            # {var: {method: metrics}}
            for meth, mm in val.items():
                if isinstance(mm, dict):
                    out.append((var, meth, mm, unit))
    return out


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def build(metas):
    L = []
    L.append("# 实验台账 (LEDGER)\n")
    L.append("> 本文件负责：自动汇总正式实验的身份、核心指标和已知缺口；内容由脚本生成，禁止手工编辑。\n")
    L.append("> **本文件由 `python code/collect_results.py` 自动生成, 不要手改。**")
    L.append("> 要改内容 -> 改对应 `runs/exp/<id>/meta.json` -> 重跑脚本。\n")

    # ---- 1. 实验清单 ----
    L.append("## 实验清单\n")
    L.append("| 实验 ID | 方法 | 状态 | 机时 | 一句话结论 |")
    L.append("|---|---|---|---|---|")
    for m in metas:
        L.append("| `{}` | {} | {} | {} | {} |".format(
            m["_dir"], m.get("method", "?"), m.get("status", "?"),
            m.get("elapsed", "—"), m.get("headline", "")))
    L.append("")

    # ---- 2. 指标: 按 (变量, 单位) 分组 ----
    # 同一变量若有多个单位空间 -> 拆成多张表, 并明确警告不可跨表比较
    groups = defaultdict(list)   # (var, unit) -> [(exp_id, method, metrics)]
    for m in metas:
        # 只有 status=done 的实验进对照表。发散/取消的跑, 其指标来自不该被引用的检查点
        # (例: 20260712-vit-d384 的指标出自 ep1 的 ckpt), 与真 baseline 并排会得出假结论。
        # 它们仍留在上面的"实验清单"里, 结论和曲线在各自 meta.json 中。
        if m.get("status") != "done":
            continue
        for var, meth, mm, unit in flatten(m):
            groups[(var, unit)].append((m["_dir"], meth, mm))

    var_units = defaultdict(set)
    for (var, unit) in groups:
        var_units[var].add(unit)

    L.append("## 指标对照\n")
    L.append("按 **(变量, 单位)** 分组。**不同单位的表之间不可比较** —— "
             "降水在 log1p(mm) 与 m/day 两种空间都评测过, RMSE 之间没有换算关系。\n")

    for var in sorted(var_units):
        units = sorted(var_units[var])
        for unit in units:
            rows = groups[(var, unit)]
            if not rows:
                continue
            title = f"### {var}  [{unit}]"
            if len(units) > 1:
                title += f"   ⚠️ 本变量有 {len(units)} 种单位空间, 仅可在本表内部比较"
            L.append(title + "\n")
            L.append("| 方法 | RMSE | MAE | bias | corr | 来源实验 |")
            L.append("|---|---|---|---|---|---|")
            # 去重: 同 (方法) 若多个实验给出, 全列出(便于交叉核对)
            for exp_id, meth, mm in sorted(rows, key=lambda r: (r[1], r[0])):
                L.append("| {} | {} | {} | {} | {} | `{}` |".format(
                    meth, fmt(mm.get("rmse")), fmt(mm.get("mae")),
                    fmt(mm.get("bias")), fmt(mm.get("corr")), exp_id))
            L.append("")

    # ---- 3. 待办/缺口 ----
    gaps = [(m["_dir"], m["gap"]) for m in metas if m.get("gap")]
    if gaps:
        L.append("## 已知缺口\n")
        for exp_id, g in gaps:
            L.append(f"- **`{exp_id}`**: {g}")
        L.append("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="只打印, 不写 LEDGER.md")
    a = ap.parse_args()

    metas = load_metas()
    if not metas:
        print("runs/exp/ 下没有带 meta.json 的实验", file=sys.stderr)
        return 1
    txt = build(metas)
    print(txt)
    if not a.print:
        with open(LEDGER, "w") as f:
            f.write(txt + "\n")
        print(f"\n-> 已写入 {LEDGER}  ({len(metas)} 个实验)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
