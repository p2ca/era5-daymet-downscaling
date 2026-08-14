#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
============================================================================
collect_results.py — 扫描 runs/exp/*/meta.json, 生成 runs/STATUS.md 与 runs/LEDGER.md
============================================================================
两份都是"生成物", 不手改。改了 meta.json 就重跑本脚本, 内容永远与产物一致。

  python code/collect_results.py            # 写 runs/STATUS.md + runs/LEDGER.md
  python code/collect_results.py --print    # 只打印, 不写文件

  STATUS.md  现行状态: 数据合同 + 现行基线指标 + 在途工作。开新 session 只读它。
  LEDGER.md  全部实验流水与指标对照, 按需查阅。

数据合同段不是手写的, 而是从代码常量派生(DEFAULT_IN / TARGETS / splits / precip_*),
因此不会与实现漂移; era5_daymet.tests.test_spec_contract 另行断言这些常量本身没被改动。

核心约束: 指标按 units 分组输出。降水在两种空间评测过
(train_statistical=log1p(mm), eval_all_methods=m/day), 两者 RMSE 不通约,
所以本脚本绝不把它们并进同一张表 —— 不同单位 = 不同表。
============================================================================
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

from era5_daymet.paths import PROJECT_ROOT

ROOT = os.fspath(PROJECT_ROOT)
EXP = os.path.join(ROOT, "runs", "exp")
LEDGER = os.path.join(ROOT, "runs", "LEDGER.md")
STATUS = os.path.join(ROOT, "runs", "STATUS.md")
STATS_META = os.path.join(ROOT, "runs", "stats", "train_dayofyear", "daymet", "meta.json")

COLS = ("rmse", "mae", "bias", "corr")

# 基线表逐目标出一张: 同一目标下各方法并排, 避免把不同目标的数塞进一行。
PRECIP_VAR = "total_precipitation_24hr"
BASELINE_VARS = (("2m_temperature_max", "2m_temperature_max [K]"),
                 ("2m_temperature_min", "2m_temperature_min [K]"),
                 (PRECIP_VAR, "total_precipitation_24hr"))


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
        # 集合方法与确定性方法同表, 必须带上成员数: 集合平均本身就压 RMSE, 不标出来会被误读成模型更强
        ens = (m.get("eval") or {}).get("ensemble", 1)
        ssim = m.get("ssim") or {}
        for var, meth, mm, unit in flatten(m):
            # ssim 块只描述本实验自己的方法; 一次跑多方法时(如 BCSD 那次带了插值对照)
            # 不能把它套到别的方法行上
            ss = (ssim.get(var) or {}).get("ssim") if meth == m.get("method") else None
            groups[(var, unit)].append((m["_dir"], meth, mm, ens, ss))

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
            L.append("| 方法 | 成员数 | RMSE | MAE | bias | corr | SSIM | 来源实验 |")
            L.append("|---|---|---|---|---|---|---|---|")
            # 去重: 同 (方法) 若多个实验给出, 全列出(便于交叉核对)
            for exp_id, meth, mm, ens, ss in sorted(rows, key=lambda r: (r[1], r[0])):
                L.append("| {} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                    meth, ens, fmt(mm.get("rmse")), fmt(mm.get("mae")),
                    fmt(mm.get("bias")), fmt(mm.get("corr")), fmt(ss), exp_id))
            L.append("> 成员数 >1 的行是集合均值上的指标; 与成员数 1 的确定性方法并排看时, "
                     "集合平均本身就会压低 RMSE。\n")

    # ---- 3. 待办/缺口 ----
    gaps = [(m["_dir"], m["gap"]) for m in metas if m.get("gap")]
    if gaps:
        L.append("## 已知缺口\n")
        for exp_id, g in gaps:
            L.append(f"- **`{exp_id}`**: {g}")
        L.append("")

    return "\n".join(L)


def spec_contract():
    """把固定数据合同从实现里读出来, 而不是抄一遍。任何一处改了实现, 本表随之改变。"""
    from era5_daymet.data import match_era5_daymet as M
    from era5_daymet import contract as C

    sp = {k: (f"{v[0]}–{v[-1]}" if len(v) > 1 else str(v[0])) for k, v in M.splits.items()}
    n_cond = C.cond_channels(C.DEFAULT_IN, C.TARGETS, use_clim=False)
    pm = {}
    if os.path.exists(STATS_META):
        with open(STATS_META) as f:
            pm = json.load(f)
    clip = pm.get("precip_clip", 0.1)
    scale = pm.get("precip_scale", 1000.0)

    L = ["## 1. 数据合同（派生自代码常量, 非手写）\n",
         "| 项 | 固定值 |", "|---|---|",
         f"| 空间倍率 | ERA5 双线性上采样 {C.FACTOR}× |",
         f"| 条件输入 | **{n_cond} 通道** = {len(C.DEFAULT_IN)} ERA5 动态 + 3 Daymet 静态"
         f"(Δz / landcover / land_sea_mask); 无气候态 |",
         f"| 预测目标 | {', '.join(C.TARGETS)} |",
         f"| 数据划分 | train {sp['train']} / val {sp['val']} / test {sp['test']}; 365 天历 |",
         f"| 降水管线 | ×{scale:g} → mm → <{clip:g} mm 置零 → log1p → z-score;"
         f" 反变换 expm1 并钳到 log1p ≤ {C.PRECIP_LOG_MAX:g} |",
         "| 掩膜 | 只在陆地算 loss 与指标 |", "",
         "**ERA5 动态输入必须严格按此顺序读取**（`era5_daymet/contract.py` 的 `DEFAULT_IN`）:\n",
         "```", *(f"{i:2d}. {v}" for i, v in enumerate(C.DEFAULT_IN, 1)), "```", "",
         "> 降水存在 `log1p(mm)` 与 `m/day` 两种单位空间, RMSE 之间没有换算关系, 排名甚至相反;",
         "> 任何跨方法比较前先确认单位一致。\n"]
    return L


def audit(metas):
    """把 meta.json 的自述与实验目录里的产物对拍, 返回 (实验 ID, 问题) 列表。

    meta.json 是人写的, loss_history.json 与 checkpoint 是训练进程写的。两者一旦分叉,
    本脚本会照抄 meta 的那一份 —— 生成物忠实反映输入, 不会自己发现输入是错的。这里做
    只读对拍, 不修改任何 meta: 谁写的谁负责改, 但必须让分叉在 STATUS 里可见。

    比较用容差而非严格相等: 验证点每 val_every 个 samples 才落一个, 训练可以停在两个
    验证点之间, 因此 samples_trained 合法地略大于曲线末点, 差值应小于一个验证间隔。
    """
    out = []
    for m in metas:
        d = os.path.join(EXP, m["_dir"])
        hp = os.path.join(d, "loss_history.json")
        tr = m.get("training") or {}
        status = str(m.get("status", ""))

        if os.path.exists(hp):
            try:
                with open(hp) as f:
                    hist = json.load(f)
            except (OSError, ValueError):
                hist = []
            if hist:
                every = int(tr.get("val_every") or 0)
                last_n = int(hist[-1].get("samples", 0))
                best = min(hist, key=lambda x: x.get("val", float("inf")))
                claim_n = tr.get("samples_trained")
                if claim_n is not None:
                    gap = int(claim_n) - last_n
                    if gap < 0 or (every and gap >= every):
                        out.append((m["_dir"], f"meta 记 samples_trained {int(claim_n):,}, "
                                    f"曲线末点 {last_n:,} (差 {gap:,}, 超出一个验证间隔 {every:,})"))
                claim_v = tr.get("best_val")
                if claim_v is not None and best.get("val") is not None:
                    # 按 meta 自己声明的精度比较: meta 合法地记四舍五入值(如 0.05259),
                    # 用绝对容差会把"精度不同"误报成"数值不符"。问的是"在它声称的精度下是否一致"。
                    dec = len(str(claim_v).split(".")[-1]) if "." in str(claim_v) else 0
                    if round(float(best["val"]), dec) != round(float(claim_v), dec):
                        out.append((m["_dir"], f"meta 记 best_val {claim_v}, "
                                    f"曲线最优 {best['val']:.6f} @ {int(best['samples']):,}"))
                claim_at = tr.get("best_val_at_samples")
                # 样本数是整数, 无舍入余地, 严格比较
                if claim_at is not None and int(claim_at) != int(best.get("samples", -1)):
                    out.append((m["_dir"], f"meta 记 best_val_at_samples {int(claim_at):,}, "
                                f"曲线最优点在 {int(best.get('samples', -1)):,}"))

        if status.startswith("running"):
            # 训练进程会持续写 last.pt; 长时间不动而 status 仍是 running, 多半是作业早已结束
            lp = os.path.join(d, "last.pt")
            if os.path.exists(lp):
                idle_h = (time.time() - os.path.getmtime(lp)) / 3600.0
                if idle_h > 3.0:
                    out.append((m["_dir"], f"status 仍是 running, 但 last.pt 已 {idle_h:.1f} 小时未更新"))
            else:
                out.append((m["_dir"], "status 是 running, 但目录里没有 last.pt"))
    return out


def build_status(metas):
    L = ["# 当前状态 (STATUS)\n",
         "> 本文件负责：汇总现行数据合同、现行基线指标与在途工作，供新 session 单点读取；"
         "内容由脚本生成，禁止手工编辑。\n",
         "> **本文件由 `python code/collect_results.py` 自动生成, 不要手改。**",
         "> 合同段派生自代码常量; 指标与状态来自 `runs/exp/<id>/meta.json`。",
         "> 要改内容 -> 改代码或对应 meta.json -> 重跑脚本。\n"]
    L += spec_contract()

    # ---- 现行确定性基线: 由 meta.json 的 current_baseline 显式登记 ----
    L.append("## 2. 现行基线（2020 测试年, 陆地, 365 天）\n")
    base = []
    for m in metas:
        label = m.get("current_baseline")
        if not label:
            continue
        # 一个实验可能评了多个方法(如 BCSD 那次同时跑了插值), 取与本实验 method 同名的那条
        own, km = m.get("method"), {}
        for var, meth, mm, unit in flatten(m):
            if var not in km or meth == own:
                km[var] = (mm, unit)
        # SSIM 不在 key_metrics 里, 由 meta 的 ssim 块单独登记(逐目标 {"ssim": ...})
        base.append((label, km, m.get("ssim") or {}, m["_dir"]))

    if base:
        for var, head in BASELINE_VARS:
            L.append(f"### {head}\n")
            wide = var == PRECIP_VAR          # 降水两种单位空间并存, 必须逐行标单位
            L += ["| 方法 | MAE | RMSE | SSIM | corr |" + (" 单位 |" if wide else "") + " 来源实验 |",
                  "|---|---|---|---|---|" + ("---|" if wide else "") + "---|"]
            for label, km, ssim, d in sorted(base, key=lambda r: r[0]):
                mm, unit = km.get(var, ({}, "?"))
                cells = [label, fmt(mm.get("mae")), fmt(mm.get("rmse")),
                         fmt((ssim.get(var) or {}).get("ssim")), fmt(mm.get("corr"))]
                if wide:
                    cells.append(unit)
                L.append("| " + " | ".join(cells) + f" | `{d}` |")
            L.append("")
        L += ["> 降水的 MAE/RMSE 存在 `log1p(mm)` 与 `m/day` 两种单位空间, 之间没有换算关系,",
              "> 排名甚至相反; SSIM 一律在物理空间上算, 各方法可比。\n"]
    else:
        L.append("_尚无实验登记 `current_baseline` 字段。_\n")

    # ---- CorrDiff: 概率指标自成一表, 与确定性 RMSE 不同框 ----
    cd = [m for m in metas if m.get("model_version") and m.get("test_2020", {}).get("bigcheck_365d")]
    if cd:
        L.append("### CorrDiff（集合方法, 概率指标不与上表同框）\n")
        L += ["| 目标 | 版本 | CRPS | μ 的 MAE | CRPSS | ens-mean RMSE | 来源实验 |",
              "|---|---|---|---|---|---|---|"]
        for m in sorted(cd, key=lambda x: str(x.get("target", x["_dir"]))):
            b = m["test_2020"]["bigcheck_365d"]
            L.append("| {} | {} | {} | {} | {} | {} | `{}` |".format(
                m.get("target", m.get("headline", "")[:24]), m["model_version"],
                fmt(b.get("crps_ens")), fmt(b.get("mae_mu")), fmt(b.get("crpss")),
                fmt(b.get("rmse_ens_mean")), m["_dir"]))
        L.append("")

    # ---- 在途/阻塞: 状态里带"待"或 pending 的实验 ----
    pend = [m for m in metas
            if any(k in str(m.get("status", "")) for k in ("待", "pending", "running", "queued"))]
    L.append("## 3. 在途与阻塞\n")
    if pend:
        L += ["| 实验 ID | 状态 |", "|---|---|"]
        L += [f"| `{m['_dir']}` | {m.get('status')} |" for m in pend]
        L.append("")
    else:
        L.append("_无。_\n")

    issues = audit(metas)
    if issues:
        L += ["## 3.5 待核对（meta.json 与产物对不上）\n",
              "> 以下是 meta.json 的自述与实验目录里 `loss_history.json` / `last.pt` 的对拍结果。",
              "> 本表由脚本对拍生成, 不代表实验有问题, 但**引用这些实验的数字前必须先核对**。\n",
              "| 实验 ID | 分歧 |", "|---|---|"]
        L += [f"| `{d}` | {msg} |" for d, msg in issues]
        L.append("")

    n_sup = sum(1 for m in metas if "supersed" in str(m.get("status", "")))
    L += ["## 4. 更细的东西去哪查\n",
          f"- 全部 {len(metas)} 个实验的流水与指标对照（含 {n_sup} 个 superseded）: `runs/LEDGER.md`",
          "- 某次实验的完整命令、参数、逐项指标: `runs/exp/<id>/meta.json`",
          "- 重大实验与协议变更的时间线: `docs/HISTORY.md`",
          "- 学长下发的完整规范（CorrDiff 调参表、验收清单）: `docs/reference/instruction.html`",
          "- 集群、分区、环境事实: `docs/reference/ornl-environment.md`", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="只打印, 不写文件")
    a = ap.parse_args()

    metas = load_metas()
    if not metas:
        print("runs/exp/ 下没有带 meta.json 的实验", file=sys.stderr)
        return 1
    status, ledger = build_status(metas), build(metas)
    print(status)
    if not a.print:
        for path, txt in ((STATUS, status), (LEDGER, ledger)):
            with open(path, "w") as f:
                f.write(txt + "\n")
            print(f"-> 已写入 {path}", file=sys.stderr)
        print(f"   ({len(metas)} 个实验)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
