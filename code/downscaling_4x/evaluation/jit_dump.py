#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jit_dump.py — JiT 整幅采样落盘(逐日逐像素场, 不做任何指标计算或绘图)

对给定年份的每一天, 用整幅 ODE 采样器采 N 个成员, 只把"可由成员归约出的逐日逐像素
场"落盘, 全部下游聚合与出图交给 evaluation.render 离线完成。布局与 det_dump 一致,
同一 render / dump_metrics 管线直接通吃:

  ens_mean/<年>_d<日>.npy   成员均值          float32  物理单位   (有效域外 NaN)
  crps/<年>_d<日>.npy       逐像素 CRPS       float32  物理单位   (有效域外 NaN)
  crps_log/<年>_d<日>.npy   逐像素 CRPS       float32  log1p(mm)  (仅降水)
  spread/<年>_d<日>.npy     成员标准差        float32  物理单位   (有效域外 NaN)
  rank/<年>_d<日>.npy       真值在成员中的名次 int8    0..members(有效域外 -1)
  members/<年>_d<日>.npy    全部成员           float32  物理单位   (--save-members, 有效域外 NaN)

CRPS 与 rank 依赖全体成员, 采样结束即无法重算, 因此必须在此处算好。--save-members 则把成员
本身留下((M,H,W), 480x960x32 约 56 MiB/天), 此后任何按成员分解的问题都不必重采。成员是无损
记录: 落盘的 float32 与归约场用的是同一份数值, 离线重算 ens_mean/spread/crps 只差浮点归约误差。

--members-only 给已有落场目录补录成员: 不写归约场、不动采样记录, 但逐日把重采成员归约出的
ens_mean / spread / crps 与目录里已有的场对拍, 且本次的 checkpoint 与采样参数必须与该目录 meta
记录的逐项一致。成员与该目录已发布的集合指标是否出自同一次采样, 是一切按成员分解的结论的前提,
而换了 checkpoint 或改了 members/steps/seed/输入产品时, 两边文件都完整、都能读、都不会报错。

单阶段与两阶段一个入口通吃, 按 checkpoint 自述自动分派: args 里带 mu_cache 即为残差
模式 —— 条件拼上域外钉零的 μ(共 52 通道), 采样得到归一化残差 r̂, 重组 y = μ + σ_r·r̂
后走与单阶段完全相同的反变换。μ 缓存按 stage-A checkpoint 的 SHA-256 校验, 拿错缓存
当场拒绝。采样权重默认取 raw(--ema 0)。

种子 = seed*100003 + (年*1000+日)*131 + 成员, 只依赖 (seed, 年, 日, 成员); 平局名次的
随机劈分种子只依赖 (seed, 年, 日)。故场文件与分片无关, 重跑/改分片逐位一致。

多卡: 按天分给 SLURM rank, 各写各的天文件(天互不相交, 无需合并); rank0 写 meta.json。
--finalize: srun 结束后单进程校验各场文件数并把 meta.status 置 done。

路由落盘(--routing-dump, 仅 JiT-MoE): 采样时截获每层的路由决策, 每成员每天一个
routing/<年>_d<日>_m<成员>.npz(被选次数、专家数、门控权重、亲和分、专家输出范数、难度
先验, 见 evaluation.routing_dump)。截获是只读补丁, 场文件与不落路由时逐位相同。
--routing-only 只采样落路由成员、不写场, 用于给已有落场目录补录路由。
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.data.mu_cache import MuCache
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation import routing_dump as RD
from downscaling_4x.models.jit_backbone import draw_patch_offset, token_domain_mask
from downscaling_4x.models.jit_sampler import generate
from downscaling_4x.models.moe_ffn import dec_stats_summary
from downscaling_4x.training.train_jit import build_model

BASE_FIELDS = ("ens_mean", "crps", "spread", "rank")
MEMBERS_SUB = "members"
REDUCED_KEYS = ("ens_mean", "spread", "crps", "crps_log")   # --members-only 逐日对拍的场


def default_days(years, doms=(5, 15, 25)):
    """每月固定日 x 12 月 x 各年 -> [(year, 0基日序), ...]; 365 日历下均匀覆盖季节。"""
    out = []
    for y in years:
        for m in range(1, 13):
            for dom in doms:
                doy = (datetime.date(y, m, dom) - datetime.date(y, 1, 1)).days
                out.append((y, doy))
    return out


def resolve_days(a):
    """显式 --days(年-日序 列表)优先于 years/all-days; 用于冒烟或补采个别日。"""
    if a.days:
        return [(int(s.split("-")[0]), int(s.split("-")[1])) for s in a.days]
    if a.all_days:
        return [(y, t) for y in a.years for t in range(C.DAYS_PER_YEAR)]
    return default_days(a.years)


def fields_for(target):
    return list(BASE_FIELDS) + (["crps_log"] if target == C.PRECIP else [])


def file_sig(path):
    """轻量溯源: 路径/大小/mtime; 不做 sha 以免拖慢作业启动。"""
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def write_meta(out, a, args, ckpt_path, samples, days, residual, dec=None, arch=None):
    target = args["target"]
    meta = {
        "id": Path(out).name,
        "date": datetime.date.today().isoformat(),
        "kind": "jit_sample_dump",
        "target": target,
        "unit": "mm/day" if target == C.PRECIP else "K",
        "mode": args.get("mode", C.DEFAULT_MODE),
        "diffusion_ckpt": file_sig(ckpt_path),
        "run": str(a.run),
        "input": II.describe(a.era5_dir),
        "weight": "raw" if a.ema == 0 else f"ema{a.ema}",
        "trained_samples": int(samples) if samples is not None else -1,
        "members": a.members,
        "steps": a.steps,
        "noise_scale": args["noise_scale"],
        "t_eps": args["t_eps"],
        "patch": args["patch"],
        "seed": a.seed,
        "years": a.years,
        "all_days": bool(a.all_days),
        "n_days_expected": len(days),
        "fields": fields_for(target),
        "members_dump": ({"sub": MEMBERS_SUB, "members": a.members,
                          "layout": "(M,H,W) float32, 成员序号 0..M-1, 与 ens_mean 同单位同掩膜"}
                         if a.save_members else None),
        "stage_b": residual,        # None = 单阶段
        "dec": dec,                 # None = 非 D-EC 路由; 否则含成分开关、λ 与推理期规则
        "arch": arch,               # None = 单流; 否则含两条流开关表与推理期干预(τ 缩放、键置零)
        "mask_note": "有效域 = daymet_land AND era5_valid; 场按 (H,W) 落盘, 域外 NaN",
        "seed_scheme": "seed*100003+(year*1000+day)*131+member",
        "status": "running",
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(Path(out) / "meta.json", "w"), indent=1, ensure_ascii=False)


def finalize(out, target, days):
    """校验各场文件数, 标注缺失日, 把 status 置 done。纯文件系统, 无需 GPU/数据。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    want_names = {f"{y}_d{t}" for y, t in days}
    report, missing = {}, {}
    for fld in fields_for(target):
        have = {fp.stem for fp in (out / fld).glob("*.npy")}
        report[fld] = len(have)
        miss = sorted(want_names - have)
        if miss:
            missing[fld] = miss
    meta["written"] = report
    meta["missing"] = missing
    routing = merge_dec_routing(out)
    if routing is not None:
        meta["dec_routing"] = routing
    meta["status"] = "done" if not missing else "incomplete"
    json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
    tag = "OK" if not missing else f"缺 {sum(len(v) for v in missing.values())} 个场-日"
    print(f"[jit_dump] finalize {out.name}: {report} -> status={meta['status']} ({tag})")


def finalize_routing(out, days):
    """校验路由文件数(成员 × 落路由的日), 写进 routing/meta.json; 纯文件系统。"""
    info = RD.load_routing_meta(out)
    if info is None:
        return None
    rdays = {tuple(d) for d in info["days"]}
    want = {(y, t, m) for (y, t) in days if (y, t) in rdays for m in info["members"]}
    have = set(RD.available_routing(out))
    missing = sorted(want - have)
    info["written"] = len(have & want)
    info["missing"] = [f"{y}_d{t}_m{m}" for y, t, m in missing]
    info["status"] = "done" if not missing else "incomplete"
    RD.write_routing_meta(out, info)
    print(f"[jit_dump] routing {Path(out).name}: {info['written']}/{len(want)} 文件 -> status={info['status']}")
    return info


def merge_dec_routing(out):
    """合并各 rank 落下的 D-EC 路由统计(按域内 token 数加权), 每层一条; 无文件返回 None。"""
    files = sorted(Path(out).glob("dec_routing_rank*.json"))
    if not files:
        return None
    acc = None
    for f in files:
        rows = json.load(open(f))
        if acc is None:
            acc = [dict(k_num=[0.0] * len(r["k_dist"]), n=0) for r in rows]
        for a_, r in zip(acc, rows):
            n = r["n_in_tokens"]
            a_["n"] += n
            a_["k_num"] = [x + n * y for x, y in zip(a_["k_num"], r["k_dist"])]
    out_rows = []
    for a_ in acc:
        n = max(a_["n"], 1)
        dist = [x / n for x in a_["k_num"]]
        out_rows.append({"k_dist": [round(x, 4) for x in dist],
                         "mean_k": round(sum(i * x for i, x in enumerate(dist)), 4),
                         "frac_k0": round(dist[0], 4), "n_in_tokens": int(a_["n"])})
    return {"eval_rule": "帧内 top-C 乘容量系数; 域内 token 的专家数分布, 每层一条",
            "layers": out_rows}


def members_field(mem_phys, land):
    """(M,H,W) float32 物理单位, 有效域外 NaN; 与 ens_mean 等场同一空间、同一掩膜口径。"""
    return np.where(land, mem_phys, np.nan).astype(np.float32)


def routing_offset(out, y, day, m):
    """已落盘路由里记的该成员切块起点 (dy, dx); 没有该文件返回 None。"""
    f = Path(out) / "routing" / f"{y}_d{day}_m{m}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        return tuple(int(v) for v in z["offset"])


def _norm(v):
    """json 往返归一化, 用于把 meta 里读回的值与本次的值放在同一表示下比较。"""
    return json.loads(json.dumps(v, ensure_ascii=False, sort_keys=True))


def check_members_dir(out, a, args, ckpt_path, samples, target, dec_info):
    """--members-only 的身份守卫: 本次采样配置须与该落场目录 meta 记录的逐项一致, 否则拒绝。

    补录的成员只有与目录里已有的归约场同源才有意义; 而换 checkpoint、改 members/steps/seed
    或换输入产品时, 成员照样写满 365 天, 两边文件都完整、都能读, 没有任何东西会报错。
    """
    mp = Path(out) / "meta.json"
    if not mp.exists():
        raise SystemExit(f"--members-only 要求 {out} 已是完整落场目录, 但其中没有 meta.json")
    m = json.load(open(mp))
    want = {"kind": "jit_sample_dump", "target": target, "members": a.members, "steps": a.steps,
            "seed": a.seed, "weight": "raw" if a.ema == 0 else f"ema{a.ema}",
            "trained_samples": int(samples) if samples is not None else -1,
            "mode": args.get("mode", C.DEFAULT_MODE), "noise_scale": args["noise_scale"],
            "t_eps": args["t_eps"], "patch": args["patch"],
            "dec": dec_info, "input": II.describe(a.era5_dir)}
    bad, unchecked = {}, []
    for k, v in want.items():
        if k in m:
            if _norm(m[k]) != _norm(v):
                bad[k] = (m[k], v)
        elif v is not None:
            unchecked.append(k)          # 早于该字段的老落场没记这一项, 无从核对
    new_ck = file_sig(ckpt_path)
    if "diffusion_ckpt" in m:
        old_ck = m["diffusion_ckpt"] or {}
        if (old_ck.get("path"), old_ck.get("bytes")) != (new_ck["path"], new_ck["bytes"]):
            bad["diffusion_ckpt"] = (old_ck, new_ck)
    else:
        unchecked.append("diffusion_ckpt")
    missing = [sub for sub in BASE_FIELDS if not (Path(out) / sub).is_dir()]
    if missing:
        raise SystemExit(f"--members-only 要求已有归约场用于对拍, 但 {Path(out).name} 缺 {missing}")
    if bad:
        lines = "\n".join(f"    {k}: 落场记录 {o!r} != 本次 {n!r}" for k, (o, n) in sorted(bad.items()))
        raise SystemExit(f"--members-only 身份守卫拒绝: 本次采样与 {Path(out).name} 的记录不一致\n{lines}")
    if unchecked:
        print(f"[jit_dump] 注意: {Path(out).name} 的 meta 没记 {sorted(unchecked)}, 这几项无从核对; "
              f"输入产品另由 checkpoint 自述的 era5_dir 守卫把关", flush=True)
    return m


def compare_reduced(out, y, day, land, ens_mean, spread, crps_field, rank_field, crps_log_field):
    """把重采成员归约出的场与该目录已有的同名场逐项比对, 返回各项在有效域内的最大绝对差。

    rank 是整数场, 只报不一致像素占比: 名次对微小数值差不敏感, 与浮点差互为补充。
    """
    out = Path(out)
    res = {"year": int(y), "day": int(day)}
    pairs = [("ens_mean", ens_mean), ("spread", spread), ("crps", crps_field)]
    if crps_log_field is not None:
        pairs.append(("crps_log", crps_log_field))
    for sub, new in pairs:
        f = out / sub / f"{y}_d{day}.npy"
        if not f.exists():
            raise RuntimeError(f"--members-only: 已有落场缺 {sub}/{y}_d{day}.npy, 无法证明成员与它同源")
        res[sub] = float(np.nanmax(np.abs(new[land].astype(np.float64) - np.load(f)[land])))
    rf = out / "rank" / f"{y}_d{day}.npy"
    if rf.exists():
        res["rank_mismatch_frac"] = float((np.load(rf)[land] != rank_field[land]).mean())
    return res


def merge_members_check(out):
    """合并各 rank 的对拍记录, 各项取最大值; 无记录返回 None。"""
    d = Path(out) / "members_check"
    files = sorted(d.glob("rank*.json")) if d.is_dir() else []
    if not files:
        return None
    rows = [r for f in files for r in json.load(open(f))]
    keys = [k for k in REDUCED_KEYS if any(k in r for r in rows)]
    worst = max(rows, key=lambda r: r.get("ens_mean", 0.0))
    return {"n_days": len(rows),
            "max_abs": {k: max(r[k] for r in rows if k in r) for k in keys},
            "rank_mismatch_frac_max": max(r.get("rank_mismatch_frac", 0.0) for r in rows),
            "routing_offset_checked": sum(r.get("routing_offset_checked", 0) for r in rows),
            "worst_day": f"{worst['year']}_d{worst['day']}",
            "note": "重采成员归约出的场与已有场的最大绝对差(物理单位); 成员与既有指标同源的凭据"}


def finalize_members(out, days, members):
    """校验成员场文件数与对拍结果, 写进 meta 的 members_dump 段; 纯文件系统, 无需 GPU/数据。"""
    out = Path(out)
    mp = out / "meta.json"
    meta = json.load(open(mp)) if mp.exists() else {}
    rec = meta.get("members_dump") or {"sub": MEMBERS_SUB, "members": members,
                                       "layout": "(M,H,W) float32, 成员序号 0..M-1, 与 ens_mean 同单位同掩膜"}
    have = {fp.stem for fp in (out / MEMBERS_SUB).glob("*.npy")} if (out / MEMBERS_SUB).is_dir() else set()
    miss = sorted({f"{y}_d{t}" for y, t in days} - have)
    rec.update(written=len(have), missing=miss, status="done" if not miss else "incomplete")
    chk = merge_members_check(out)
    if chk is not None:
        rec["check"] = chk
    meta["members_dump"] = rec
    json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)
    print(f"[jit_dump] members {out.name}: {len(have)}/{len(days)} 文件 -> status={rec['status']}"
          + (f"; 与已有场最大偏差 {chk['max_abs']}" if chk else ""))
    return rec


def save_field(out, sub, y, day, arr):
    d = Path(out) / sub
    d.mkdir(exist_ok=True)
    tmp = d / f".{y}_d{day}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{y}_d{day}.npy")   # 原子改名: 存在即完整


def main():
    ap = argparse.ArgumentParser(description="JiT 整幅采样落盘(逐日逐像素场, 不出图)")
    ap.add_argument("--run", required=True, help="JiT 训练 run 目录, 取其 checkpoint")
    ap.add_argument("--which", choices=["ckpt", "last"], default="ckpt",
                    help="ckpt=best-val(默认) / last=末端")
    ap.add_argument("--ema", type=int, choices=[0, 1, 2], default=0,
                    help="采样权重: 0=raw(默认, 家族已定档) / 1,2=EMA")
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--all-days", action="store_true",
                    help="取 years 内全部日(365 日历/年); 缺省用每月 5/15/25 抽样")
    ap.add_argument("--days", nargs="+", default=None,
                    help='指定 "年-日序" 列表(如 2020-0 2020-100), 覆盖 years/all-days')
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dec-eval-prior", type=int, choices=[0, 1], default=1,
                    help="推理期是否让难度先验进选择分; 0 = 关(权重不变, 只改这一次采样的路由决策)")
    ap.add_argument("--dec-eval-drop", type=int, choices=[0, 1], default=1,
                    help="推理期是否把有效域外 token 排除在路由之外; 0 = 关, 域外 token 与域内一起竞争容量")
    ap.add_argument("--dec-capacity", type=float, default=1.0,
                    help="[D-EC] 推理期容量系数: 每专家取 ⌈n_in·K·系数/E⌉ 个域内 token(帧内规则, "
                         "与 batch 组成无关); 非 D-EC 模型上给非 1 的值会被拒绝")
    ap.add_argument("--tau-scale", type=float, default=1.0,
                    help="[两条流] 推理期把按风推移的 τ 统一乘此系数(0 = 全部头不推); 训练时为 1, 改动即干预, 记进 meta")
    ap.add_argument("--zero-keys", action="store_true",
                    help="[两条流] 推理期把科室键与地形键的投影置零(专家分工退回纯内容); 属干预, 记进 meta")
    ap.add_argument("--routing-dump", choices=["none", "member0", "all"], default="none",
                    help="[MoE] 把采样时每层的路由决策落到 <out>/routing/: none=不落(缺省, 产物逐字节不变) / "
                         "member0=只落成员 0 / all=全部成员")
    ap.add_argument("--routing-days", nargs="+", default=None,
                    help='[MoE] 只在这些 "年-日序" 落路由(缺省 = 本次采样的全部日); 配合 all 用于 case study 日期')
    ap.add_argument("--routing-scores", action=argparse.BooleanOptionalAction, default=True,
                    help="[MoE] 路由落盘时附带按 t 档的亲和分之和(每成员每天约 1.8 MB)")
    ap.add_argument("--routing-norms", action=argparse.BooleanOptionalAction, default=True,
                    help="[MoE] 路由落盘时附带专家输出范数占 token 隐状态范数的比例之和")
    ap.add_argument("--save-members", action="store_true",
                    help="把全部成员的逐日场落到 <out>/members/: (M,H,W) float32 物理单位, 域外 NaN; "
                         "480x960x32 约 56 MiB/天, 365 天约 21 GiB")
    ap.add_argument("--members-only", action="store_true",
                    help="只给已有落场目录补录成员场: 不写归约场、不动采样记录; 逐日用重采成员归约出的 "
                         "ens_mean/spread/crps 与已有场对拍, 且采样配置须与该目录 meta 逐项一致")
    ap.add_argument("--members-tol", type=float, default=1e-2,
                    help="[--members-only] 与已有场的最大允许偏差(物理单位); 超过即判定不是同一次采样")
    ap.add_argument("--routing-only", action="store_true",
                    help="[MoE] 只采样落路由成员、不写场; 给已有落场目录补录路由时用, 已有场文件不动; "
                         "若成员集合与已有落场相同, 会用重采得到的集合均值与已有场对拍")
    II.add_arg(ap)
    ap.add_argument("--finalize", action="store_true",
                    help="srun 后单进程校验各场文件数并标注 status; 不采样")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.members_only and not a.save_members:
        raise SystemExit("--members-only 需要同时给 --save-members")
    if a.members_only and a.routing_only:
        raise SystemExit("--members-only 与 --routing-only 不能同用: 各自只写自己那一份产物")

    out = Path(a.out)
    ckpt_path = Path(a.run) / f"{a.which}.pt"
    days = resolve_days(a)

    if a.finalize:
        mp = out / "meta.json"
        if mp.exists():
            target = json.load(open(mp))["target"]
        else:
            target = torch.load(ckpt_path, map_location="cpu",
                                weights_only=False)["args"]["target"]
        if not a.routing_only and not a.members_only:
            finalize(out, target, days)
        if a.save_members:
            finalize_members(out, days, a.members)
        if not a.members_only:                 # 补录成员不写路由, 也就不去动路由的记录
            finalize_routing(out, days)
        return

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    II.check(args.get("era5_dir"), a.era5_dir, f"checkpoint {ckpt_path}",
             allow=getattr(a, II.ALLOW_DEST, False))
    target = args["target"]
    ti = C.TARGETS.index(target)
    is_precip = target == C.PRECIP
    mode = args.get("mode", C.DEFAULT_MODE)

    # ---- 残差模式按 checkpoint 自述分派 ----
    mu_cache, sigma_r, residual_meta = None, 1.0, None
    if args.get("mu_cache"):
        mu_cache = MuCache(args["mu_cache"], [target])
        if args.get("stage_a_ckpt"):
            mu_cache.verify({target: args["stage_a_ckpt"]})
        II.check(mu_cache.manifest.get("era5_dir"), a.era5_dir, f"μ 缓存 {args['mu_cache']}",
                 allow=getattr(a, II.ALLOW_DEST, False))
        rs = args.get("residual_scale", "")
        if Path(str(rs)).is_file():
            sigma_r = float(json.loads(Path(rs).read_text())["residual_std"])
        else:
            sigma_r = float(rs)
        assert sigma_r > 0, f"σ_r 必须为正, 得到 {sigma_r}"
        residual_meta = {"mu_cache": str(args["mu_cache"]), "sigma_r": sigma_r,
                         "stage_a_ckpt": str(args.get("stage_a_ckpt", "")),
                         "recon": "y_norm = μ·land + σ_r·r̂·land"}

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    years_all = sorted({y for y, _ in days})
    avail = sorted({y - 1 for y in years_all} | set(years_all)) if need else years_all
    fi = FrameIndex(years_all, avail, need, lags, split="dump")
    frame_of = {f: k for k, f in enumerate(fi.frames)}
    skipped = [d for d in days if d not in frame_of]
    if skipped:
        print(f"[jit_dump] {len(skipped)} 天因缺完整历史被跳过: {skipped[:5]}...", flush=True)
        days = [d for d in days if d in frame_of]
    ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
    dd = DownscaleData(a.era5_dir, a.daymet_dir, ds_years, stats, mode=mode,
                       era5_cache_years=len(ds_years))
    H, W = dd.H, dd.W
    land = dd.mask

    net = build_model(args, (H, W))
    sd = dict(ck["model"])
    if a.ema:
        for n, v in ck[f"ema{a.ema}"].items():
            sd[n] = v                      # 参数用 EMA, buffer(含路由偏置)保持 model 值
    net.load_state_dict(sd)
    net = net.to(device).eval()
    dec_info = None
    if getattr(net, "dec", None) is not None:
        # 先验系数按 checkpoint 记录的训练进度复原(跑满即上限); 推理规则固定为帧内 top-C
        frac = float(ck.get("samples") or 0) / max(1.0, float(args.get("duration", 1)))
        lam = net.set_dec_progress(frac)
        net.set_dec_eval("frame", a.dec_capacity)
        # 推理期成分开关: route_dec 每次调用都读这两个属性, 因此关掉只影响本次采样的路由决策,
        # 权重一个字节都不动。训练时开着而推理时关掉是一次干预, 不是复现该模型的正常用法,
        # 所以要进落场 meta —— 否则这份落场与正常落场在目录里看不出区别。
        if not a.dec_eval_prior or not a.dec_eval_drop:
            for m_ in net.moe_layers():
                m_.dec_prior = bool(a.dec_eval_prior) and m_.dec_prior
                m_.dec_drop = bool(a.dec_eval_drop) and m_.dec_drop
        dec_info = {**net.dec, "lambda": round(lam, 4), "eval_mode": "frame",
                    "capacity": a.dec_capacity,
                    "eval_prior": bool(a.dec_eval_prior), "eval_drop": bool(a.dec_eval_drop),
                    "eval_intervention": (not a.dec_eval_prior) or (not a.dec_eval_drop)
                                         or a.dec_capacity != 1.0}
    elif a.dec_capacity != 1.0 or not a.dec_eval_prior or not a.dec_eval_drop:
        raise SystemExit("--dec-capacity / --dec-eval-prior / --dec-eval-drop 只对 D-EC 模型有意义, "
                         "该 checkpoint 的路由不是 dec")
    # ---- 两条流的推理期干预: 只改本次采样的路由/位置规则, 权重一个字节不动, 全部记进 meta ----
    arch_info = None
    if getattr(net, "arch", None):
        arch_info = {**net.arch, "tau_scale": a.tau_scale, "zero_keys": bool(a.zero_keys),
                     "intervention": (a.tau_scale != 1.0) or bool(a.zero_keys)}
        if a.tau_scale != 1.0:
            if not net.lagrangian:
                raise SystemExit("--tau-scale 只对按风推移(lagrangian=1)的模型有意义")
            net.tau_spec = [(lev, hours * a.tau_scale) for lev, hours in net.tau_spec]
        if a.zero_keys:
            n_zero = 0
            for m_ in net.moe_layers():
                if m_.ward_proj is not None or m_.terrain_proj is not None:
                    m_.zero_key_projections(); n_zero += 1
            if n_zero == 0:
                raise SystemExit("--zero-keys: 该模型没有科室键/地形键投影")
    elif a.tau_scale != 1.0 or a.zero_keys:
        raise SystemExit("--tau-scale / --zero-keys 只对两条流 checkpoint 有意义, 该模型是单流")
    if a.members_only:
        check_members_dir(out, a, args, ckpt_path, ck.get("samples"), target, dec_info)

    # ---- 路由落盘: 只对 MoE 模型有意义; 截获是只读补丁, 采样数值不受影响 ----
    moe_layers = net.moe_layers()
    routing_on = a.routing_dump != "none"
    if (routing_on or a.routing_only) and not moe_layers:
        raise SystemExit("--routing-dump / --routing-only 只对 JiT-MoE checkpoint 有意义, 该模型没有 MoE 层")
    if a.routing_only and not routing_on:
        raise SystemExit("--routing-only 需要同时给 --routing-dump member0|all")
    routing_members = list(range(a.members)) if a.routing_dump == "all" else ([0] if routing_on else [])
    routing_days = (set((int(s.split("-")[0]), int(s.split("-")[1])) for s in a.routing_days)
                    if a.routing_days else set(days))
    routing_info = None
    cap = None
    if routing_on:
        L, E, K = len(moe_layers), moe_layers[0].n_experts, moe_layers[0].top_k
        gh, gw = net.x_embedder.gh, net.x_embedder.gw
        cap = RD.RoutingCapture(net, norms=a.routing_norms).install()
        routing_info = {
            "members": routing_members, "days": sorted(routing_days & set(days)),
            "scores": bool(a.routing_scores), "norms": bool(a.routing_norms), "routing_only": bool(a.routing_only),
            "router": cap.mode, "n_moe_layers": L, "moe_block_indices": [i for i, blk in enumerate(net.blocks)
                                                                         if isinstance(blk.mlp, type(moe_layers[0]))],
            "n_experts": E, "top_k": K, "token_grid": [gh, gw], "patch": int(net.patch), "grid_hw": list(net.grid_hw),
            "t_bins": RD.T_BINS.tolist(), "steps": a.steps, "nfwd_per_member": 2 * (a.steps - 1) + 1,
            "dec": dec_info,
            "fields": {"offset": "(2,) int16 切块起点 (dy, dx)", "tok_in": "(T,) bool 域内 token",
                       "sel_cnt": "(L, T, E) uint8 被选前向次数", "sel_cnt_t": "(L, nb, T, E) uint8 按 t 档",
                       "k_sum": "(L, T) uint16 专家数之和", "k0_cnt": "(L, T) uint8 零专家的前向次数",
                       "gate_sum": "(L, T, E) float16 门控权重之和", "score_sum_t": "(L, nb, T, E) float16 亲和分之和",
                       "norm_sum": "(L, T, E) float16 ‖w·f_e(x)‖/‖x‖ 之和",
                       "norm_sum_t": "(L, nb, T, E) float16 同上, 按噪声档分开", "prior_sum_t": "(nb, T) float16 难度先验之和",
                       "nfwd_t": "(nb,) uint16 各 t 档前向数", "nfwd": "前向总数"},
            "selfcheck": "TC: 每 token 每次前向 K 次选中; D-EC: 截获的域内专家数直方图与模型 dec_acc 逐元素相等",
        }
        khist_cap = torch.zeros(L, E + 1, dtype=torch.float64, device=device)   # D-EC 自检用的截获直方图

    out.mkdir(parents=True, exist_ok=True)
    mine = days[rank::ntasks]
    if rank == 0:
        if not a.routing_only and not a.members_only:
            write_meta(out, a, args, ckpt_path, ck.get("samples"), days, residual_meta, dec_info, arch_info)
        if routing_info is not None:
            RD.write_routing_meta(out, {**routing_info, "run": str(a.run), "diffusion_ckpt": file_sig(ckpt_path),
                                        "seed": a.seed, "seed_scheme": "seed*100003+(year*1000+day)*131+member"})

    weight = "raw" if a.ema == 0 else f"ema{a.ema}"
    what = ("members/ 补录(归约场只读对拍)" if a.members_only
            else f"场={fields_for(target)}" + (" + members/" if a.save_members else ""))
    print(f"[jit_dump] rank {rank}/{ntasks} device={device} weight={weight} target={target} "
          f"members={a.members} steps={a.steps} "
          f"{'残差模式 σ_r=%.5f' % sigma_r if mu_cache is not None else '单阶段'} "
          f"分到 {len(mine)} 天; {what}", flush=True)
    checks = []

    land_t = torch.from_numpy(land.astype(np.float32)[None, None]).to(device)
    for y, day in mine:
        t0 = time.time()
        cond, _tgt, _mask, hr = dd.full(y, day, fi.history_of(frame_of[(y, day)]))
        truth = hr[ti].astype(np.float64)
        if is_precip:
            truth = truth * stats.precip_scale
        cond_t = torch.from_numpy(cond[None]).float().to(device)
        mu_np = None
        if mu_cache is not None:
            mu_np = mu_cache.get(target, y, day)
            if not np.isfinite(mu_np[land]).all():
                raise RuntimeError(f"μ 缓存 {y}-d{day} 在有效域内含非有限值")
            mu_t = torch.from_numpy(mu_np[None, None]).float().to(device) * land_t
            cond_t = torch.cat([cond_t, mu_t], dim=1)

        members_norm = []
        off_seen = 0
        record_day = routing_on and (y, day) in routing_days
        for m in (routing_members if a.routing_only else range(a.members)):
            g = torch.Generator(device=device)
            g.manual_seed(a.seed * 100003 + (y * 1000 + day) * 131 + m)
            acc, hooks = None, []
            if routing_on:
                # 切块起点在此预抽: 与 generate 内部的抽取顺序相同, 轨迹逐位不变; 落盘要记它
                offset = draw_patch_offset(net.patch, device, g)
                dy, dx = int(offset[0]) % net.patch, int(offset[1]) % net.patch
                tok_in = token_domain_mask(land_t, net.patch, (dy, dx), net.grid_hw).reshape(-1)
                if record_day and m in routing_members:
                    acc = RD.RoutingAccumulator(len(moe_layers), tok_in.numel(), moe_layers[0].n_experts, device,
                                                scores=a.routing_scores, norms=a.routing_norms)
                cap.acc = acc
                valid_hist = tok_in if (dec_info is not None and net.dec["drop"]) else torch.ones_like(tok_in)

                def _after(_m, _args, _out):
                    t = float(_args[1][0])
                    if acc is not None:
                        acc.add_forward(t, cap)
                    if dec_info is not None:
                        for l in range(len(moe_layers)):
                            kv = cap.sel[l][valid_hist].sum(1)
                            khist_cap[l] += torch.bincount(kv, minlength=khist_cap.shape[1])[:khist_cap.shape[1]].double()
                hooks.append(net.register_forward_hook(_after))
                gen_kw = {"offset": (dy, dx)}
            elif a.members_only:
                # 与落路由时同一条抽取顺序(先起点后噪声), 轨迹逐位不变; 抽到的起点拿去与该成员
                # 已落盘的路由核对 —— 成员与路由同属一条轨迹, 是把两者放进同一张图的前提
                dy, dx = draw_patch_offset(net.patch, device, g)
                rec_off = routing_offset(out, y, day, m)
                if rec_off is not None:
                    off_seen += 1
                    if rec_off != (dy, dx):
                        raise RuntimeError(f"{y}-d{day} 成员 {m}: 本次切块起点 {(dy, dx)} 与已落盘路由记的 "
                                           f"{rec_off} 不同, 成员与路由不是同一条轨迹")
                gen_kw = {"offset": (dy, dx)}
            else:
                gen_kw = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device != "cpu")):
                s = generate(net, cond_t, steps=a.steps, noise_scale=args["noise_scale"],
                             t_eps=args["t_eps"], land=land_t, generator=g, **gen_kw)
            for h in hooks:
                h.remove()
            if acc is not None:
                if acc.nfwd != 2 * (a.steps - 1) + 1:
                    raise RuntimeError(f"路由累加的前向次数 {acc.nfwd} 与采样器 NFE {2 * (a.steps - 1) + 1} 不符")
                acc.check(cap.mode, moe_layers[0].top_k)
                RD.save_routing(out, y, day, m, acc.to_numpy((dy, dx), tok_in.cpu().numpy()))
                cap.acc = None
            r = (s[0, 0].float() * land_t[0, 0]).cpu().numpy()
            if mu_cache is not None:
                r = mu_np * land + sigma_r * r     # 残差重组回归一化目标空间
            members_norm.append(r)
        members_norm = np.stack(members_norm, 0)              # (M,H,W) 归一化空间
        if a.routing_only:
            # 补录模式: 不写场; 成员集合与已有落场相同时, 用重采的集合均值对拍已有场, 证明路由对应同一次采样
            existing = out / "ens_mean" / f"{y}_d{day}.npy"
            if existing.exists() and len(routing_members) == a.members:
                mem_chk = members_norm * stats.d_std[ti] + stats.d_mean[ti]
                if is_precip and stats.precip_log:
                    mem_chk = C.precip_inv(mem_chk, stats.precip_scale) * stats.precip_scale
                    mem_chk = np.where(mem_chk < stats.precip_clip, 0.0, mem_chk)
                old = np.load(existing)
                diff = float(np.nanmax(np.abs(mem_chk.mean(0)[land] - old[land])))
                if diff > 1e-2:
                    raise RuntimeError(f"{y}-d{day}: 重采集合均值与已有场最大差 {diff:.4g}, 路由与落场不是同一次采样")
                print(f"  [rank{rank}] {y}-d{day} 路由补录完成, 集合均值对拍最大差 {diff:.2e} {time.time() - t0:.0f}s", flush=True)
            else:
                print(f"  [rank{rank}] {y}-d{day} 路由补录完成(成员 {routing_members}, 未对拍场) {time.time() - t0:.0f}s", flush=True)
            continue

        if is_precip and stats.precip_log:
            mem_phys = C.precip_inv(members_norm * stats.d_std[ti] + stats.d_mean[ti],
                                    stats.precip_scale) * stats.precip_scale
            # 融合后 drizzle 截断, 与预处理同口径, 恢复零质量
            mem_phys = np.where(mem_phys < stats.precip_clip, 0.0, mem_phys)
        else:
            mem_phys = members_norm * stats.d_std[ti] + stats.d_mean[ti]

        ens_mean = mem_phys.mean(0)
        spread = mem_phys.std(0)

        _, crps_px = MT.crps_ensemble(mem_phys[:, None], truth[None], land, per_pixel=True)
        crps_field = np.where(land, crps_px[0], np.nan).astype(np.float32)
        crps_log_field = None
        if is_precip:
            mem_log = MT.precip_log_mm(mem_phys, stats.precip_clip)     # 与指标、统计基线同一条变换
            truth_log = MT.precip_log_mm(truth, stats.precip_clip)
            _, crps_log_px = MT.crps_ensemble(mem_log[:, None], truth_log[None], land,
                                              per_pixel=True)
            crps_log_field = np.where(land, crps_log_px[0], np.nan).astype(np.float32)

        # rank: 真值在成员中的名次, 平局均匀劈分(种子只依赖 seed/年/日)
        mem_l = mem_phys[:, land]
        tr_l = truth[land]
        ties = (mem_l == tr_l[None]).sum(0)
        tie_rng = np.random.default_rng(a.seed * 100003 + (y * 1000 + day) * 131 + 7)
        ranks_l = (mem_l < tr_l[None]).sum(0) + tie_rng.integers(0, ties + 1)
        rank_field = np.full((H, W), -1, np.int8)
        rank_field[land] = np.clip(ranks_l, 0, a.members).astype(np.int8)

        if a.members_only:
            chk = compare_reduced(out, y, day, land, ens_mean, spread, crps_field,
                                  rank_field, crps_log_field)
            chk["routing_offset_checked"] = off_seen
            tag = " ".join(f"{k}={chk[k]:.2e}" for k in REDUCED_KEYS if k in chk)
            worst = max(chk[k] for k in REDUCED_KEYS if k in chk)
            if worst > a.members_tol:
                raise RuntimeError(f"{y}-d{day}: 重采成员归约出的场与已有场最大偏差 {worst:.4g} > "
                                   f"--members-tol {a.members_tol:.4g} ({tag}); 成员与该落场不是同一次采样")
            checks.append(chk)
            save_field(out, MEMBERS_SUB, y, day, members_field(mem_phys, land))
            print(f"  [rank{rank}] {y}-d{day} 成员补录完成, 对拍 {tag} "
                  f"rank_diff={chk.get('rank_mismatch_frac', 0.0):.2e} "
                  f"路由起点核对 {off_seen}/{a.members} {time.time() - t0:.0f}s", flush=True)
            continue

        save_field(out, "ens_mean", y, day, np.where(land, ens_mean, np.nan).astype(np.float32))
        save_field(out, "spread", y, day, np.where(land, spread, np.nan).astype(np.float32))
        save_field(out, "crps", y, day, crps_field)
        save_field(out, "rank", y, day, rank_field)
        if crps_log_field is not None:
            save_field(out, "crps_log", y, day, crps_log_field)
        if a.save_members:
            save_field(out, MEMBERS_SUB, y, day, members_field(mem_phys, land))

        print(f"  [rank{rank}] {y}-d{day} 落场完成 {time.time() - t0:.0f}s", flush=True)

    if a.members_only and checks:
        cd = out / "members_check"
        cd.mkdir(exist_ok=True)
        json.dump(checks, open(cd / f"rank{rank}.json", "w"), indent=1)
        worst = {k: max(c[k] for c in checks if k in c)
                 for k in REDUCED_KEYS + ("rank_mismatch_frac",) if any(k in c for c in checks)}
        print(f"[jit_dump] rank {rank} 成员补录 {len(checks)} 天, 与已有场最大偏差 "
              + " ".join(f"{k}={v:.2e}" for k, v in worst.items()), flush=True)

    if dec_info is not None:
        if routing_on and mine:
            # 截获看到的路由必须就是模型执行的路由: 域内专家数直方图逐元素相等
            E = moe_layers[0].n_experts
            for l, m_ in enumerate(moe_layers):
                got, ref = khist_cap[l].cpu(), m_.dec_acc[:E + 1].double().cpu()
                if not torch.equal(got, ref):
                    raise RuntimeError(f"D-EC 路由自检失败: 层 {l} 截获直方图 {got.tolist()} 与模型 dec_acc {ref.tolist()} 不等")
            print(f"[jit_dump] rank {rank} D-EC 路由自检通过: {len(moe_layers)} 层截获直方图与模型统计逐元素相等", flush=True)
        v = net.pop_dec_stats()
        E = net.moe_layers()[0].n_experts
        rows = [dec_stats_summary(row.cpu(), E) for row in v]
        if not a.routing_only:
            json.dump(rows, open(out / f"dec_routing_rank{rank}.json", "w"), indent=1)
        print(f"[jit_dump] rank {rank} D-EC 域内 token 平均专家数(逐层): "
              f"{[r['mean_k'] for r in rows]}", flush=True)
    if cap is not None:
        cap.uninstall()
    print(f"[jit_dump] rank {rank} 完成 {len(mine)} 天", flush=True)
    if ntasks == 1:
        if not a.routing_only and not a.members_only:
            finalize(out, target, days)
        if a.save_members:
            finalize_members(out, days, a.members)
        if routing_on:
            finalize_routing(out, days)


if __name__ == "__main__":
    main()
