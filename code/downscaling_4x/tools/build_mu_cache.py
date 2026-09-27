#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
build_mu_cache.py — 预计算并缓存 CorrDiff 阶段 A 的回归均值 μ
============================================================================
阶段 B 的残差扩散每训一个 patch 都要用到 μ = E[x|y]。μ 由冻结的阶段 A 网络在**全域**
一次前向得到, 之后才裁块; 在 patch_num=1 的配置下, 每个训练样本都要付一次完整的全域
回归前向, 占单样本成本的绝大部分。

μ 只是 (阶段 A 权重, 某一帧) 的确定性函数 —— 网络冻结、推理态、无数据增强, 因此可以
一次算好存盘反复使用, 与阶段 B 的 patch 尺寸、噪声参数等全部无关。

缓存以**归一化空间**(模型直接输出的空间)保存为 fp16, 按 `<目标>/<年份>.npy` 分文件,
形状 (365, H, W), 可直接 memmap。海洋置零不在此处做, 留给使用方, 使缓存与掩膜策略解耦。

取不到完整历史的帧写 NaN: 阶段 A 本就算不出它们的 μ。阶段 B 与本脚本用同一套 FrameIndex,
正常不会取到; 真取到时 MuCache.get 会抛错, 而不是让 NaN 混进残差。

manifest.json 记录每个阶段 A checkpoint 的 SHA-256 与**完整的条件通道名序列**; 使用方必须
校验后才可信任缓存 —— 阶段 A 重训或条件口径变更后, 旧缓存即失效。口径不符不会引发任何
形状错误: 残差照样算得出来, 阶段 B 照常收敛, 只是训练目标从一开始就是错的。

年份之间互不依赖, 因此在 srun 下按进程切开并行建, 每个进程占一个 GCD、只写自己那一片。
切的只是循环范围: FrameIndex 的 available_years 始终是完整年份集。

用法:
  # 单进程
  python -m downscaling_4x.tools.build_mu_cache --ckpt 2m_temperature_max=runs/exp/<A>/ckpt.pt \\
      --out runs/mu_cache/<id> --mode history_51
  # 一节点 8 进程并行(年份按 SLURM_PROCID 切片)
  srun -n8 python -m downscaling_4x.tools.build_mu_cache --ckpt ... --out ... --mode history_51
  # 补上没建完的年份(口径必须与已有 manifest 完全一致)
  srun -n8 python -m downscaling_4x.tools.build_mu_cache --ckpt ... --out ... --resume
  # 所有进程退出后核对整份缓存是否齐全
  python -m downscaling_4x.tools.build_mu_cache --out ... --check-only
============================================================================
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.training.train_downscale import build_regressor


# 决定 μ 数值的字段。续跑进已有缓存之前这些必须逐项相同, 否则会出现"manifest 记着
# 这一套口径、盘上部分 .npy 却出自另一套"的组合, 而 MuCache.verify() 对它照常放行。
BINDING = ("mode", "cond_layout", "era5_dir", "daymet_dir")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def shard(items, rank, world):
    """把 items 连续均分成 world 份取第 rank 份; 前 len(items) % world 份各多一个。

    连续切分让一个进程拿到相邻年份, 历史回查多落在它已经打开的年度文件上。
    """
    q, r = divmod(len(items), world)
    lo = rank * q + min(rank, r)
    return items[lo:lo + q + (1 if rank < r else 0)]


def usable(path, shape):
    """盘上这一年是不是一份可直接使用的缓存。

    只看文件在不在是不够的: 进程被杀在写盘中途会留下截断的 .npy, 而续跑会把它当成
    "已经建好"跳过 —— 那一年的 μ 是垃圾, 之后没有任何一步会报错。mmap 打开只读文件头
    并按声明的形状建映射, 长度不足或形状不符时当场抛错, 因此截断也一并挡在这里。
    """
    try:
        a = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return a.shape == tuple(shape) and a.dtype == np.float16


def save_atomic(path, arr):
    """先写同目录临时文件再原子改名, 使被中断的写入只留下 .tmp 而非截断的 .npy。

    np.save 只在收到路径时补 .npy 后缀, 收到文件对象时不补; 临时名要留在 .npy 之外
    才不会被续跑当成一个年份文件。
    """
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="预计算 CorrDiff 阶段 A 的 μ 缓存")
    ap.add_argument("--ckpt", action="append", metavar="TARGET=PATH",
                    help="阶段 A checkpoint, 可给多次(每个目标一个)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default=C.DEFAULT_MODE, choices=sorted(C.MODES))
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--years", type=int, nargs="+", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="只补缺的年份, 已在盘且可读的跳过; 要求已有 manifest 的口径与本次完全一致")
    ap.add_argument("--check-only", action="store_true",
                    help="不算 μ, 只核对 manifest 声明的年份是否都在盘上且可读")
    II.add_arg(ap)
    a = ap.parse_args()

    out = Path(a.out)
    shape = (C.DAYS_PER_YEAR, C.HR_SHAPE[0], C.HR_SHAPE[1])

    # 每个建缓存的进程只保证自己那一片, 谁都不知道别的进程跑没跑; 整份缓存是否齐全
    # 只能在所有进程退出之后单独核一次。少了的年份若落在 val/test 上, 训练侧的
    # residual_scale 只读训练年, 不会碰到, 要等到评测才暴露。
    if a.check_only:
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        years = a.years or man["years"]
        want = [(t, y) for t in man["checkpoints"] for y in years]
        bad = [f"{t}/{y}.npy" for t, y in want if not usable(out / t / f"{y}.npy", shape)]
        if bad:
            raise SystemExit(f"[mu-cache] 缓存不完整: 应有 {len(want)} 份, 其中 {len(bad)} 份"
                             f"缺失或不可读, 首个 {bad[0]} -> {out}")
        print(f"[mu-cache] 完整性 OK: {len(man['checkpoints'])} 目标 x {len(years)} 年"
              f" 共 {len(want)} 份全部可读 -> {out}")
        return

    if not a.ckpt:
        raise SystemExit("--ckpt 是必需的(只有 --check-only 例外)")
    pairs = {}
    for item in a.ckpt:
        t, _, path = item.partition("=")
        if t not in C.TARGETS or not path:
            raise SystemExit(f"--ckpt 需写成 目标=路径, 且目标属于 {C.TARGETS}; 收到 {item!r}")
        pairs[t] = os.path.abspath(path)

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    tag = f"rank{rank}/{world} " if world > 1 else ""

    years = a.years or (list(M.splits["train"]) + list(M.splits["val"]))
    if [y for k in range(world) for y in shard(years, k, world)] != years:
        raise SystemExit("[mu-cache] 年份分片没有覆盖且只覆盖一次全部年份; 这是代码缺陷。")
    my_years = shard(years, rank, world)

    if torch.cuda.is_available():
        if local >= torch.cuda.device_count():
            raise SystemExit(f"[mu-cache] {tag}SLURM_LOCALID={local} 超出本进程可见的 GPU 数 "
                             f"{torch.cuda.device_count()}")
        device = torch.device(f"cuda:{local}")
    elif os.environ.get("SLURM_JOB_ID"):
        # 退回 CPU 既不报错、结果也对, 只是慢几十倍从而必然撞上墙钟, 在事件流里与
        # "还在跑"完全一样; 在作业里出现这种情况一定是 GPU 没绑上, 就地停。
        raise SystemExit(f"[mu-cache] {tag}作业内没有可用的 GPU; 拒绝退回 CPU 运行。")
    else:
        device = torch.device("cpu")

    layout = C.cond_layout(a.mode)
    need, lags = C.pairing_history_days(a.mode), C.history_lags(a.mode)

    manifest = {
        "checkpoints": {t: {"path": p, "sha256": sha256(p)} for t, p in pairs.items()},
        "mode": a.mode, "cond_layout": layout,
        "space": "normalized (模型直接输出); 反归一化需乘 d_std 加 d_mean",
        "dtype": "float16", "layout": f"<target>/<year>.npy, 形状 (365, {C.HR_SHAPE[0]}, {C.HR_SHAPE[1]})",
        "ocean": "未做海洋置零; 使用方按需处理",
        "missing_history": "取不到完整历史的帧为 NaN",
        "era5_dir": os.path.abspath(a.era5_dir), "daymet_dir": os.path.abspath(a.daymet_dir),
        "input": II.describe(a.era5_dir),
        "years": years,
    }
    mpath = out / "manifest.json"

    if a.resume:
        if not mpath.exists():
            raise SystemExit(f"[mu-cache] --resume 需要已有 manifest, 但 {mpath} 不存在。")
        old = json.loads(mpath.read_text(encoding="utf-8"))
        diff = [k for k in BINDING if old.get(k) != manifest[k]]
        if ({t: v.get("sha256") for t, v in (old.get("checkpoints") or {}).items()}
                != {t: v["sha256"] for t, v in manifest["checkpoints"].items()}):
            diff.append("checkpoints.sha256")
        if diff:
            raise SystemExit(f"[mu-cache] --resume 但已有缓存的口径与本次不符: {diff}。"
                             "混着建会留下 manifest 记着一套口径、部分 .npy 出自另一套的缓存, "
                             "而 MuCache.verify() 对它照常放行。请改用空目录。")
        if not set(years) <= set(old.get("years") or []):
            raise SystemExit(f"[mu-cache] --resume 但本次年份超出已有 manifest 声明的范围 "
                             f"{min(old.get('years') or [0])}-{max(old.get('years') or [0])}。"
                             "请改用空目录。")
        # manifest 不重写: 口径与年份都已确认一致, 重写只会引入并发写并丢掉原始出处记录。
    else:
        clash = [f"{t}/{y}.npy" for t in pairs for y in my_years
                 if (out / t / f"{y}.npy").exists()]
        if clash:
            raise SystemExit(f"[mu-cache] {tag}{clash[0]} 已存在; 缓存不就地重建。"
                             "请改用空目录, 或加 --resume 只补缺的年份。")
        out.mkdir(parents=True, exist_ok=True)
        # 各进程写出的内容逐字节相同, 原子改名让并发写落到同一份结果上。
        tmp = mpath.with_name(f"manifest.json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, mpath)

    nets = {}
    for t, path in pairs.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cargs = ck.get("args", {})
        II.check(cargs.get("era5_dir"), a.era5_dir, f"{t}: 阶段 A checkpoint {path}",
                 allow=getattr(a, II.ALLOW_DEST, False))
        saved = (ck.get("resume_contract") or {}).get("cond_layout")
        if saved is not None and list(saved) != layout:
            raise SystemExit(f"{t}: 阶段 A checkpoint 的条件布局与 --mode {a.mode} 不符; "
                             f"checkpoint {len(saved)} 通道, 现行 {len(layout)} 通道。")
        # μ 必须由**整幅**前向得到。JiT 的位置编码与 RoPE 是 persistent=False 的 buffer,
        # 所以裁块训练出来的权重能无声地装进整幅模型 —— 跑得起来, 但 token 网格与 RoPE 都
        # 变了, 不是同一个函数。这里按 checkpoint 自述的 crop 拦一道。
        ct = cargs.get("target")
        if ct is not None and ct != t:
            raise SystemExit(f"{t}: checkpoint 自述的目标是 {ct!r}, 与 --ckpt 的键 {t!r} 不符。"
                             "键错了会建出 manifest 记着一个目标、μ 数值出自另一个目标的缓存, "
                             "阶段 B 照样训得下去、也不会报错。")
        if cargs.get("crop"):
            raise SystemExit(f"{t}: 阶段 A 是按 crop={cargs['crop']} 训的(仅冒烟用), "
                             "不能用来建整幅 μ 缓存。")
        n = build_regressor(len(layout), 1, cargs, hw=C.HR_SHAPE).to(device)
        n.load_state_dict(ck["model"])
        n.eval()
        for q in n.parameters():
            q.requires_grad_(False)
        nets[t] = n
        print(f"[mu-cache] {tag}{t}: {path} (arch={cargs.get('arch', 'unet')}) device={device} "
              f"分到 {len(my_years)} 年 {my_years[:1]}..{my_years[-1:]}", flush=True)

    stats = Stats(a.era5_dir, a.daymet_dir)
    for y in my_years:
        if a.resume and all(usable(out / t / f"{y}.npy", shape) for t in pairs):
            print(f"[mu-cache] {tag}{y}: 已在盘且可读, 跳过", flush=True)
            continue
        # available_years 恒为完整年份集: 只给本片会把每片年初取不到上一年的那几帧
        # 判成缺历史而写成 NaN, 缓存照常建完, 阶段 B 直到取到那天才抛错。
        fi = FrameIndex([y], sorted(set(years)), need, lags, split=f"mu{y}")
        ds = DownscaleData(a.era5_dir, a.daymet_dir,
                           sorted({y} | {h[0] for hs in fi.history for h in hs}),
                           stats, mode=a.mode, era5_cache_years=3)
        H, W = ds.H, ds.W
        buf = {t: np.full((C.DAYS_PER_YEAR, H, W), np.nan, np.float16) for t in pairs}
        with torch.no_grad():
            for i, (_, day) in enumerate(fi.frames):
                cond = ds.assemble(ds.cond_fields(y, day, fi.history_of(i)))
                x = torch.from_numpy(cond)[None].to(device)
                for t, n in nets.items():
                    buf[t][day] = n(x)[0, 0].float().cpu().numpy().astype(np.float16)
        for t in pairs:
            (out / t).mkdir(parents=True, exist_ok=True)
            save_atomic(out / t / f"{y}.npy", buf[t])
        print(f"[mu-cache] {tag}{y}: {len(fi)}/{C.DAYS_PER_YEAR} 帧 (丢 {fi.dropped} 帧无完整历史)",
              flush=True)

    bad = [f"{t}/{y}.npy" for t in pairs for y in my_years
           if not usable(out / t / f"{y}.npy", shape)]
    if bad:
        raise SystemExit(f"[mu-cache] {tag}本片建完后仍有不可读的年份: {bad}")
    print(f"[mu-cache] {tag}完成 {len(my_years)} 年 -> {out}", flush=True)


if __name__ == "__main__":
    main()
