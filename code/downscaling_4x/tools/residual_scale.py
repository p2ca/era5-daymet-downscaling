#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
residual_scale.py — 量出阶段 B 的残差尺度 σ_r
============================================================================
残差 r = y − μ 在归一化目标空间里的标准差, 只在**有效域**上统计(域外 μ 未被监督, 残差
按约定恒为 0, 计入会把 σ_r 拉低)。

为什么必须量而不是拍脑袋填: μ 会解释掉目标的大部分方差, 残差的标准差远小于 1。而扩散
侧的噪声调度是按"数据方差约为 1"调好的 —— CorrDiff 的 `--sigma-data` 与 JiT 的 t 采样
分布都是。直接拿原始残差去训, 整条信噪比曲线都偏掉, **不会报错, 只是训不好**, 而且看
loss 曲线分辨不出来是尺度问题还是别的。

输出 json 记录 σ_r、均值、样本数与所用的 μ 缓存与年份, 供阶段 B 以参数形式引用。

用法:
  python -m downscaling_4x.tools.residual_scale --cache runs/mu_cache/<id> \\
      --target 2m_temperature_max --out runs/mu_cache/<id>/residual_scale.json
============================================================================
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.data.mu_cache import MuCache
from downscaling_4x.evaluation import input_identity as II


def main():
    ap = argparse.ArgumentParser(description="量出阶段 B 的残差尺度")
    ap.add_argument("--cache", required=True, help="μ 缓存目录")
    ap.add_argument("--target", required=True, choices=C.TARGETS)
    ap.add_argument("--stage-a-ckpt", default="", help="给了就顺带做 SHA 校验")
    ap.add_argument("--era5-dir", default=M.ERA5_DIR)
    ap.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="缺省用训练年; 只应在训练年上量, 免得把验证信息带进训练设置")
    ap.add_argument("--stride", type=int, default=1, help="按天抽样; >1 时结果标为抽样估计")
    ap.add_argument("--out", required=True)
    II.add_arg(ap)
    a = ap.parse_args()

    cache = MuCache(a.cache, [a.target])
    if a.stage_a_ckpt:
        cache.verify({a.target: a.stage_a_ckpt})
    II.check(cache.manifest.get("era5_dir"), a.era5_dir, f"μ 缓存 {a.cache}",
             allow=getattr(a, II.ALLOW_DEST, False))
    mode = cache.manifest.get("mode") or C.DEFAULT_MODE
    years = a.years or list(M.splits["train"])
    missing = [y for y in years if y not in cache.years()]
    if missing:
        raise SystemExit(f"μ 缓存未覆盖年份 {missing}")

    stats = Stats(a.era5_dir, a.daymet_dir)
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    fi = FrameIndex(years, sorted(set(years)), need, lags, split="resid")
    ds = DownscaleData(a.era5_dir, a.daymet_dir, years, stats, mode=mode,
                       era5_cache_years=1)
    land = ds.mask
    ti = C.TARGETS.index(a.target)

    n = 0.0
    s1 = s2 = 0.0
    lo, hi = np.inf, -np.inf
    for k, (y, day) in enumerate(fi.frames):
        if k % a.stride:
            continue
        norm, _ = ds.target(y, day)
        r = norm[ti][land] - cache.get(a.target, y, day)[land]
        n += r.size
        s1 += float(r.sum()); s2 += float(np.square(r, dtype=np.float64).sum())
        lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
        if k % (200 * a.stride) == 0:
            print(f"  {k}/{len(fi)} 帧", flush=True)

    mean = s1 / n
    var = max(s2 / n - mean * mean, 0.0)
    res = {"target": a.target, "mode": mode, "cache": os.path.abspath(a.cache),
           "years": years, "stride": a.stride,
           "input": II.describe(a.era5_dir),
           "domain": C.EFFECTIVE_DOMAIN, "n_values": int(n),
           "residual_mean": round(mean, 6), "residual_std": round(var ** 0.5, 6),
           "residual_min": round(lo, 4), "residual_max": round(hi, 4),
           "space": "normalized target space (与 μ 同空间)",
           "complete": a.stride == 1,
           "note": "σ_r 用于阶段 B 把残差归一到单位方差; 不归一会让噪声调度整体偏掉"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"σ_r = {res['residual_std']}  (均值 {res['residual_mean']}, "
          f"{int(n):,} 个值) -> {a.out}")


if __name__ == "__main__":
    main()
