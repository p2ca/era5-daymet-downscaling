#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""把 CorrDiff 分片评测(train_corrdiff.py --dump-sums)的累计量精确合并成规范
metrics_corrdiff.json。分片必须覆盖不相交的 test 天且合起来是完整 365 天。

  python _merge_corrdiff_sums.py --sums runs/exp/.../shard_*.pkl \
      --out runs/exp/20260723-fulltest-eval/corrdiff --n-total-days 365
"""
import argparse
import glob
import os
import sys

from era5_daymet.evaluation import eval_common as EC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sums", nargs="+", required=True, help="shard .pkl(可用通配)")
    p.add_argument("--out", required=True)
    p.add_argument("--test-year", type=int, default=2020)
    p.add_argument("--n-total-days", type=int, default=365)
    p.add_argument("--tag", default="corrdiff")
    a = p.parse_args()
    paths = sorted({q for pat in a.sums for q in glob.glob(pat)})
    if not paths:
        sys.exit(f"没找到任何 shard: {a.sums}")
    print(f"合并 {len(paths)} 个 shard:")
    for q in paths:
        print("  ", q)
    EC.merge_sums_and_finalize(paths, a.out, a.test_year, tag=a.tag,
                               n_total_days=a.n_total_days)


if __name__ == "__main__":
    main()
