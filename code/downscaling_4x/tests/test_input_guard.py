#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""输入身份守卫: 产物记录的输入目录与本次目录不一致必须当场拒绝。

真实 ERA5 与 Daymet-oracle 输入同名、同形状、同布局, 混用不会引发任何错误, 只会得到
错标的结果。本测试伪造三类产物 —— checkpoint 参数、BCSD 系数、μ 缓存 manifest ——
核对守卫在一致、不一致、显式放行、旧产物未记录四种情形下的行为。纯 CPU, 不需要真实数据。

运行: python -m downscaling_4x.tests.test_input_guard
"""
import argparse
import json
import os
import tempfile

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.evaluation import input_identity as II


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    raise AssertionError("应当 SystemExit 却通过了")


def _dirs(tmp, *names):
    out = []
    for n in names:
        d = os.path.join(tmp, n)
        os.makedirs(d, exist_ok=True)
        out.append(d)
    return out


def test_describe_by_content(tmp):
    plain, ora = _dirs(tmp, "plain", "oracle_named_anything")
    assert II.describe(plain)["input_product"] == II.PRODUCT_ERA5
    assert "oracle" not in II.describe(plain)
    json.dump({"algorithm": "x_v1", "config_sha256": "ab" * 32,
               "replaced_variables": list(C.TARGETS), "scientific_scope": "leak"},
              open(os.path.join(ora, II.ORACLE_MANIFEST), "w"))
    d = II.describe(ora)
    assert d["input_product"] == II.PRODUCT_ORACLE
    assert d["oracle"]["config_sha256"] == "ab" * 32
    assert d["oracle"]["replaced_variables"] == list(C.TARGETS)
    assert d["era5_dir"] == os.path.abspath(ora)


def test_check_matrix(tmp):
    a, b = _dirs(tmp, "a", "b")
    warns = []
    assert II.check(a, a, "x", warn=warns.append) is True and not warns
    # 同一目录的另一种写法也算一致
    assert II.check(a, os.path.join(a, "..", "a"), "x", warn=warns.append) is True and not warns
    msg = _expect_exit(lambda: II.check(a, b, "伪 checkpoint"))
    assert "伪 checkpoint" in msg and II.ALLOW_FLAG in msg
    assert II.check(a, b, "x", allow=True, warn=warns.append) is False and len(warns) == 1
    warns.clear()
    assert II.check(None, b, "旧产物", warn=warns.append) is False
    assert len(warns) == 1 and "旧产物" in warns[0]
    assert II.check("", b, "旧产物", warn=warns.append) is False


def test_add_arg(tmp):
    p = II.add_arg(argparse.ArgumentParser())
    assert getattr(p.parse_args([]), II.ALLOW_DEST) is False
    assert getattr(p.parse_args([II.ALLOW_FLAG]), II.ALLOW_DEST) is True


def _write_coefs(cdir, era5_dir=None):
    H, W = C.HR_SHAPE
    for v in C.TARGETS:
        extra = {"era5_dir": era5_dir} if era5_dir else {}
        np.savez_compressed(os.path.join(cdir, f"{v}.npz"), a=np.ones((H, W), np.float32),
                            b=np.zeros((H, W), np.float32), n_train_days=3,
                            space=("log1p(mm)" if v == C.PRECIP else "K"), var=v,
                            factor=C.FACTOR, hr_shape=np.array(C.HR_SHAPE), **extra)


def test_bcsd_coefs_guard(tmp):
    from downscaling_4x.baselines.eval_baselines import load_coefs
    fitted, other, cdir, legacy = _dirs(tmp, "fitted", "other", "coefs", "legacy")
    _write_coefs(cdir, fitted)
    assert set(load_coefs(cdir, fitted)) == set(C.TARGETS)
    _expect_exit(lambda: load_coefs(cdir, other))
    assert set(load_coefs(cdir, other, allow_input_mismatch=True)) == set(C.TARGETS)
    _write_coefs(legacy)                      # 旧格式: 无记录, 只警告不拒绝
    assert set(load_coefs(legacy, other)) == set(C.TARGETS)


def test_det_dump_ckpt_guard(tmp):
    import torch
    from downscaling_4x.evaluation import det_dump
    from downscaling_4x.training import train_downscale as TD
    fitted, other = _dirs(tmp, "fitted2", "other2")
    mode = "baseline_21"
    cargs = {"arch": "unet", "base": 8, "pos_grid": 0, "mode": mode,
             "target": C.TARGETS[0], "era5_dir": fitted, "crop": 0}
    net = TD.build_regressor(C.cond_channels(mode), 1, cargs)
    ck = os.path.join(tmp, "ckpt.pt")
    torch.save({"model": net.state_dict(), "args": cargs}, ck)
    ns = argparse.Namespace(checkpoint=ck, method="unet", era5_dir=other,
                            **{II.ALLOW_DEST: False})
    msg = _expect_exit(lambda: det_dump._nn_predictor(ns, "cpu"))
    assert "checkpoint" in msg
    ns.era5_dir = fitted
    out_vars, m, _ = det_dump._nn_predictor(ns, "cpu")
    assert out_vars == [C.TARGETS[0]] and m == mode
    ns.era5_dir = other
    setattr(ns, II.ALLOW_DEST, True)
    det_dump._nn_predictor(ns, "cpu")


def test_mu_cache_manifest_guard(tmp):
    built, other = _dirs(tmp, "built", "other3")
    manifest = {"era5_dir": built}
    assert II.check(manifest.get("era5_dir"), built, "μ 缓存") is True
    _expect_exit(lambda: II.check(manifest.get("era5_dir"), other, "μ 缓存"))
    warns = []
    assert II.check({}.get("era5_dir"), other, "旧 μ 缓存", warn=warns.append) is False


TESTS = [test_describe_by_content, test_check_matrix, test_add_arg,
         test_bcsd_coefs_guard, test_det_dump_ckpt_guard, test_mu_cache_manifest_guard]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for fn in TESTS:
            fn(tmp)
            print(f"✓ {fn.__name__}")
    print("test_input_guard: all passed")


if __name__ == "__main__":
    main()
