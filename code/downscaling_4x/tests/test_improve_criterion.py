#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 val 改善判据开关(--improve-criterion abs|rel)的口径。

判据同时决定 plateau 减半与早停; 写错不会报错, 只会让学习率与停止点静默漂移。要点:
缺省 abs/1e-4 与既有 run 逐项相同; rel 按比例判且与损失量级无关; 阈值缺省按判据回填;
判据在续训契约里, 续训段改判据被拒, 旧断点缺键按 abs/1e-4 解释。

纯 CPU, 不需要数据。运行: python -m downscaling_4x.tests.test_improve_criterion
"""
import argparse

from downscaling_4x.training import train_downscale as TD


def _expect_exit(fn):
    try:
        fn()
    except SystemExit:
        return True
    raise AssertionError("应当 SystemExit 却通过了")


def test_defaults_match_legacy_runs():
    """缺省 auto: 真实 ERA5 目录 -> abs/1e-4, 与既有 run 逐项相同。"""
    p = argparse.ArgumentParser()
    TD.add_common_args(p)
    a = p.parse_args(["--out", "x"])
    assert a.improve_criterion == "auto" and a.improve_tol is None
    crit, tol = TD.resolve_improve_args(a)
    assert (crit, tol) == ("abs", 1e-4) and (a.improve_criterion, a.improve_tol) == ("abs", 1e-4)
    # 未走 argparse 的 Namespace(测试与工具直接构造)也按缺省解释
    ns = argparse.Namespace()
    assert TD.resolve_improve_args(ns) == ("abs", 1e-4) and ns.improve_criterion == "abs"


def test_auto_follows_input_product():
    """auto: oracle 目录(带 manifest) -> rel/1e-3; 显式 abs 仍以显式为准。"""
    import json, os, tempfile
    from downscaling_4x.evaluation import input_identity as II
    with tempfile.TemporaryDirectory() as tmp:
        ora = os.path.join(tmp, "any_name"); os.makedirs(ora)
        json.dump({"algorithm": "x", "config_sha256": "0" * 64}, open(os.path.join(ora, II.ORACLE_MANIFEST), "w"))
        p = argparse.ArgumentParser(); TD.add_common_args(p)
        a = p.parse_args(["--out", "x", "--era5-dir", ora])
        assert TD.resolve_improve_args(a) == ("rel", 1e-3)
        a = p.parse_args(["--out", "x", "--era5-dir", ora, "--improve-criterion", "abs"])
        assert TD.resolve_improve_args(a) == ("abs", 1e-4)
        plain = os.path.join(tmp, "plain"); os.makedirs(plain)
        a = p.parse_args(["--out", "x", "--era5-dir", plain])
        assert TD.resolve_improve_args(a) == ("abs", 1e-4)


def test_rel_defaults_and_explicit_tol():
    p = argparse.ArgumentParser()
    TD.add_common_args(p)
    a = p.parse_args(["--out", "x", "--improve-criterion", "rel"])
    assert TD.resolve_improve_args(a) == ("rel", 1e-3)
    a = p.parse_args(["--out", "x", "--improve-criterion", "rel", "--improve-tol", "0.02"])
    assert TD.resolve_improve_args(a) == ("rel", 0.02)
    a = p.parse_args(["--out", "x", "--improve-tol", "0.5"])
    assert TD.resolve_improve_args(a) == ("abs", 0.5)
    for bad in ("0", "1", "-0.1"):
        _expect_exit(lambda: TD.resolve_improve_args(p.parse_args(["--out", "x", "--improve-tol", bad])))
    _expect_exit(lambda: TD.resolve_improve_args(argparse.Namespace(improve_criterion="pct")))


def test_is_improved_semantics():
    inf = float("inf")
    assert TD.is_improved(0.5, inf, "abs", 1e-4) and TD.is_improved(0.5, inf, "rel", 1e-3)
    # abs: 1e-4 在 val≈0.001 的量级上是 10%, 0.5% 的下降算停滞; rel 1e-3 下同一步算改善
    assert not TD.is_improved(0.000995, 0.001, "abs", 1e-4)
    assert TD.is_improved(0.000995, 0.001, "rel", 1e-3)
    # rel 与量级无关: 同比例下降在 0.1 与 0.001 上判断相同
    assert TD.is_improved(0.0995, 0.1, "rel", 1e-3) == TD.is_improved(0.000995, 0.001, "rel", 1e-3)
    # 恰好等于阈值不算改善; 略超过才算
    assert not TD.is_improved(0.1 - 1e-4, 0.1, "abs", 1e-4)
    assert TD.is_improved(0.1 - 1.01e-4, 0.1, "abs", 1e-4)
    assert not TD.is_improved(0.1 * (1 - 1e-3), 0.1, "rel", 1e-3)
    assert TD.is_improved(0.1 * (1 - 1.01e-3), 0.1, "rel", 1e-3)


def test_contract_pins_and_legacy_fill():
    assert "improve_criterion" in TD.PINNED and "improve_tol" in TD.PINNED
    d = {"target": "x"}
    filled = TD.apply_legacy_defaults(d)
    assert {"improve_criterion", "improve_tol"} <= set(filled)
    assert (d["improve_criterion"], d["improve_tol"]) == ("abs", 1e-4)
    # 旧断点(无键)对上缺省新 run: 一致; 对上 rel 新 run: 必须被 contract_mismatches 逮住
    saved = {"state_version": 1, "sampler": "s", "dp_size": 1, "steps_per_epoch_effective": 2,
             "train_span_per_epoch": 2, "cond_layout": ["a"], "train_pairing": {}, "val_pairing": {},
             "values": {"lr": 1e-4}}
    TD.apply_legacy_defaults(saved["values"])
    # fit() 对断点侧与当前侧都做旧键回填后再比较, 这里照同一流程构造当前侧
    cur = dict(saved); cur["values"] = {"lr": 1e-4, "improve_criterion": "abs", "improve_tol": 1e-4}
    TD.apply_legacy_defaults(cur["values"])
    assert TD.contract_mismatches(saved, cur) == []
    cur["values"] = {"lr": 1e-4, "improve_criterion": "rel", "improve_tol": 1e-3}
    TD.apply_legacy_defaults(cur["values"])
    diffs = TD.contract_mismatches(saved, cur)
    assert any("improve_criterion" in x for x in diffs) and any("improve_tol" in x for x in diffs)


TESTS = [test_defaults_match_legacy_runs, test_auto_follows_input_product, test_rel_defaults_and_explicit_tol,
         test_is_improved_semantics, test_contract_pins_and_legacy_fill]


def main():
    for fn in TESTS:
        fn()
        print(f"[PASS] {fn.__name__}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
