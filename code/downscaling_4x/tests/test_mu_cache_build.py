#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""μ 缓存的分片、续跑与落盘: 三处写错都不会报错, 在这里钉死。

  * 年份分片必须恰好覆盖每个年份一次。漏掉的年份若落在 val/test 上, 训练侧的
    residual_scale 只读训练年也碰不到, 要等到评测才暴露。
  * 续跑判"这一年已经建好"不能只看文件在不在: 写盘中途被杀会留下截断的 .npy,
    被跳过之后那一年的 μ 是垃圾, 而之后没有任何一步会报错。
  * 落盘必须原子, 否则上一条描述的截断文件本身就会被造出来。
  * 续跑进口径不同的旧缓存会留下"manifest 记着一套、部分 .npy 出自另一套"的组合,
    而 MuCache.verify() 对它照常放行。

纯 CPU, 不需要真实数据, 不建任何真的 μ。
运行: python -m downscaling_4x.tests.test_mu_cache_build
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.tools import build_mu_cache as B

SHAPE = (C.DAYS_PER_YEAR, C.HR_SHAPE[0], C.HR_SHAPE[1])
SMALL = (4, 3, 5)          # 形状检查与内容无关, 用小数组跑得快


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    raise AssertionError("应当 SystemExit 却通过了")


def test_shard_is_a_partition():
    for n in range(0, 45):
        for world in range(1, 13):
            parts = [B.shard(list(range(n)), k, world) for k in range(world)]
            flat = [v for p in parts for v in p]
            assert flat == list(range(n)), f"n={n} world={world} 分片不是一个划分"
            assert len(parts) == world
            sizes = [len(p) for p in parts]
            assert max(sizes) - min(sizes) <= 1, f"n={n} world={world} 分片不均 {sizes}"
    # 41 年 8 进程是实际用法: 每片 5-6 年, 合起来正好 41
    parts = [B.shard(list(range(1980, 2021)), k, 8) for k in range(8)]
    assert sorted(v for p in parts for v in p) == list(range(1980, 2021))
    print("  shard: 0-44 年 x 1-12 进程 全部构成划分且均衡")


def test_usable_rejects_truncated(tmp):
    good = Path(tmp) / "good.npy"
    B.save_atomic(good, np.zeros(SMALL, np.float16))
    assert B.usable(good, SMALL)

    cut = Path(tmp) / "cut.npy"
    cut.write_bytes(good.read_bytes()[:-8])
    # 分辨力自证: 只看"文件在不在"会把截断文件当成建好的年份
    assert cut.exists() and not B.usable(cut, SMALL), "截断的 .npy 必须判为不可用"

    head = Path(tmp) / "head.npy"
    head.write_bytes(good.read_bytes()[:20])
    assert not B.usable(head, SMALL), "连文件头都不全的必须判为不可用"

    wrong_shape = Path(tmp) / "shape.npy"
    B.save_atomic(wrong_shape, np.zeros((SMALL[0] - 1,) + SMALL[1:], np.float16))
    assert not B.usable(wrong_shape, SMALL), "天数不对的必须判为不可用"

    wrong_dtype = Path(tmp) / "dtype.npy"
    B.save_atomic(wrong_dtype, np.zeros(SMALL, np.float32))
    assert not B.usable(wrong_dtype, SMALL), "dtype 不是 float16 的必须判为不可用"

    assert not B.usable(Path(tmp) / "nope.npy", SMALL)
    print("  usable: 截断/半个文件头/形状不符/dtype 不符/不存在 全部判为不可用")


def test_save_atomic_naming_and_cleanup(tmp):
    d = Path(tmp) / "atomic"
    d.mkdir()
    p = d / "2011.npy"
    arr = np.arange(np.prod(SMALL), dtype=np.float16).reshape(SMALL)
    B.save_atomic(p, arr)
    names = sorted(x.name for x in d.iterdir())
    # np.save 收到路径会补 .npy; 这里必须走文件对象, 否则临时名会变成 2011.npy.tmp.N.npy
    assert names == ["2011.npy"], f"落盘后目录里应只有 2011.npy, 实际 {names}"
    assert np.array_equal(np.load(p), arr)

    # 模拟写到一半被杀: 临时文件留在 .npy 之外, 不会被当成一个年份
    leftover = d / "2012.npy.tmp.99999"
    leftover.write_bytes(b"partial")
    assert not (d / "2012.npy").exists()
    assert not B.usable(d / "2012.npy", SMALL)
    print("  save_atomic: 落盘只留 <年>.npy; 中断只留 .tmp, 不会冒充年份文件")


def _fake_cache(tmp, mode, sha, era5, daymet, years):
    """伪造一份 manifest + 一个假 checkpoint 文件, 只为触发续跑口径检查。"""
    out = Path(tmp) / "cache"
    out.mkdir(exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "checkpoints": {C.TARGETS[0]: {"path": "/old/ckpt.pt", "sha256": sha}},
        "mode": mode, "cond_layout": C.cond_layout(mode),
        "era5_dir": era5, "daymet_dir": daymet, "years": years,
    }, ensure_ascii=False), encoding="utf-8")
    return out


def test_resume_binding_guard(tmp):
    import sys
    era5, daymet = [str(Path(tmp) / n) for n in ("era5", "daymet")]
    for d in (era5, daymet):
        os.makedirs(d, exist_ok=True)
    ck = Path(tmp) / "ckpt.pt"
    ck.write_bytes(b"not a real checkpoint, only its sha matters here")
    sha = B.sha256(ck)
    mode = C.DEFAULT_MODE
    years = list(range(1980, 1985))

    def run(out, extra=()):
        argv = ["build_mu_cache", "--ckpt", f"{C.TARGETS[0]}={ck}", "--out", str(out),
                "--mode", mode, "--era5-dir", era5, "--daymet-dir", daymet,
                "--years", *[str(y) for y in years], "--resume", *extra]
        old = sys.argv
        sys.argv = argv
        try:
            B.main()
        finally:
            sys.argv = old

    # 口径一致: 守卫放行, 之后才在加载假 checkpoint 时失败(说明确实走过了守卫)
    out = _fake_cache(tmp, mode, sha, era5, daymet, years)
    try:
        run(out)
    except SystemExit as e:
        assert "口径与本次不符" not in str(e) and "超出" not in str(e), f"不该被守卫拦: {e}"
    except Exception:
        pass    # 假 checkpoint 加载失败是预期的, 守卫已经放行

    for label, kw in [
        ("checkpoint 变了", dict(sha="00" * 32)),
        ("输入目录变了", dict(era5=daymet)),
    ]:
        out = _fake_cache(tmp, mode, **{**dict(sha=sha, era5=era5, daymet=daymet,
                                               years=years), **kw})
        msg = _expect_exit(lambda: run(out))
        assert "口径与本次不符" in msg, f"{label}: 应被守卫拦下, 实际 {msg}"

    # 年份超出 manifest 声明的范围
    out = _fake_cache(tmp, mode, sha, era5, daymet, years[:3])
    msg = _expect_exit(lambda: run(out))
    assert "超出" in msg, msg

    # 没有 manifest 就不能续跑
    empty = Path(tmp) / "empty"
    empty.mkdir()
    assert "需要已有 manifest" in _expect_exit(lambda: run(empty))
    print("  resume 守卫: checkpoint/输入目录/年份范围/无 manifest 四种情形全部拒绝")


def test_check_only(tmp):
    import sys
    out = Path(tmp) / "chk"
    (out / C.TARGETS[0]).mkdir(parents=True)
    years = [1980, 1981]
    (out / "manifest.json").write_text(json.dumps({
        "checkpoints": {C.TARGETS[0]: {"sha256": "x"}}, "years": years}), encoding="utf-8")

    def run():
        old = sys.argv
        sys.argv = ["build_mu_cache", "--out", str(out), "--check-only"]
        try:
            B.main()
        finally:
            sys.argv = old

    assert "缓存不完整" in _expect_exit(run), "全缺时必须失败"
    full = np.zeros(SHAPE, np.float16)
    B.save_atomic(out / C.TARGETS[0] / "1980.npy", full)
    assert "缓存不完整" in _expect_exit(run), "缺一年也必须失败"
    B.save_atomic(out / C.TARGETS[0] / "1981.npy", full)
    run()
    print("  check-only: 全缺/缺一年都失败, 齐全才通过")


def main():
    print("== μ 缓存 分片 / 续跑 / 落盘 ==")
    test_shard_is_a_partition()
    with tempfile.TemporaryDirectory() as tmp:
        test_usable_rejects_truncated(tmp)
        test_save_atomic_naming_and_cleanup(tmp)
        test_resume_binding_guard(tmp)
        test_check_only(tmp)
    print("全部通过")


if __name__ == "__main__":
    main()
