#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_meta_audit.py — collect_results.audit() 的对拍能力
============================================================================
STATUS.md 与 LEDGER.md 是从 meta.json 渲染的, 渲染器忠实反映输入, 不会自己发现输入
是错的。audit() 是唯一让"meta 与产物分叉"变得可见的环节, 因此它必须既抓得住真分歧,
又不能把合法的精度差异报成分歧 —— 一个只会误报的检测器会被忽略, 一个从不触发的
检测器与坏掉的没有区别, 两种都会让错误重新变成静默的。

本测试用合成实验目录锁定两侧边界。

运行: python -m era5_daymet.tests.test_meta_audit
============================================================================
"""
import json
import os
import sys
import tempfile
import time

from era5_daymet.tools.reporting import collect_results as CR


def _make(root, name, hist, meta, last_pt_age_h=None):
    """在 root 下造一个合成实验目录: loss_history.json + meta.json (+ 可选 last.pt)。"""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "loss_history.json"), "w") as f:
        json.dump(hist, f)
    m = dict(meta); m["_dir"] = name
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({k: v for k, v in m.items() if k != "_dir"}, f)
    if last_pt_age_h is not None:
        p = os.path.join(d, "last.pt")
        open(p, "wb").close()
        t = time.time() - last_pt_age_h * 3600
        os.utime(p, (t, t))
    return m


def _audit(root, metas):
    old = CR.EXP
    CR.EXP = root
    try:
        return CR.audit(metas)
    finally:
        CR.EXP = old


CURVE = [{"samples": s, "train": 0.03, "val": v} for s, v in
         [(32768, 0.5), (65536, 0.4), (3702784, 0.021637), (4161536, 0.022129)]]
TRAIN_OK = {"samples_trained": 4179968, "best_val": 0.021637,
            "best_val_at_samples": 3702784, "val_every": 32768}


def test_flags_stale_snapshot():
    """本周的真实故障: meta 停在首段起步时的快照, 曲线已跑到 4.18M。必须被抓住。"""
    with tempfile.TemporaryDirectory() as root:
        m = _make(root, "stale", CURVE,
                  {"status": "running (训练中)",
                   "training": {"samples_trained": 65536, "best_val": 0.556488,
                                "best_val_at_samples": 65536, "val_every": 32768}})
        issues = _audit(root, [m])
        assert issues, "meta 差 25 倍却未被抓出"
        blob = " ".join(msg for _, msg in issues)
        assert "samples_trained" in blob and "best_val" in blob, blob
    print("[PASS] test_flags_stale_snapshot")


def test_flags_wrong_best_at():
    """best_val 对但最优点位置错 —— 样本数是整数, 无舍入余地, 必须严格抓。"""
    with tempfile.TemporaryDirectory() as root:
        tr = dict(TRAIN_OK); tr["best_val_at_samples"] = 4161536
        m = _make(root, "badat", CURVE, {"status": "done", "training": tr})
        issues = _audit(root, [m])
        assert any("best_val_at_samples" in msg for _, msg in issues), issues
    print("[PASS] test_flags_wrong_best_at")


def test_flags_running_but_idle():
    """status 仍是 running 而 last.pt 早已不动 —— 作业其实结束了。"""
    with tempfile.TemporaryDirectory() as root:
        m = _make(root, "idle", CURVE, {"status": "running (训练中)", "training": TRAIN_OK},
                  last_pt_age_h=9.0)
        assert any("running" in msg for _, msg in _audit(root, [m])), "陈旧的 running 未被抓出"
    print("[PASS] test_flags_running_but_idle")


def test_accepts_rounded_best_val():
    """meta 按声明精度记 0.02164, 曲线是 0.021637 —— 合法舍入, 不得误报。"""
    with tempfile.TemporaryDirectory() as root:
        tr = dict(TRAIN_OK); tr["best_val"] = 0.02164
        m = _make(root, "rounded", CURVE, {"status": "done", "training": tr})
        issues = _audit(root, [m])
        assert not any("best_val " in msg for _, msg in issues), issues
    print("[PASS] test_accepts_rounded_best_val")


def test_accepts_samples_between_val_points():
    """训练可以停在两个验证点之间, samples_trained 合法地略大于曲线末点。"""
    with tempfile.TemporaryDirectory() as root:
        m = _make(root, "between", CURVE, {"status": "done", "training": TRAIN_OK})
        issues = _audit(root, [m])
        assert not issues, f"合法的 {TRAIN_OK['samples_trained']:,} vs 曲线末 4,161,536 被误报: {issues}"
    print("[PASS] test_accepts_samples_between_val_points")


def test_accepts_clean_running_job():
    """真正在跑的作业: status running 且 last.pt 刚更新过, 不得误报。"""
    with tempfile.TemporaryDirectory() as root:
        m = _make(root, "live", CURVE, {"status": "running (训练中)", "training": TRAIN_OK},
                  last_pt_age_h=0.1)
        assert not _audit(root, [m]), "在跑的作业被误报"
    print("[PASS] test_accepts_clean_running_job")


def main():
    for t in (test_flags_stale_snapshot, test_flags_wrong_best_at, test_flags_running_but_idle,
              test_accepts_rounded_best_val, test_accepts_samples_between_val_points,
              test_accepts_clean_running_job):
        t()
    print("test_meta_audit: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
