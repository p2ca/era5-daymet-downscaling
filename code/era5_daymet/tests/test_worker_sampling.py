#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 DataLoader worker 采样修复(PatchDS/FullFrameDS 的 RNG-in-Dataset 重复采样问题)。

根因: 数据集把 np.random RNG 存在对象里且 __getitem__ 不用索引 i。num_workers>0 时
每个 fork 出的 worker 复制同一 RNG 状态 -> 产生完全重复的采样序列; 非持久 worker 每个
epoch 从相同状态重来 -> 逐 epoch 也重复。

本测试用轻量 FakeData(把帧号编码进返回张量), 不依赖真实 npz/stats:
  1) 训练集: 不加 worker_init_fn(复现旧 bug) 会出现 "相邻样本成对相同" 的签名;
     加了 ds_worker_init 后该签名消失 -> 各 worker 采样流不同。
  2) 训练集: 相邻两个 epoch 的采样序列不同 -> 每 epoch 重新采样(不再逐 epoch 重复)。
  3) 验证集(deterministic=True): 采样只依赖索引 i, 因而跨 num_workers、跨 epoch 完全稳定。
  4) PatchDS deterministic 对同一索引可复现。

运行: python -m era5_daymet.tests.test_worker_sampling
"""
import numpy as np
import torch

from era5_daymet.training.train_downscale import PatchDS, FullFrameDS, ds_worker_init


class FakeData:
    """最小数据接口: 把帧号 t / patch 抽样值编码进返回的 cond 张量, 便于追踪采到了哪一帧。"""
    def __init__(self, nframes):
        self.years = [2000]
        self.ndays = {2000: nframes}

    def full(self, y, t):
        cond = np.full((1, 1, 1), float(t), dtype=np.float32)
        tgt = np.zeros((1, 1, 1), dtype=np.float32)
        m = np.ones((1, 1), dtype=np.float32)
        return cond, tgt, m, None

    def random_patch(self, rng, P):
        v = float(rng.integers(1_000_000))
        cond = np.full((1, 1, 1), v, dtype=np.float32)
        tgt = np.zeros((1, 1, 1), dtype=np.float32)
        m = np.ones((1, 1), dtype=np.float32)
        return cond, tgt, m, None


def _drain(loader):
    """跑一个 epoch, 返回每个样本编码出来的 id 列表(顺序即 DataLoader 返回顺序)。"""
    ids = []
    for cond, _tgt, _m in loader:
        ids.append(int(cond.flatten()[0].item()))
    return ids


def _loader(ds, workers, seeded_worker):
    return torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=workers, drop_last=False,
        worker_init_fn=(ds_worker_init if seeded_worker else None))


def test_train_worker_dedup():
    """核心: 2 worker 下, 未修复会成对重复(id[2k]==id[2k+1]); 修复后不再。"""
    N, L = 200, 64
    # --- 复现旧 bug: 同类数据集但 DataLoader 不带 worker_init_fn ---
    buggy = _drain(_loader(FullFrameDS(FakeData(N), L, seed=42), workers=2, seeded_worker=False))
    pairs_eq_buggy = [buggy[2 * k] == buggy[2 * k + 1] for k in range(len(buggy) // 2)]
    assert all(pairs_eq_buggy), (
        "预期在未修复(无 worker_init_fn)下复现出成对重复签名, 实际未复现 -> 测试失去判别力")

    # --- 修复: 带 ds_worker_init ---
    fixed = _drain(_loader(FullFrameDS(FakeData(N), L, seed=42), workers=2, seeded_worker=True))
    pairs_eq_fixed = [fixed[2 * k] == fixed[2 * k + 1] for k in range(len(fixed) // 2)]
    assert not all(pairs_eq_fixed), "修复后仍每对相同 -> worker 采样流未去重"
    # 绝大多数对应当不同(允许极少数偶然相同)
    assert sum(pairs_eq_fixed) <= len(pairs_eq_fixed) // 4, (
        f"修复后成对相同比例过高: {sum(pairs_eq_fixed)}/{len(pairs_eq_fixed)}")
    # 唯一值数量应远高于旧 bug
    assert len(set(fixed)) > len(set(buggy)), (
        f"修复后唯一样本数未提升: fixed={len(set(fixed))} buggy={len(set(buggy))}")


def test_train_fresh_across_epochs():
    """训练集相邻两个 epoch 的采样序列应不同(不再逐 epoch 重复同一序列)。"""
    N, L = 200, 64
    dl = _loader(FullFrameDS(FakeData(N), L, seed=7), workers=2, seeded_worker=True)
    ep_a = _drain(dl)
    ep_b = _drain(dl)
    assert ep_a != ep_b, "两个 epoch 采样序列相同 -> 逐 epoch 重复未修复"


def test_val_deterministic_stable():
    """验证集(deterministic)跨 num_workers 与跨 epoch 完全稳定, 且不受 worker 影响。"""
    N, L = 200, 64
    v0 = _drain(_loader(FullFrameDS(FakeData(N), L, seed=987, deterministic=True), 0, False))
    dl2 = _loader(FullFrameDS(FakeData(N), L, seed=987, deterministic=True), 2, True)
    v2a = _drain(dl2)
    v2b = _drain(dl2)
    assert v0 == v2a, f"验证集 num_workers 0 vs 2 不一致: {v0[:8]} vs {v2a[:8]}"
    assert v2a == v2b, "验证集跨 epoch 不稳定(deterministic 应逐 epoch 固定)"
    # 不同 seed 应给出不同的固定序列(确认 seed 确实进了采样)
    v_other = _drain(_loader(FullFrameDS(FakeData(N), L, seed=123, deterministic=True), 0, False))
    assert v_other != v0, "不同 seed 的确定性验证序列相同 -> seed 未进入采样"


def test_patchds_deterministic_reproducible():
    """PatchDS deterministic 对同一索引可复现; 训练模式则随取随进。"""
    dsd = PatchDS(FakeData(0), P=4, length=16, seed=5, deterministic=True)
    assert int(dsd[3][0].flatten()[0]) == int(dsd[3][0].flatten()[0]), "同索引应复现"
    assert int(dsd[3][0].flatten()[0]) != int(dsd[4][0].flatten()[0]), "不同索引应不同"
    dst = PatchDS(FakeData(0), P=4, length=16, seed=5, deterministic=False)
    a, b = int(dst[0][0].flatten()[0]), int(dst[0][0].flatten()[0])
    assert a != b, "训练模式 self.rng 应随取随进(两次取同一索引结果不同)"


def test_val_shard_disjoint_across_ranks():
    """★val 分片(2026-07-25): 不同 dp_rank 的 index_offset 必须落在互不重复的帧上。

    修复前 index_offset 恒为 0 -> 所有 rank 评同一批帧(128 卡重复 128 遍, 只覆盖 val_steps 天)。
    修复后 rank r 取全局索引 [r*L, (r+1)*L), 经固定置换映射到互不相同的帧。"""
    N, L, ranks = 200, 5, 8          # 8 个 rank x 每 rank 5 帧 = 40 帧 < 200 -> 应全不重复
    per_rank = [_drain(_loader(
        FullFrameDS(FakeData(N), L, seed=987, deterministic=True, index_offset=r * L), 0, False))
        for r in range(ranks)]
    for r, ids in enumerate(per_rank):
        assert len(set(ids)) == L, f"rank {r} 自身取到重复帧: {ids}"
    union = [i for ids in per_rank for i in ids]
    assert len(set(union)) == ranks * L, (
        f"跨 rank 覆盖数 {len(set(union))} != {ranks * L} -> 分片有重叠(修复未生效)")
    # 同一 rank 跨 epoch 仍必须完全固定(否则 LR 减半/早停判据会被采样噪声污染)
    dl = _loader(FullFrameDS(FakeData(N), L, seed=987, deterministic=True, index_offset=3 * L), 2, True)
    assert _drain(dl) == _drain(dl), "分片后跨 epoch 不再固定 -> 破坏了确定性验证前提"
    assert _drain(dl) == per_rank[3], "num_workers=2 与 0 结果不一致"


def test_val_shard_wraps_when_oversubscribed():
    """请求帧数超过验证集容量时按置换回绕, 不越界、不崩(覆盖数 = 验证集大小)。"""
    N, L, ranks = 10, 4, 5           # 5x4=20 > 10 -> 必然回绕
    union = [i for r in range(ranks) for i in _drain(_loader(
        FullFrameDS(FakeData(N), L, seed=987, deterministic=True, index_offset=r * L), 0, False))]
    assert len(union) == ranks * L, "样本数不对"
    assert set(union) == set(range(N)), f"回绕后未覆盖全部 {N} 帧: {sorted(set(union))}"


def test_patchds_shard_differs_across_ranks():
    """PatchDS(裁块 val, 供 corrdiff/scd 用) 不同 index_offset 应给出不同 patch。"""
    a = _drain(_loader(PatchDS(FakeData(0), P=4, length=8, seed=987,
                               deterministic=True, index_offset=0), 0, False))
    b = _drain(_loader(PatchDS(FakeData(0), P=4, length=8, seed=987,
                               deterministic=True, index_offset=8), 0, False))
    assert not set(a) & set(b), f"两个 rank 的 val patch 有重叠: {set(a) & set(b)}"


def _stream(N, per_rank, ranks, epoch, span=None, workers=0, seed=1234):
    """模拟一个 epoch 内全体 rank 的训练取样(洗牌无放回流)。"""
    span = span if span is not None else ranks * per_rank
    out = []
    for r in range(ranks):
        ds = FullFrameDS(FakeData(N), per_rank, seed=seed, stream=True,
                         index_offset=r * per_rank, epoch_span=span)
        ds.epoch = epoch
        out.append(_drain(_loader(ds, workers, True)))
    return out


def test_train_stream_no_replacement_within_epoch():
    """★标准做法(2026-07-26): 一个 epoch 内全体 rank 取到的帧必须互不重复(无放回)。

    改前用 self.rng 有放回抽 -> 单 epoch 期望 12.6% 重复; 改后由置换保证零重复。"""
    N, PER, R = 200, 5, 8            # 8 rank x 5 = 40 <= 200, 单 epoch 落在同一遍数据内
    per_rank = _stream(N, PER, R, epoch=0)
    flat = [x for ids in per_rank for x in ids]
    assert len(flat) == R * PER, "样本数不对"
    assert len(set(flat)) == R * PER, f"epoch 内出现重复帧: {len(flat) - len(set(flat))} 次"


def test_train_stream_exposure_is_uniform():
    """每帧曝光次数只能取 floor/ceil 两个值(标准做法的核心性质)。

    N=50, 每 epoch 20 帧, 跑 10 epoch = 200 帧次 = 恰好 4 遍 -> 每帧应正好 4 次。"""
    N, PER, R, EP = 50, 4, 5, 10
    cnt = {}
    for ep in range(EP):
        for ids in _stream(N, PER, R, epoch=ep):
            for x in ids:
                cnt[x] = cnt.get(x, 0) + 1
    assert len(cnt) == N, f"只覆盖到 {len(cnt)}/{N} 帧"
    assert set(cnt.values()) == {EP * R * PER // N}, f"曝光次数不均匀: {sorted(set(cnt.values()))}"

    # 非整数遍时应只出现 floor / ceil 两档
    N2, EP2 = 47, 10                                  # 200/47 = 4.25 遍
    cnt2 = {}
    for ep in range(EP2):
        for ids in _stream(N2, PER, R, epoch=ep):
            for x in ids:
                cnt2[x] = cnt2.get(x, 0) + 1
    assert set(cnt2.values()) <= {4, 5}, f"曝光次数应只有 4/5 两档, 实际 {sorted(set(cnt2.values()))}"


def test_train_stream_advances_across_epochs():
    """不同 epoch 必须取到不同样本(epoch 没生效的话会逐轮重复同一批)。"""
    N, PER, R = 200, 5, 4
    a = [x for ids in _stream(N, PER, R, epoch=0) for x in ids]
    b = [x for ids in _stream(N, PER, R, epoch=1) for x in ids]
    assert a != b, "epoch 递增后样本未变 -> set_epoch 等价机制失效"
    assert not (set(a) & set(b)), "相邻 epoch 样本重叠 -> 未按连续流推进"


def test_train_stream_worker_invariant():
    """洗牌流只依赖全局索引, 因而与 num_workers 无关(不受 fork 影响)。"""
    N, PER, R = 200, 5, 3
    w0 = _stream(N, PER, R, epoch=2, workers=0)
    w2 = _stream(N, PER, R, epoch=2, workers=2)
    assert w0 == w2, f"num_workers 改变了训练取样: {w0[0]} vs {w2[0]}"


def main():
    tests = [
        test_train_worker_dedup,
        test_train_fresh_across_epochs,
        test_val_deterministic_stable,
        test_patchds_deterministic_reproducible,
        test_val_shard_disjoint_across_ranks,
        test_val_shard_wraps_when_oversubscribed,
        test_patchds_shard_differs_across_ranks,
        test_train_stream_no_replacement_within_epoch,
        test_train_stream_exposure_is_uniform,
        test_train_stream_advances_across_epochs,
        test_train_stream_worker_invariant,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
