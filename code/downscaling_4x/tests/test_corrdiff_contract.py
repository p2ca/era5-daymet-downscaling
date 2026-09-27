#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定 CorrDiff 两阶段与本合同的接口: 全域上下文的通道选取、阶段 B 的条件宽度。

上游按"末尾 n 个"切片排除空间常数通道。本合同的历史段排在时间通道之后, 那样切会丢掉
两个真实风场而不是时间通道 —— **宽度完全相同**, 权重照样装得回去, 训练照常收敛, 只是
每个 patch 拿到的全域上下文换成了错的一组通道。本文件把这件事变成会失败的检查, 并显式
构造上游写法作负对照: 只验证正确实现能过, 等于没验证这个测试有分辨力。

纯 CPU, 不需要数据。

运行: python -m downscaling_4x.tests.test_corrdiff_contract
"""
import torch

from downscaling_4x import contract as C
from downscaling_4x.models.corrdiff_loss import (_global_context, constant_cond_index,
                                                 global_context_index)
from downscaling_4x.training.stage_b_mean import stage_b_cond_channels


def test_constant_channels_are_exactly_the_time_channels():
    for mode in C.MODES:
        layout = C.cond_layout(mode)
        idx = constant_cond_index(mode)
        assert [layout[i] for i in idx] == list(C.TIME_ORDER), (mode, idx)
        # 历史段是空间场, 绝不能被当成常数
        assert not any(layout[i].startswith(C.HISTORY_PREFIX) for i in idx), mode


def test_global_context_is_the_complement():
    for mode in C.MODES:
        n = C.cond_channels(mode)
        const, keep = set(constant_cond_index(mode)), global_context_index(mode)
        assert sorted(set(keep) | const) == list(range(n))
        assert not (set(keep) & const)
        assert len(keep) == n - len(C.TIME_ORDER)


def test_global_context_selects_those_channels():
    for mode in C.MODES:
        n = C.cond_channels(mode)
        keep = global_context_index(mode)
        x = torch.arange(n, dtype=torch.float32).reshape(1, n, 1, 1).expand(1, n, 3, 3)
        got = _global_context(x, keep)
        assert got.shape[1] == len(keep)
        assert got[0, :, 0, 0].tolist() == [float(i) for i in keep]


def test_upstream_tail_slicing_would_pick_wrong_channels():
    """负对照: 上游写法在有历史段的模式下会丢错通道, 且宽度一模一样。"""
    mode = "history_51"
    layout = C.cond_layout(mode)
    n, k = len(layout), len(C.TIME_ORDER)
    tail = tuple(range(n - k))                       # 上游: img_lr[:, : C - n_const]
    ours = global_context_index(mode)
    assert len(tail) == len(ours), "两种取法宽度相同 —— 这正是它不会报错的原因"
    assert tail != ours, "负对照与正确实现必须不同, 否则本测试无分辨力"
    dropped_by_tail = [layout[i] for i in range(n) if i not in tail]
    assert dropped_by_tail == ["t_minus_1:v_component_of_wind_500",
                               "t_minus_1:v_component_of_wind_850"], dropped_by_tail
    # 而 baseline 模式下两者恰好一致, 所以只测 baseline 是发现不了这个问题的
    b = "baseline_21"
    nb = C.cond_channels(b)
    assert tuple(range(nb - k)) == global_context_index(b)


def test_stage_b_cond_width_matches_what_the_loss_builds():
    """阶段 B 的条件由 μ + 本 patch 条件 + 全域上下文 三段拼成; 声明的宽度必须等于实际。

    两处各算各的, 差一个通道就会让权重装不回去 —— 那是会报错的; 但若两处都错成同一个数,
    就变成静默的错口径。这里把"声明"钉在"实际拼进去的那组通道"上。
    """
    for mode in C.MODES:
        n_target, n_grid = 1, 7
        want = n_target + C.cond_channels(mode) + len(global_context_index(mode)) + n_grid
        assert stage_b_cond_channels(mode, n_target=n_target, n_grid=n_grid) == want, mode
    assert stage_b_cond_channels("baseline_21", n_grid=0) == 1 + 21 + 19
    assert stage_b_cond_channels("history_51", n_grid=0) == 1 + 51 + 49


def test_global_context_rejects_out_of_range():
    x = torch.zeros(1, 4, 2, 2)
    try:
        _global_context(x, (0, 1, 9))
    except ValueError as e:
        assert "越界" in str(e)
        return
    raise AssertionError("越界下标没有被拒绝")


def main():
    tests = [
        test_constant_channels_are_exactly_the_time_channels,
        test_global_context_is_the_complement,
        test_global_context_selects_those_channels,
        test_upstream_tail_slicing_would_pick_wrong_channels,
        test_stage_b_cond_width_matches_what_the_loss_builds,
        test_global_context_rejects_out_of_range,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
