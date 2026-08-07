#!/usr/bin/env python
"""Regression test for exact full-frame deterministic training resume.

Run:
    python -m era5_daymet.tests.test_deterministic_resume
"""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from era5_daymet.training import train_downscale as TD


class _FakeFullFrameData:
    """Small deterministic replacement for DownscaleData used by this test."""

    def __init__(self, _era5, _daymet, years, _in_vars, _out_vars, _stats, use_clim=False):
        del use_clim
        self.years = list(years)
        self.ndays = {y: (7 if y == 2000 else 5) for y in self.years}

    def full(self, y, t):
        base = np.float32((y - 1999) * 0.1 + t * 0.03)
        yy, xx = np.mgrid[:6, :8].astype(np.float32)
        cond = (base + yy[None] * 0.01 + xx[None] * 0.02).astype(np.float32)
        target = (0.7 * cond + 0.15).astype(np.float32)
        mask = np.ones((1, 6, 8), np.float32)
        return cond, target, mask, target


def _args(out, epochs, resume_from=""):
    return SimpleNamespace(
        model="resume_test",
        era5_dir="fake_era5",
        daymet_dir="fake_daymet",
        stats_dir="fake_stats",
        in_vars=["x"],
        out_vars=["y"],
        use_clim=False,
        train_years=[2000],
        val_years=[2001],
        test_year=2002,
        out=out,
        patch=6,
        batch=1,
        lr=1e-2,
        weight_decay=0.0,
        epochs=epochs,
        steps_per_epoch=4,
        epoch_frames=4,
        val_steps=2,
        patience=20,
        warmup=0,
        lr_patience=3,
        lr_factor=0.5,
        min_lr=1e-6,
        grad_clip=0.0,
        workers=0,
        amp=False,
        full_frame=True,
        resume_from=resume_from,
    )


def _model(seed):
    torch.manual_seed(seed)
    return torch.nn.Conv2d(1, 1, kernel_size=1)


def _load(path):
    return torch.load(path, map_location="cpu")


def _assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def _assert_nested_equal(a, b, path="root"):
    if torch.is_tensor(a):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            _assert_nested_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for idx, (left, right) in enumerate(zip(a, b)):
            _assert_nested_equal(left, right, f"{path}[{idx}]")
    else:
        assert a == b, path


def test_continuous_equals_segmented_resume():
    original_data = TD.DownscaleData
    TD.DownscaleData = _FakeFullFrameData
    try:
        with tempfile.TemporaryDirectory(prefix="era5-daymet-resume-test-") as root:
            full_out = os.path.join(root, "continuous")
            split_out = os.path.join(root, "segmented")

            full_model = _model(123)
            TD.fit_deterministic(
                full_model, object(), _args(full_out, 4), "cpu", (0, 1, 0, False))

            first_model = _model(123)
            TD.fit_deterministic(
                first_model, object(), _args(split_out, 2), "cpu", (0, 1, 0, False))
            split_last = os.path.join(split_out, "last.pt")
            assert os.path.isfile(split_last)

            resumed_model = _model(999)  # must be overwritten by last.pt
            TD.fit_deterministic(
                resumed_model, object(), _args(split_out, 4, split_last),
                "cpu", (0, 1, 0, False))

            continuous = _load(os.path.join(full_out, "last.pt"))
            segmented = _load(split_last)
            _assert_state_equal(continuous["model"], segmented["model"])
            _assert_nested_equal(continuous["opt"], segmented["opt"], "optimizer")
            assert continuous["history"] == segmented["history"]
            for key in ("epoch", "best", "bad", "plateau", "cur_lr"):
                assert continuous[key] == segmented[key], key
            assert segmented["epoch"] == 3
            assert [h["epoch"] for h in segmented["history"]] == [1, 2, 3, 4]
    finally:
        TD.DownscaleData = original_data


def test_resume_rejects_wrong_lr_and_output_directory():
    original_data = TD.DownscaleData
    TD.DownscaleData = _FakeFullFrameData
    try:
        with tempfile.TemporaryDirectory(prefix="era5-daymet-resume-guard-") as root:
            out = os.path.join(root, "source")
            TD.fit_deterministic(
                _model(123), object(), _args(out, 1), "cpu", (0, 1, 0, False))
            last = os.path.join(out, "last.pt")

            bad_lr = _args(out, 2, last)
            bad_lr.lr = 2e-2
            try:
                TD.fit_deterministic(
                    _model(999), object(), bad_lr, "cpu", (0, 1, 0, False))
            except RuntimeError as exc:
                assert "lr:" in str(exc)
            else:
                raise AssertionError("resume accepted a mismatched LR")

            wrong_out = _args(os.path.join(root, "other"), 2, last)
            try:
                TD.fit_deterministic(
                    _model(999), object(), wrong_out, "cpu", (0, 1, 0, False))
            except RuntimeError as exc:
                assert "当前 --out 目录" in str(exc)
            else:
                raise AssertionError("resume accepted a checkpoint from another OUT")
    finally:
        TD.DownscaleData = original_data


def _ddp_worker(rank, world, init_method, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    original_data = TD.DownscaleData
    original_ddp = TD.DDP
    TD.DownscaleData = _FakeFullFrameData
    TD.DDP = lambda model, device_ids, static_graph: original_ddp(  # CPU DDP test wrapper
        model, device_ids=None, static_graph=static_graph)
    try:
        ddp_info = (rank, world, rank, True)
        full_out = os.path.join(root, "ddp_continuous")
        split_out = os.path.join(root, "ddp_segmented")

        TD.fit_deterministic(
            _model(123), object(), _args(full_out, 4),
            torch.device("cpu"), ddp_info)
        dist.barrier()
        TD.fit_deterministic(
            _model(123), object(), _args(split_out, 2),
            torch.device("cpu"), ddp_info)
        dist.barrier()
        split_last = os.path.join(split_out, "last.pt")
        TD.fit_deterministic(
            _model(999), object(), _args(split_out, 4, split_last),
            torch.device("cpu"), ddp_info)
        dist.barrier()

        if rank == 0:
            continuous = _load(os.path.join(full_out, "last.pt"))
            segmented = _load(split_last)
            _assert_state_equal(continuous["model"], segmented["model"])
            _assert_nested_equal(continuous["opt"], segmented["opt"], "ddp_optimizer")
            assert continuous["history"] == segmented["history"]
    finally:
        TD.DownscaleData = original_data
        TD.DDP = original_ddp
        dist.destroy_process_group()


def test_two_rank_ddp_continuous_equals_resume():
    with tempfile.TemporaryDirectory(prefix="era5-daymet-resume-ddp-") as root:
        init_method = f"file://{os.path.join(root, 'dist_init')}"
        mp.spawn(_ddp_worker, args=(2, init_method, root), nprocs=2, join=True)


def main():
    tests = (
        test_continuous_equals_segmented_resume,
        test_resume_rejects_wrong_lr_and_output_directory,
        test_two_rank_ddp_continuous_equals_resume,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
