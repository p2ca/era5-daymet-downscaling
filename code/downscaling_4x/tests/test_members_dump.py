#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
test_members_dump.py — 成员场落盘与补录守卫的回归测试(纯 CPU)
============================================================================
钉住四件事, 任何一件坏了都是静默错误(成员照样写满 365 天, 文件完整、能读、不报错):

  1. 成员场的空间与掩膜口径: (M,H,W) float32 物理单位, 有效域外恰好 NaN, 域内逐位保留;
     由它离线归约出的 ens_mean / spread / crps 与落盘的归约场一致
  2. 逐日对拍: compare_reduced 在同源时给 0, 被扰动时给出该扰动的量级, 缺场当场报错
  3. 身份守卫: members / steps / seed / 权重 / mode / checkpoint / D-EC 设定 / 输入产品
     任一项与落场目录记录不符即拒绝 —— 逐项都有分辨力
  4. finalize_members: 文件数、缺失日与各 rank 对拍记录的合并

运行: python -m downscaling_4x.tests.test_members_dump
============================================================================
"""
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from downscaling_4x.evaluation import jit_dump as JD
from downscaling_4x.evaluation import metrics as MT

H, W, M = 12, 20, 6


def fake_case(seed=0):
    """一天的成员与真值, 外加与 jit_dump 同一口径的陆地掩膜。"""
    rng = np.random.default_rng(seed)
    land = np.zeros((H, W), bool)
    land[2:10, 3:17] = True
    mem = (285.0 + rng.normal(0, 1.5, (M, H, W))).astype(np.float32)
    truth = 285.0 + rng.normal(0, 1.5, (H, W))
    return land, mem, truth


def write_dump(out, land, mem, truth, y=2020, day=7):
    """按 jit_dump 的写法落一份归约场, 作为"已有落场"。"""
    ens, spread = mem.mean(0), mem.std(0)
    _, crps_px = MT.crps_ensemble(mem[:, None], truth[None], land, per_pixel=True)
    JD.save_field(out, "ens_mean", y, day, np.where(land, ens, np.nan).astype(np.float32))
    JD.save_field(out, "spread", y, day, np.where(land, spread, np.nan).astype(np.float32))
    JD.save_field(out, "crps", y, day, np.where(land, crps_px[0], np.nan).astype(np.float32))
    rank_field = np.full((H, W), -1, np.int8)
    rank_field[land] = (mem[:, land] < truth[land][None]).sum(0).astype(np.int8)
    JD.save_field(out, "rank", y, day, rank_field)
    return ens, spread, np.where(land, crps_px[0], np.nan).astype(np.float32), rank_field


def test_members_field_layout_and_reduction():
    land, mem, truth = fake_case()
    f = JD.members_field(mem, land)
    assert f.dtype == np.float32 and f.shape == (M, H, W)
    assert np.isnan(f[:, ~land]).all(), "有效域外必须全是 NaN"
    assert not np.isnan(f[:, land]).any()
    assert np.array_equal(f[:, land], mem[:, land]), "域内必须逐位保留成员数值"

    with tempfile.TemporaryDirectory() as td:
        ens, spread, crps, _ = write_dump(td, land, mem, truth)
        JD.save_field(td, JD.MEMBERS_SUB, 2020, 7, f)
        back = np.load(Path(td) / JD.MEMBERS_SUB / "2020_d7.npy")
        # 离线从成员重算三件归约场, 必须回到落盘的那一份
        assert abs(back[:, land].mean(0) - ens[land]).max() < 1e-5
        assert abs(back[:, land].std(0) - spread[land]).max() < 1e-5
        _, px = MT.crps_ensemble(np.where(land, back, 0.0)[:, None], truth[None], land, per_pixel=True)
        assert abs(px[0][land] - crps[land]).max() < 1e-5
    print("  成员场口径与离线归约 OK")


def test_compare_reduced_detects_perturbation():
    land, mem, truth = fake_case(1)
    with tempfile.TemporaryDirectory() as td:
        ens, spread, crps, rank_field = write_dump(td, land, mem, truth)
        same = JD.compare_reduced(td, 2020, 7, land, ens, spread, crps, rank_field, None)
        assert max(same[k] for k in ("ens_mean", "spread", "crps")) == 0.0
        assert same["rank_mismatch_frac"] == 0.0

        bad_ens = ens.copy()
        bad_ens[land] += 0.03
        got = JD.compare_reduced(td, 2020, 7, land, bad_ens, spread, crps, rank_field, None)
        assert abs(got["ens_mean"] - 0.03) < 1e-4 and got["spread"] == 0.0

        shifted = rank_field.copy()
        shifted[land] = (shifted[land] + 1) % (M + 1)
        got = JD.compare_reduced(td, 2020, 7, land, ens, spread, crps, shifted, None)
        assert got["rank_mismatch_frac"] == 1.0, "名次整体错位必须被看见"

        (Path(td) / "spread" / "2020_d7.npy").unlink()
        try:
            JD.compare_reduced(td, 2020, 7, land, ens, spread, crps, rank_field, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("缺少可对拍的已有场时必须报错")
    print("  逐日对拍的分辨力 OK")


def base_dump_meta(td, ckpt, era5_dir):
    return {"kind": "jit_sample_dump", "target": "2m_temperature_max", "members": 32, "steps": 50,
            "seed": 0, "weight": "raw", "trained_samples": 123456, "mode": "history_51",
            "noise_scale": 2.65, "t_eps": 0.05, "patch": 16,
            "dec": {"pool": True, "drop": True, "prior": True, "eval_mode": "frame", "capacity": 1.0},
            "input": {"era5_dir": str(Path(era5_dir).absolute()), "input_product": "era5"},
            "diffusion_ckpt": JD.file_sig(ckpt)}


def test_identity_guard_rejects_each_mismatch():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        era5 = td / "era5"
        era5.mkdir()
        oracle = td / "oracle"
        oracle.mkdir()
        (oracle / "oracle_dataset_manifest.json").write_text(json.dumps({"algorithm": "blockmean"}))
        ckpt = td / "ckpt.pt"
        ckpt.write_bytes(b"x" * 64)
        out = td / "dump"
        for sub in JD.BASE_FIELDS:
            (out / sub).mkdir(parents=True)
        meta = base_dump_meta(out, ckpt, era5)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

        a = Namespace(members=32, steps=50, seed=0, ema=0, era5_dir=str(era5))
        args = {"mode": "history_51", "noise_scale": 2.65, "t_eps": 0.05, "patch": 16}
        dec = dict(meta["dec"])
        ok = lambda **kw: JD.check_members_dir(out, Namespace(**{**vars(a), **kw.pop("a", {})}),
                                               kw.get("args", args), kw.get("ckpt", ckpt),
                                               kw.get("samples", 123456), kw.get("target", "2m_temperature_max"),
                                               kw.get("dec", dec))
        ok()                                            # 全对: 放行

        def rejects(label, **kw):
            try:
                ok(**kw)
            except SystemExit:
                return
            raise AssertionError(f"身份守卫漏掉了: {label}")

        rejects("成员数", a={"members": 16})
        rejects("步数", a={"steps": 40})
        rejects("种子", a={"seed": 1})
        rejects("采样权重", a={"ema": 1})
        rejects("输入产品", a={"era5_dir": str(oracle)})
        rejects("训练进度", samples=99)
        rejects("目标变量", target="2m_temperature_min")
        rejects("条件模式", args={**args, "mode": "baseline_21"})
        rejects("噪声尺度", args={**args, "noise_scale": 3.0})
        rejects("D-EC 容量", dec={**dec, "capacity": 1.5})
        rejects("非 MoE 权重", dec=None)
        ckpt.write_bytes(b"x" * 128)
        rejects("checkpoint 内容", )
        ckpt.write_bytes(b"x" * 64)
        ok()

        # 早于 input/dec 字段的老落场: 缺的项无从核对, 但不能因此假性拒绝
        legacy = dict(meta)
        legacy.pop("input"); legacy.pop("dec")
        (out / "meta.json").write_text(json.dumps(legacy, ensure_ascii=False))
        ok()
        ok(dec=None)
        legacy_in = dict(legacy)
        legacy_in["input"] = {"era5_dir": str(oracle.absolute()), "input_product": "daymet-oracle-4x"}
        (out / "meta.json").write_text(json.dumps(legacy_in, ensure_ascii=False))
        rejects("记了输入产品且不符")
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

        (out / "spread").rmdir()
        rejects("缺归约场")
        (out / "spread").mkdir()
        (out / "meta.json").unlink()
        rejects("空目录")
    print("  身份守卫逐项分辨力 OK")


def test_routing_offset_readback():
    """成员与路由同轨迹的凭据: 补录时抽到的起点要与该成员路由里记的逐个相同。"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        assert JD.routing_offset(td, 2020, 7, 0) is None, "没有路由文件时返回 None, 跳过核对"
        (td / "routing").mkdir()
        np.savez(td / "routing" / "2020_d7_m3.npz", offset=np.array([5, 9], np.int16))
        assert JD.routing_offset(td, 2020, 7, 3) == (5, 9)
        assert JD.routing_offset(td, 2020, 7, 4) is None
    print("  路由起点读取 OK")


def test_finalize_members_counts_and_merges():
    land, mem, truth = fake_case(2)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_dump(td, land, mem, truth, day=7)
        (td / "meta.json").write_text(json.dumps({"kind": "jit_sample_dump"}))
        JD.save_field(td, JD.MEMBERS_SUB, 2020, 7, JD.members_field(mem, land))
        (td / "members_check").mkdir()
        json.dump([{"year": 2020, "day": 7, "ens_mean": 1e-6, "spread": 2e-6, "crps": 3e-6,
                    "rank_mismatch_frac": 0.0}], open(td / "members_check" / "rank0.json", "w"))
        json.dump([{"year": 2020, "day": 9, "ens_mean": 5e-6, "spread": 1e-6, "crps": 1e-6,
                    "rank_mismatch_frac": 0.01}], open(td / "members_check" / "rank1.json", "w"))

        rec = JD.finalize_members(td, [(2020, 7), (2020, 9)], 32)
        assert rec["written"] == 1 and rec["missing"] == ["2020_d9"] and rec["status"] == "incomplete"
        assert rec["check"]["n_days"] == 2 and rec["check"]["worst_day"] == "2020_d9"
        assert rec["check"]["max_abs"]["ens_mean"] == 5e-6 and rec["check"]["max_abs"]["spread"] == 2e-6
        assert rec["check"]["rank_mismatch_frac_max"] == 0.01
        assert json.load(open(td / "meta.json"))["members_dump"]["status"] == "incomplete"

        JD.save_field(td, JD.MEMBERS_SUB, 2020, 9, JD.members_field(mem, land))
        rec = JD.finalize_members(td, [(2020, 7), (2020, 9)], 32)
        assert rec["written"] == 2 and rec["missing"] == [] and rec["status"] == "done"
    print("  finalize_members 计数与合并 OK")


if __name__ == "__main__":
    test_members_field_layout_and_reduction()
    test_compare_reduced_detects_perturbation()
    test_identity_guard_rejects_each_mismatch()
    test_routing_offset_readback()
    test_finalize_members_counts_and_merges()
    print("test_members_dump: 全部通过")
