#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
port_manifest.py — 本包相对上一代包的搬运清单
============================================================================
本包是从 `era5_daymet` **复制后改动**得来的, 不是从头新建。复制的好处是把"改了什么"
压缩成可审查的 diff, 但重复代码一旦看不见就会各自漂移。这个工具把对照关系随时算出来:

  逐字节相同  该文件与合同无关, 复制未改; 上一代改了它, 这里也该同步
  已改动      给出归一化包名后的 diff 行数, 那份 diff 就是"为什么改"的完整记录
  新写        上一代没有对应物
  未搬运      上一代有而这里没有, 附带说明

运行: python -m downscaling_4x.tools.port_manifest [--diff 路径]
============================================================================
"""
import argparse
import difflib
import hashlib
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
OLD = PKG.parent / "era5_daymet"

# 新包相对路径 -> 上一代对应文件; 值为 None 表示新写
COUNTERPART = {
    "paths.py": "paths.py",
    "contract.py": "contract.py",
    "data/match_era5_daymet.py": "data/match_era5_daymet.py",
    "data/downscale_baseline.py": "data/downscale_baseline.py",
    "data/dataset.py": "data/dataset.py",
    "data/grid.py": None,
    "data/frames.py": None,
    "models/unet.py": "models/unet.py",
    "models/pos_embed.py": None,
    "models/moe_ffn.py": "models/moe_ffn.py",
    "models/jit_backbone.py": "models/jit_backbone.py",
    "models/jit_sampler.py": "models/jit_sampler.py",
    "models/jit_regressor.py": None,
    "models/song_unet.py": "models/song_unet.py",
    "models/patching.py": "models/patching.py",
    "models/corrdiff_unet.py": "models/corrdiff_unet.py",
    "models/preconditioning.py": "models/preconditioning.py",
    "models/corrdiff_loss.py": "models/corrdiff_loss.py",
    "models/stochastic_sampler.py": "models/stochastic_sampler.py",
    "data/mu_cache.py": "data/mu_cache.py",
    "evaluation/metrics.py": "evaluation/metrics.py",
    "evaluation/eval_common.py": "evaluation/eval_common.py",
    "baselines/fit_bcsd_coefs.py": "baselines/fit_bcsd_coefs.py",
    "baselines/eval_baselines.py": "baselines/train_statistical.py",
    "tests/test_spec_contract.py": "tests/test_spec_contract.py",
    "tests/test_cond_channels.py": "tests/test_cond_channels.py",
    "tests/test_causal_pairing.py": None,
    "tests/test_ddp_grad_sync.py": "tests/test_ddp_grad_sync.py",
    "tests/test_train_resume.py": "tests/test_deterministic_resume.py",
    "tests/test_train_ddp.py": None,
    "tests/test_corrdiff_contract.py": None,
    "tests/test_jit_moe.py": "tests/test_jit_moe.py",
    "tests/test_jit_backbone.py": "tests/test_jit_backbone.py",
    "tests/test_jit_sampler.py": "tests/test_jit_sampler.py",
    "tests/test_jit_patch_phase.py": "tests/test_jit_patch_phase.py",
    "tests/test_jit_ddp_sync.py": "tests/test_jit_ddp_sync.py",
    "tests/test_jit_regressor.py": None,
    "tests/test_jit_stage_b.py": None,
    "training/train_downscale.py": "training/train_downscale.py",
    "training/train_unet.py": "training/train_unet.py",
    "training/stage_b_mean.py": "training/stage_b_mean.py",
    "training/train_stage_b.py": "training/train_stage_b.py",
    "tools/build_mu_cache.py": "tools/preprocessing/build_mu_cache.py",
    "tools/residual_scale.py": None,
    "training/train_jit.py": "training/train_jit.py",
    "tools/port_manifest.py": None,
}

# 上一代有、本包**故意**不搬的, 与原因
NOT_PORTED = {
    "models/edm_diffusion.py": "SCD 线专用的 EDM 封装; CorrDiff 走 preconditioning + corrdiff_loss",
    "training/train_vit.py / train_scd.py": "ViT 与 SCD 两条线暂不启用"
    , "models/vit.py": "只需其中的 get_2d_sincos_pos_embed, 已摘进 models/pos_embed.py",
    "train_downscale 里的裁块训练、序列并行、作业内评测": "本线只做整幅; 评测走独立流程",
    "models/seq_parallel_attn.py": "序列并行只服务整幅 ViT",
    "evaluation/render/ 与 tools/plotting/**": "出图管线待新线有结果后再搬",
    "data/compute_norm_stats.py": "新数据集自带 train-only 统计, 直接复用并由 Stats 校验",
}


def _md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def _norm(text, pkg):
    """归一化包名, 使 diff 只反映实质改动。"""
    return re.sub(r"\b(era5_daymet|downscaling_4x)\b", "PKG", text)


def rows():
    out = []
    for rel in sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")):
        if rel.endswith("__init__.py"):
            continue
        new = PKG / rel
        old_rel = COUNTERPART.get(rel, "?")
        nlines = new.read_text(encoding="utf-8").count("\n") + 1
        if old_rel == "?":
            out.append((rel, "未登记", "", nlines, "请在 COUNTERPART 里登记对照关系"))
            continue
        if old_rel is None:
            out.append((rel, "新写", "", nlines, ""))
            continue
        old = OLD / old_rel
        if not old.exists():
            out.append((rel, "新写", "", nlines, f"上一代无 {old_rel}"))
            continue
        if _md5(new) == _md5(old):
            out.append((rel, "逐字节相同", old_rel, nlines, ""))
            continue
        a = _norm(old.read_text(encoding="utf-8"), "old").splitlines()
        b = _norm(new.read_text(encoding="utf-8"), "new").splitlines()
        d = list(difflib.unified_diff(a, b, lineterm="", n=0))
        chg = sum(1 for x in d if x.startswith(("+", "-")) and not x.startswith(("+++", "---")))
        out.append((rel, "已改动", old_rel, nlines, f"实质 diff {chg} 行"))
    return out


def main():
    ap = argparse.ArgumentParser(description="搬运清单")
    ap.add_argument("--diff", default="", help="打印某个文件相对上一代的实质 diff")
    a = ap.parse_args()

    if a.diff:
        old_rel = COUNTERPART.get(a.diff)
        if not old_rel:
            raise SystemExit(f"{a.diff} 没有上一代对照物")
        x = _norm((OLD / old_rel).read_text(encoding="utf-8"), "old").splitlines()
        y = _norm((PKG / a.diff).read_text(encoding="utf-8"), "new").splitlines()
        for line in difflib.unified_diff(x, y, fromfile=f"era5_daymet/{old_rel}",
                                         tofile=f"downscaling_4x/{a.diff}", lineterm=""):
            print(line)
        return

    data = rows()
    w = max(len(r[0]) for r in data) + 2
    print(f"{'文件':<{w}} {'状态':<12} {'行数':>5}  {'对照 / 说明'}")
    print("-" * (w + 46))
    for rel, kind, old_rel, n, note in data:
        tail = " ".join(x for x in (old_rel, note) if x)
        print(f"{rel:<{w}} {kind:<12} {n:>5}  {tail}")
    tot = {}
    for _, kind, *_ in data:
        tot[kind] = tot.get(kind, 0) + 1
    print("-" * (w + 46))
    print("合计: " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items())))
    print("\n未搬运(有意):")
    for k, v in NOT_PORTED.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
