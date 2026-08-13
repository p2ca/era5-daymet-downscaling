#!/usr/bin/env python
"""Render full-year annual-mean and bias maps for one deterministic UNet.

The production path always evaluates every available day in ``--year``.  It
also recomputes aggregate metrics while accumulating the maps, so a report
cannot silently mix full-year maps with subsampled metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from era5_daymet.data import match_era5_daymet as M
from era5_daymet.tools.plotting import plot_model_maps as PM
from era5_daymet.training import train_downscale as TD
from era5_daymet.evaluation import eval_common as EC


VAR_NAMES = {
    "2m_temperature_max": "tmax",
    "2m_temperature_min": "tmin",
    TD.PRECIP: "precip",
}
TEMP_CMAP = plt.get_cmap("RdYlBu_r").copy()
BIAS_CMAP = plt.get_cmap("RdBu_r").copy()
PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "precip",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#3f007d"],
)
for cmap in (TEMP_CMAP, BIAS_CMAP, PRECIP_CMAP):
    cmap.set_bad("#eeeeee")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(var: str, field: np.ndarray) -> np.ndarray:
    if var == TD.PRECIP:
        return np.maximum(field, 0.0) * 1000.0
    return field - 273.15


def _load_reference_scales(
    cache_path: Path | None,
    truth_mean: np.ndarray,
    land: np.ndarray,
    out_vars: list[str],
    year: int,
    n_days: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    """Use the existing report-map scale contract when a verified cache exists."""
    reference: dict[str, object] = {"cache": None, "truth_max_abs_diff": None}
    cached = None
    if cache_path is not None:
        if not cache_path.is_file():
            raise FileNotFoundError(f"reference cache does not exist: {cache_path}")
        cached = np.load(cache_path)
        cached_year = int(cached["_year"]) if "_year" in cached.files else -1
        cached_stride = int(cached["_stride"]) if "_stride" in cached.files else -1
        cached_days = int(cached["_ndays"]) if "_ndays" in cached.files else -1
        if (cached_year, cached_stride, cached_days) != (year, 1, n_days):
            raise ValueError(
                "reference cache is not full-year compatible: "
                f"year={cached_year}, stride={cached_stride}, days={cached_days}; "
                f"expected year={year}, stride=1, days={n_days}"
            )
        if not {"truth", "land"}.issubset(cached.files):
            raise ValueError("reference cache must contain truth and land")
        if not np.array_equal(cached["land"].astype(bool), land):
            raise ValueError("reference cache land mask differs from current U3 evaluation")
        truth_diff = float(np.max(np.abs(cached["truth"] - truth_mean)))
        if truth_diff > 2e-5:
            raise ValueError(
                f"reference cache truth differs from current full-year truth: max abs diff={truth_diff}"
            )
        reference = {
            "cache": str(cache_path.resolve()),
            "truth_max_abs_diff": truth_diff,
        }

    scales: dict[str, dict[str, float]] = {}
    for i, var in enumerate(out_vars):
        truth_display = _display(var, truth_mean[i])
        valid_truth = truth_display[land]
        if var == TD.PRECIP:
            field_min, field_max = 0.0, float(np.percentile(valid_truth, 99))
        else:
            field_min = float(np.percentile(valid_truth, 2))
            field_max = float(np.percentile(valid_truth, 98))

        bias_max = 0.0
        if cached is not None:
            for method in (
                "bcsd",
                "unet",
                "vit_c60",
                "vit_cos30",
                "vit_r0",
                "vit_r1",
                "corrdiff",
            ):
                if method not in cached.files:
                    continue
                bias = _display(var, cached[method][i]) - truth_display
                bias_max = max(bias_max, float(np.percentile(np.abs(bias[land]), 98)))
        scales[var] = {
            "field_min": field_min,
            "field_max": field_max,
            "bias_abs_max": bias_max,
        }
    return scales, reference


def _render(
    arr: np.ndarray,
    land: np.ndarray,
    *,
    title: str,
    unit: str,
    cmap,
    vmin: float,
    vmax: float,
    out_path: Path,
    dpi: int = 160,
) -> None:
    shown = np.where(land, arr, np.nan)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    image = ax.imshow(
        shown,
        origin="lower",
        extent=PM.EXTENT,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=PM.ASPECT,
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("lon", fontsize=10)
    ax.set_ylabel("lat", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=unit)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _metric_result(acc: dict[str, float]) -> dict[str, float | None]:
    n = max(int(acc["n"]), 1)
    mean_p = acc["sp"] / n
    mean_t = acc["st"] / n
    cov = acc["spt"] / n - mean_p * mean_t
    var_p = acc["spp"] / n - mean_p**2
    var_t = acc["stt"] / n - mean_t**2
    return {
        # Keep full precision for the cross-check against eval_metrics.json.
        # Rounding before comparison can turn a valid value just below a
        # four-decimal boundary into an apparent >1e-4 mismatch.
        "rmse": (acc["se"] / n) ** 0.5,
        "mae": acc["ae"] / n,
        "bias": acc["be"] / n,
        "corr": cov / (var_p * var_t) ** 0.5
        if var_p > 0 and var_t > 0
        else None,
    }


def _build_model(ckpt_args: dict, cin: int, cout: int, device):
    """按 checkpoint 记录的结构还原模型。含 vit_patch 键的为 ViT, 否则为 UNet。"""
    if "vit_patch" in ckpt_args:
        from era5_daymet.training import train_vit as TV
        return TV.ViT(
            cin, cout,
            img=int(ckpt_args.get("patch", 64)),
            patch=int(ckpt_args.get("vit_patch", 2)),
            dim=int(ckpt_args.get("dim", 384)),
            depth=int(ckpt_args.get("depth", 8)),
            heads=int(ckpt_args.get("heads", 12)),
            mlp=float(ckpt_args.get("mlp", 4.0)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
            drop_path=float(ckpt_args.get("drop_path", 0.0)),
            pos_type=ckpt_args.get("pos_type", "sincos"),
            head_up=ckpt_args.get("head_up", "pixelshuffle"),
            full_frame=bool(ckpt_args.get("full_frame", False)),
        ).to(device)
    return TD.build_regressor(cin, cout, ckpt_args).to(device)


class _StackedTargets(torch.nn.Module):
    """把每目标一个的单目标回归器拼成一个三通道模型。

    输出按 TARGETS 顺序在通道维拼接, 因而下游的年均场、指标与出图链路无需区分模型是
    联合训练还是分目标训练。
    """

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1)


def _load_single_target_checkpoints(paths):
    """读取每目标一个的 checkpoint, 校验它们构成一套完整且口径一致的三目标模型。

    返回 {变量名: (ckpt_args, state_dict)}。三者的输入变量、气候态开关和归一化统计必须
    一致 —— 否则拼出来的三通道预测不是同一个数据合同下的结果。
    """
    loaded = {}
    contract = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        a = payload.get("args", {})
        ov = list(a.get("out_vars", []))
        if len(ov) != 1:
            raise ValueError(f"{path}: --target-checkpoints 需要单目标 checkpoint, 实际 out_vars={ov}")
        var = ov[0]
        if var not in TD.TARGETS:
            raise ValueError(f"{path}: 未知目标 {var!r}, 应属于 {TD.TARGETS}")
        if var in loaded:
            raise ValueError(f"目标 {var!r} 提供了多个 checkpoint")
        if bool(a.get("use_clim", False)):
            raise ValueError(f"{path}: 单目标 + 气候态通道会让各目标的输入通道数不一致, 不支持拼接")
        cur = (tuple(a.get("in_vars", TD.DEFAULT_IN)), str(a.get("stats_dir", "")))
        if contract is None:
            contract = cur
        elif cur != contract:
            raise ValueError(f"{path}: 输入变量或统计目录与其他 checkpoint 不一致, 无法拼接")
        loaded[var] = (a, payload["model"])
    missing = [v for v in TD.TARGETS if v not in loaded]
    if missing:
        raise ValueError(f"缺少目标 checkpoint: {missing}")
    return loaded


def _load_expected_metrics(path, out_vars):
    """读取用于自洽性比对的指标文件, 兼容两种 schema 并在此处立即校验。

    扁平结构   {var: {...}}                     —— 训练脚本写出的 eval_metrics.json
    嵌套结构   {method: {var: {...}}, "_...": .} —— 多方法评测写出的 metrics_all.json

    键不匹配时在推理开始前就抛错, 避免跑完整年后才失败。
    """
    data = json.loads(Path(path).read_text())
    if all(v in data for v in out_vars):
        return data
    methods = [k for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)]
    for name in methods:
        if all(v in data[name] for v in out_vars):
            merged = dict(data[name])
            merged["_n_test_days"] = data.get("_n_test_days", -1)
            return merged
    raise ValueError(
        f"{path} 不含所需变量 {list(out_vars)}; 顶层键={list(data)[:6]}, 方法键={methods}"
    )


def _empty_metrics(out_vars: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        var: {
            "se": 0.0,
            "ae": 0.0,
            "be": 0.0,
            "n": 0,
            "sp": 0.0,
            "st": 0.0,
            "spp": 0.0,
            "stt": 0.0,
            "spt": 0.0,
        }
        for var in out_vars
    }


def _bilinear_geometry(height: int, width: int, factor: int):
    """Precompute the same half-pixel bilinear indices used by make_bilinear."""
    yy = np.clip((np.arange(height * factor) + 0.5) / factor - 0.5, 0, height - 1)
    xx = np.clip((np.arange(width * factor) + 0.5) / factor - 0.5, 0, width - 1)
    y0 = np.floor(yy).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    wy = (yy - y0).astype(np.float32)
    x0 = np.floor(xx).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    wx = (xx - x0).astype(np.float32)
    return y0, y1, wy, x0, x1, wx


def _full_fields_vectorized(
    test: TD.DownscaleData,
    stats: TD.Stats,
    out_vars: list[str],
    year: int,
    day: int,
    geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all dynamic input channels in one bilinear pass.

    ``DownscaleData.full`` applies the same interpolation channel by channel.
    Stacking first preserves the arithmetic for every channel while avoiding
    17 repetitions of large NumPy indexing setup and allocation.
    """
    y0, y1, wy, x0, x1, wx = geometry
    low = np.stack(
        [test.lr[year][var][day].astype(np.float32, copy=False) for var in test.in_vars],
        axis=0,
    )
    top = low[:, y0, :][:, :, x0] * (1.0 - wx)[None, None, :]
    top += low[:, y0, :][:, :, x1] * wx[None, None, :]
    bottom = low[:, y1, :][:, :, x0] * (1.0 - wx)[None, None, :]
    bottom += low[:, y1, :][:, :, x1] * wx[None, None, :]
    dynamic = top * (1.0 - wy)[None, :, None]
    dynamic += bottom * wy[None, :, None]
    dynamic = dynamic.astype(np.float32, copy=False)

    if TD.PRECIP in test.in_vars and stats.precip_log:
        precip_index = test.in_vars.index(TD.PRECIP)
        dynamic[precip_index] = TD.precip_fwd(
            dynamic[precip_index], stats.precip_clip, stats.precip_scale
        )
    dynamic = (
        (dynamic - stats.e_mean[:, None, None]) / stats.e_std[:, None, None]
    ).astype(np.float32)

    parts = [
        dynamic,
        (test.dz[year] / stats.oro_std)[None],
        ((test.lc[year] - stats.lc_mean) / stats.lc_std)[None],
        test.lsm[year][None],
    ]
    if test.use_clim:
        slot = int(test.slots[year][day])
        climatology = np.stack(
            [
                (stats.clim[var][slot] - stats.d_mean[i]) / stats.d_std[i]
                for i, var in enumerate(out_vars)
            ],
            axis=0,
        )
        parts.append(climatology)
    condition = np.concatenate(parts, axis=0).astype(np.float32)
    truth = np.stack([test._hr(year, var, day) for var in out_vars], axis=0).astype(
        np.float32
    )
    mask = test.mask[year]
    return condition, truth, mask


def _finalize_outputs(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    output_dir: Path,
    out_vars: list[str],
    land: np.ndarray,
    pred_sum: np.ndarray,
    truth_sum: np.ndarray,
    n_days: int,
    metrics: dict[str, dict[str, float | int]],
    input_channels: int,
    use_clim: bool,
) -> None:
    tag = getattr(args, "tag", "u3")
    display_name = getattr(args, "title_name", "") or tag
    expected_metrics = _load_expected_metrics(args.metrics, out_vars) if args.metrics else None
    if expected_metrics is not None:
        expected_days = int(expected_metrics.get("_n_test_days", -1))
        if expected_days != n_days:
            raise ValueError(
                f"metrics coverage={expected_days} days but merged maps cover {n_days} days"
            )

    pred_mean = (pred_sum / n_days).astype(np.float32)
    truth_mean = (truth_sum / n_days).astype(np.float32)
    result_metrics = {var: _metric_result(metrics[var]) for var in out_vars}
    if expected_metrics is not None:
        for var in out_vars:
            for key in ("rmse", "mae", "bias", "corr"):
                expected = expected_metrics[var][key]
                actual = result_metrics[var][key]
                if expected is None or actual is None:
                    if expected != actual:
                        raise ValueError(f"metric mismatch {var}/{key}: {actual} vs {expected}")
                elif abs(float(actual) - float(expected)) > max(5e-4, 5e-3 * abs(float(expected))):
                    raise ValueError(f"metric mismatch {var}/{key}: {actual} vs {expected}")

    cache_path = Path(args.reference_cache).resolve() if args.reference_cache else None
    scales, reference = _load_reference_scales(
        cache_path, truth_mean, land, out_vars, args.year, n_days
    )
    for i, var in enumerate(out_vars):
        short = VAR_NAMES[var]
        pred_display = _display(var, pred_mean[i])
        truth_display = _display(var, truth_mean[i])
        bias_display = pred_display - truth_display
        scale = scales[var]
        if scale["bias_abs_max"] <= 0:
            scale["bias_abs_max"] = (
                float(np.percentile(np.abs(bias_display[land]), 98)) or 1.0
            )
        field_unit = "mm/day" if var == TD.PRECIP else "°C"
        bias_unit = "Δmm/day" if var == TD.PRECIP else "Δ°C"
        _render(
            pred_display,
            land,
            title=f"{display_name} · {args.year} annual mean",
            unit=field_unit,
            cmap=PRECIP_CMAP if var == TD.PRECIP else TEMP_CMAP,
            vmin=scale["field_min"],
            vmax=scale["field_max"],
            out_path=output_dir / f"{short}_field_{tag}.png",
        )
        _render(
            bias_display,
            land,
            title=f"{display_name} − Daymet · {args.year} annual mean bias",
            unit=bias_unit,
            cmap=BIAS_CMAP,
            vmin=-scale["bias_abs_max"],
            vmax=scale["bias_abs_max"],
            out_path=output_dir / f"{short}_bias_{tag}.png",
        )

    np.savez_compressed(
        output_dir / f"annual_means_{tag}.npz",
        prediction=pred_mean,
        truth=truth_mean,
        land=land,
        out_vars=np.asarray(out_vars),
        _year=np.int64(args.year),
        _stride=np.int64(1),
        _ndays=np.int64(n_days),
    )
    metadata = {
        "model": tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "year": args.year,
        "eval_stride": 1,
        "n_test_days": n_days,
        "coverage": f"FULL test set ({n_days}/{n_days} days, stride 1)",
        "input_channels": input_channels,
        "use_climatology": use_clim,
        "variables": out_vars,
        "physical_units": {
            "2m_temperature_max": "K (displayed as °C)",
            "2m_temperature_min": "K (displayed as °C)",
            TD.PRECIP: "m/day (displayed as mm/day)",
        },
        "metrics_recomputed": result_metrics,
        "metrics_cross_check": str(Path(args.metrics).resolve()) if args.metrics else None,
        "reference_scale": reference,
        "scales": scales,
        "outputs": [
            f"{VAR_NAMES[var]}_{kind}_{tag}.png"
            for var in out_vars
            for kind in ("field", "bias")
        ],
    }
    (output_dir / "annual_maps_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    report_arg = "" if getattr(args, "no_embed", False) else getattr(args, "report", "")
    if report_arg:
        from era5_daymet.tools.reporting.embed_u3_annual_maps import embed

        embed(Path(report_arg), output_dir)
    print(
        f"[u3-maps] complete: {n_days}/{n_days} days, stride=1 -> {output_dir}",
        flush=True,
    )


def _save_shard(
    output_dir: Path,
    *,
    shard_index: int,
    num_shards: int,
    year: int,
    total_days: int,
    days: list[int],
    pred_sum: np.ndarray,
    truth_sum: np.ndarray,
    land: np.ndarray,
    out_vars: list[str],
    metrics: dict[str, dict[str, float | int]],
) -> Path:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}.npz"
    np.savez(
        path,
        pred_sum=pred_sum,
        truth_sum=truth_sum,
        land=land,
        out_vars=np.asarray(out_vars),
        days=np.asarray(days, dtype=np.int16),
        metrics_json=np.asarray(
            json.dumps(
                metrics,
                default=lambda value: value.item()
                if isinstance(value, np.generic)
                else value,
            )
        ),
        _year=np.int64(year),
        _total_days=np.int64(total_days),
        _shard_index=np.int64(shard_index),
        _num_shards=np.int64(num_shards),
    )
    print(f"[u3-maps shard {shard_index}] saved {len(days)} days -> {path}", flush=True)
    return path


def _merge_shards(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.out).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = payload.get("args", {})
    in_vars = list(ckpt_args.get("in_vars", TD.DEFAULT_IN))
    out_vars = list(ckpt_args.get("out_vars", TD.TARGETS))
    use_clim = bool(ckpt_args.get("use_clim", True))
    input_channels = TD.cond_channels(in_vars, out_vars, use_clim)

    pred_sum = truth_sum = land = None
    metrics = _empty_metrics(out_vars)
    seen_days: set[int] = set()
    total_days = None
    for shard_index in range(args.num_shards):
        path = (
            output_dir
            / "shards"
            / f"shard_{shard_index:02d}_of_{args.num_shards:02d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing shard: {path}")
        with np.load(path) as shard:
            if int(shard["_shard_index"]) != shard_index:
                raise ValueError(f"wrong shard index in {path}")
            if int(shard["_num_shards"]) != args.num_shards:
                raise ValueError(f"wrong shard count in {path}")
            if int(shard["_year"]) != args.year:
                raise ValueError(f"wrong year in {path}")
            shard_total = int(shard["_total_days"])
            if total_days is None:
                total_days = shard_total
            elif total_days != shard_total:
                raise ValueError("shards disagree on total test-day count")
            days = {int(day) for day in shard["days"]}
            overlap = seen_days & days
            if overlap:
                raise ValueError(f"duplicate test days across shards: {sorted(overlap)}")
            seen_days |= days
            if pred_sum is None:
                pred_sum = shard["pred_sum"].astype(np.float64)
                truth_sum = shard["truth_sum"].astype(np.float64)
                land = shard["land"].astype(bool)
            else:
                if not np.array_equal(land, shard["land"].astype(bool)):
                    raise ValueError(f"land mask differs in {path}")
                pred_sum += shard["pred_sum"]
                truth_sum += shard["truth_sum"]
            shard_metrics = json.loads(str(shard["metrics_json"]))
            for var in out_vars:
                for key in metrics[var]:
                    metrics[var][key] += shard_metrics[var][key]

    if total_days is None or pred_sum is None or truth_sum is None or land is None:
        raise ValueError("no shards were loaded")
    expected_days = set(range(total_days))
    if seen_days != expected_days:
        missing = sorted(expected_days - seen_days)
        extra = sorted(seen_days - expected_days)
        raise ValueError(f"shard coverage is not exact; missing={missing}, extra={extra}")
    print(
        f"[u3-maps merge] exact coverage verified: {len(seen_days)}/{total_days} unique days",
        flush=True,
    )
    _finalize_outputs(
        args,
        checkpoint=checkpoint,
        output_dir=output_dir,
        out_vars=out_vars,
        land=land,
        pred_sum=pred_sum,
        truth_sum=truth_sum,
        n_days=total_days,
        metrics=metrics,
        input_channels=input_channels,
        use_clim=use_clim,
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        _merge_shards(args)
        return
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"shard index must be in [0,{args.num_shards}), got {args.shard_index}"
        )

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        device_index = local_rank if local_rank < torch.cuda.device_count() else 0
        torch.cuda.set_device(device_index)
        device = f"cuda:{device_index}"
    else:
        device = "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("full-year U3 inference requires a GPU; use the Slurm submit script")

    stacked = _load_single_target_checkpoints(args.target_checkpoints) if args.target_checkpoints else None
    if stacked is not None:
        payload = None
        ckpt_args = stacked[TD.TARGETS[0]][0]        # 三者的数据合同已校验一致
        out_vars = list(TD.TARGETS)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = payload.get("args", {})
        out_vars = list(ckpt_args.get("out_vars", TD.TARGETS))
        if out_vars != TD.TARGETS:
            raise ValueError(f"expected target order {TD.TARGETS}, got {out_vars}")
    in_vars = list(ckpt_args.get("in_vars", TD.DEFAULT_IN))
    use_clim = bool(ckpt_args.get("use_clim", True))

    stats_dir = args.stats_dir or ckpt_args.get("stats_dir")
    era5_dir = args.era5_dir or ckpt_args.get("era5_dir") or M.ERA5_DIR
    daymet_dir = args.daymet_dir or ckpt_args.get("daymet_dir") or M.DAYMET_DIR
    stats = TD.Stats(stats_dir, in_vars, out_vars)
    test = TD.DownscaleData(
        era5_dir,
        daymet_dir,
        [args.year],
        in_vars,
        out_vars,
        stats,
        use_clim=use_clim,
    )
    n_days = int(test.ndays[args.year])
    if args.metrics:
        expected_metrics = _load_expected_metrics(args.metrics, out_vars)
        expected_days = int(expected_metrics.get("_n_test_days", -1))
        if expected_days != n_days:
            raise ValueError(
                f"metrics coverage={expected_days} days but dataset has {n_days} days"
            )

    cin = TD.cond_channels(in_vars, out_vars, use_clim)
    if stacked is not None:
        nets = []
        for var in TD.TARGETS:
            a, state = stacked[var]
            net = _build_model(a, cin, 1, device)
            net.load_state_dict(state)
            nets.append(net)
        model = _StackedTargets(nets).to(device)
        print(f"[u3-maps] 三个单目标 checkpoint 拼接为三通道: {TD.TARGETS}", flush=True)
    else:
        model = _build_model(ckpt_args, cin, len(out_vars), device)
        model.load_state_dict(payload["model"])
    tag_for_eval = getattr(args, "tag", "u3")
    model.eval()

    pred_sum = np.zeros((len(out_vars), test.H, test.W), dtype=np.float64)
    full_eval = None            # 首日拿到 land 后构造; 与年均场共用同一遍推理
    truth_sum = np.zeros_like(pred_sum)
    land = None
    metrics = _empty_metrics(out_vars)
    dstd = stats.d_std[:, None, None]
    dmean = stats.d_mean[:, None, None]
    geometry = _bilinear_geometry(test.Hl, test.Wl, test.f)
    started = time.time()
    days = list(range(args.shard_index, n_days, args.num_shards))
    print(
        f"[u3-maps shard {args.shard_index}/{args.num_shards}] "
        f"device={device}, days={len(days)}, first={days[0]}, last={days[-1]}",
        flush=True,
    )

    for local_day_index, day in enumerate(days):
        cond, truth, current_land = _full_fields_vectorized(
            test, stats, out_vars, args.year, day, geometry
        )
        if day == 0:
            reference_cond, _, reference_mask, reference_truth = test.full(args.year, day)
            cond_error = float(np.max(np.abs(cond - reference_cond)))
            truth_error = float(np.max(np.abs(truth - reference_truth)))
            mask_equal = np.array_equal(current_land, reference_mask[0] > 0.5)
            if cond_error > 1e-6 or truth_error != 0.0 or not mask_equal:
                raise ValueError(
                    "vectorized full-frame construction differs from canonical path: "
                    f"condition max abs={cond_error}, truth max abs={truth_error}, "
                    f"mask_equal={mask_equal}"
                )
            print(
                "[u3-maps] vectorized input verified against canonical day 0: "
                f"condition max abs={cond_error:.3g}, truth max abs={truth_error:.3g}",
                flush=True,
            )
        if land is None:
            land = current_land
        elif not np.array_equal(land, current_land):
            raise ValueError(f"land mask changed on day index {day}")

        cond_tensor = torch.from_numpy(cond[None]).float().to(device)
        normalized = TD.det_predict(model, cond_tensor, tile=0, device=device)
        pred = normalized * dstd + dmean
        if TD.PRECIP in out_vars:
            precip_index = out_vars.index(TD.PRECIP)
            pred[precip_index] = (
                TD.precip_inv(pred[precip_index], stats.precip_scale)
                if stats.precip_log
                else np.maximum(pred[precip_index], 0.0)
            )

        if full_eval is None and args.num_shards == 1:
            full_eval = EC.MultiMethodEval(
                [tag_for_eval], out_vars, test.H, test.W, land,
                precip_scale=stats.precip_scale, precip_log=stats.precip_log,
            )
        if full_eval is not None:
            full_eval.add_day(truth, land, {tag_for_eval: pred[None]})

        pred_sum += pred
        truth_sum += truth
        for i, var in enumerate(out_vars):
            p = pred[i][land].astype(np.float64)
            t = truth[i][land].astype(np.float64)
            delta = p - t
            acc = metrics[var]
            acc["se"] += float(np.dot(delta, delta))
            acc["ae"] += float(np.abs(delta).sum())
            acc["be"] += float(delta.sum())
            acc["n"] += int(delta.size)
            acc["sp"] += float(p.sum())
            acc["st"] += float(t.sum())
            acc["spp"] += float(np.dot(p, p))
            acc["stt"] += float(np.dot(t, t))
            acc["spt"] += float(np.dot(p, t))

        if (local_day_index + 1) % 10 == 0 or local_day_index + 1 == len(days):
            print(
                f"[u3-maps shard {args.shard_index}] "
                f"{local_day_index + 1}/{len(days)} days ({time.time() - started:.0f}s)",
                flush=True,
            )

    assert land is not None
    if args.num_shards > 1:
        _save_shard(
            output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            year=args.year,
            total_days=n_days,
            days=days,
            pred_sum=pred_sum,
            truth_sum=truth_sum,
            land=land,
            out_vars=out_vars,
            metrics=metrics,
        )
        return
    if full_eval is not None:
        # make_maps=False: 年均场与偏差由本脚本以单图形式输出, 不产出拼接的多面板图
        full_eval.finalize(str(output_dir), args.year, eval_stride=1, tag="all", make_maps=False)
    elif args.num_shards > 1:
        print("[maps] 分片模式: 跳过指标/SSIM(需单进程整年累加); 合并后请单独评测", flush=True)
    _finalize_outputs(
        args,
        checkpoint=checkpoint,
        output_dir=output_dir,
        out_vars=out_vars,
        land=land,
        pred_sum=pred_sum,
        truth_sum=truth_sum,
        n_days=n_days,
        metrics=metrics,
        input_channels=cin,
        use_clim=use_clim,
    )


def main() -> None:
    project = Path(__file__).resolve().parents[4]
    run_dir = project / "runs/exp/20260726-unet-U3-fullframe-16node-v3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(run_dir / "ckpt.pt"))
    parser.add_argument(
        "--target-checkpoints", nargs="+", default=[],
        help=("每目标一个的单目标 checkpoint(顺序任意, 按 out_vars 自动归位), 拼成三通道预测。"
              "给了本项则忽略 --checkpoint"),
    )
    parser.add_argument("--metrics", default=str(run_dir / "eval_metrics.json"))
    parser.add_argument("--out", default=str(run_dir / "annual_maps_2020"))
    parser.add_argument("--tag", default="u3", help="输出文件名后缀, 例如 u3 / u3v4 / vitv3")
    parser.add_argument("--title-name", default="",
                        help="图内标题使用的模型显示名, 例如 'UNet-U3' / 'ViT-V3'; 缺省用 --tag")
    parser.add_argument("--no-embed", action="store_true",
                        help="只产出图与 npz, 不自动写入 report.html")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--stats-dir", default="")
    parser.add_argument("--era5-dir", default="")
    parser.add_argument("--daymet-dir", default="")
    parser.add_argument(
        "--report", default=str(project / "docs/reports/report.html")
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--reference-cache",
        default="",
        help=(
            "Optional verified full-year annual-means cache used only to align "
            "color scales. Sampled annual-map caches must not be supplied."
        ),
    )
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
