#!/usr/bin/env python3
# Packaged implementation; the original code/ path remains compatible.
"""Diagnose fixed grid-phase bias in cached annual-mean downscaling fields.

For period ``p`` the primary diagnostic is

    I[i, j] = mean(pred - truth | y mod p = i, x mod p = j, land).

The block-demeaned variant first subtracts the mean error of every complete
land-only p x p block.  This removes broad geographical bias before averaging
by phase, making a stationary sub-pixel artifact easier to distinguish from
regional errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from era5_daymet.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_NPZ = ROOT / "runs/exp/20260720-result-maps-annual/single/annual_means.npz"
DEFAULT_OUT = ROOT / "runs/exp/20260720-midterm-report/gridphase_bias_vit_global_verified.png"
DEFAULT_JSON = ROOT / "runs/exp/20260720-midterm-report/gridphase_bias_vit_global_verified.json"

VARIABLES = (
    ("tmax", "°C", 1.0),
    ("tmin", "°C", 1.0),
    ("precip", "mm/day", 1000.0),
)


def phase_fields(
    error: np.ndarray, land: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase-conditioned mean error, MAE, and RMSE."""
    rows, cols = np.indices(error.shape)
    mean = np.full((period, period), np.nan, dtype=np.float64)
    mae = np.full_like(mean, np.nan)
    rmse = np.full_like(mean, np.nan)
    for i in range(period):
        for j in range(period):
            values = error[land & (rows % period == i) & (cols % period == j)]
            mean[i, j] = values.mean()
            mae[i, j] = np.abs(values).mean()
            rmse[i, j] = np.sqrt(np.mean(values**2))
    return mean, mae, rmse


def block_demeaned_phase(
    error: np.ndarray, land: np.ndarray, period: int
) -> tuple[np.ndarray, int]:
    """Average within-block residuals by phase over complete land-only blocks."""
    height = error.shape[0] // period * period
    width = error.shape[1] // period * period
    error_blocks = (
        error[:height, :width]
        .reshape(height // period, period, width // period, period)
        .transpose(0, 2, 1, 3)
    )
    land_blocks = (
        land[:height, :width]
        .reshape(height // period, period, width // period, period)
        .transpose(0, 2, 1, 3)
    )
    complete_land = land_blocks.all(axis=(2, 3))
    selected = error_blocks[complete_land]
    residual = selected - selected.mean(axis=(1, 2), keepdims=True)
    return residual.mean(axis=0), int(complete_land.sum())


def summarize(
    error: np.ndarray, land: np.ndarray, period: int
) -> tuple[dict[str, float | int | list[list[float]]], np.ndarray, np.ndarray]:
    phase, phase_mae, phase_rmse = phase_fields(error, land, period)
    demeaned, block_count = block_demeaned_phase(error, land, period)
    values = error[land]
    mean_abs_bias = float(np.abs(values).mean())
    summary: dict[str, float | int | list[list[float]]] = {
        "spatial_mean_bias": float(values.mean()),
        "spatial_mean_abs_bias": mean_abs_bias,
        "spatial_rmse_of_mean_bias": float(np.sqrt(np.mean(values**2))),
        "phase_mean_std": float(np.std(phase)),
        "phase_mean_range": float(np.ptp(phase)),
        "phase_mean_std_pct_of_mean_abs_bias": float(
            100.0 * np.std(phase) / mean_abs_bias
        ),
        "phase_mae_std": float(np.std(phase_mae)),
        "phase_rmse_std": float(np.std(phase_rmse)),
        "block_demeaned_phase_std": float(np.std(demeaned)),
        "block_demeaned_phase_range": float(np.ptp(demeaned)),
        "complete_land_blocks": block_count,
        "phase_mean_matrix": phase.tolist(),
        "block_demeaned_phase_matrix": demeaned.tolist(),
    }
    return summary, phase, demeaned


def draw_panel(
    axis: plt.Axes,
    field: np.ndarray,
    title: str,
    unit: str,
    std: float,
    bound: float,
    mean_bias: float | None = None,
    ratio: float | None = None,
) -> None:
    image = axis.imshow(field, cmap="RdBu_r", vmin=-bound, vmax=bound, origin="upper")
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            text_color = "white" if abs(field[i, j]) > 0.58 * bound else "black"
            axis.text(
                j,
                i,
                f"{field[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    suffix = f"phase std={std:.4f} {unit}"
    if mean_bias is not None:
        suffix = f"mean bias={mean_bias:+.4f} {unit}; {suffix}"
    if ratio is not None:
        suffix += f" ({ratio:.2f}% of mean |error|)"
    axis.set_title(f"{title}\n{suffix}", fontsize=9.5)
    axis.set_xlabel("x mod period")
    axis.set_ylabel("y mod period")
    axis.set_xticks(range(field.shape[1]))
    axis.set_yticks(range(field.shape[0]))
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label=unit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--period", type=int, default=6)
    parser.add_argument(
        "--sampling-label",
        help="Explicit provenance label for legacy NPZ files without sampling metadata",
    )
    args = parser.parse_args()

    cached = np.load(args.input)
    land = cached["land"].astype(bool)
    truth = cached["truth"]
    methods = ("vit_global", "vit_c60")
    missing = [method for method in methods if method not in cached.files]
    if missing:
        raise KeyError(f"Missing cached fields: {missing}")

    # ★口径标注跟随源 npz 的元数据(_stride/_ndays), 不再写死 -> 图/JSON 永远与数据一致
    cstride = int(cached["_stride"]) if "_stride" in cached.files else None
    cnd = int(cached["_ndays"]) if "_ndays" in cached.files else None
    if args.sampling_label:
        sampling = args.sampling_label
    elif cstride is None:
        sampling = "sampling unknown (source NPZ has no embedded metadata)"
    elif cstride == 1:
        sampling = f"2020 full year (stride=1, {cnd} days)"
    else:
        sampling = f"⚠ SUBSAMPLED 2020 (stride={cstride}, {cnd} days) — NOT full year"
    print(f"[gridphase] 源年均口径: {sampling}", flush=True)

    results: dict[str, object] = {
        "source": str(args.input),
        "period": args.period,
        "formula": "mean(pred-truth | y mod period=i, x mod period=j, land)",
        "time_sampling": sampling,
        "methods": {},
    }
    matrices: dict[tuple[str, str, str], np.ndarray] = {}

    for method in methods:
        method_result: dict[str, object] = {}
        for channel, (name, _unit, scale) in enumerate(VARIABLES):
            error = (cached[method][channel] - truth[channel]).astype(np.float64) * scale
            stats, phase, demeaned = summarize(error, land, args.period)
            method_result[name] = stats
            matrices[(method, name, "phase")] = phase
            matrices[(method, name, "demeaned")] = demeaned
        results["methods"][method] = method_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 18.0), constrained_layout=True)
    rows = (
        ("vit_global", "phase", "ViT-global: raw phase mean"),
        ("vit_c60", "phase", "ViT-crop60: raw phase mean"),
        ("vit_global", "demeaned", "ViT-global: block-demeaned"),
        ("vit_c60", "demeaned", "ViT-crop60: block-demeaned"),
    )

    # A shared scale makes the global/crop comparison visually quantitative.
    # Raw and block-demeaned panels retain separate scales because they answer
    # different questions and differ greatly in magnitude.
    shared_bounds: dict[tuple[str, str], float] = {}
    for kind in ("phase", "demeaned"):
        for name, _unit, _scale in VARIABLES:
            shared_bounds[(kind, name)] = max(
                float(np.abs(matrices[(method, name, kind)]).max())
                for method in methods
            )

    for row, (method, kind, label) in enumerate(rows):
        for col, (name, unit, _scale) in enumerate(VARIABLES):
            stats = results["methods"][method][name]
            if kind == "phase":
                std = stats["phase_mean_std"]
                mean_bias = stats["spatial_mean_bias"]
                ratio = stats["phase_mean_std_pct_of_mean_abs_bias"]
            else:
                std = stats["block_demeaned_phase_std"]
                mean_bias = None
                ratio = None
            draw_panel(
                axes[row, col],
                matrices[(method, name, kind)],
                f"{label} — {name}",
                unit,
                std,
                max(shared_bounds[(kind, name)], np.finfo(float).eps),
                mean_bias,
                ratio,
            )
    fig.suptitle(
        f"Grid-phase diagnostic, period={args.period}; {sampling} mean over land\n"
        "Raw: I(i,j)=avg(pred−truth | y mod p=i, x mod p=j). "
        "Block-demeaned: subtract each complete land-only block mean first.\n"
        "ViT-global and ViT-crop60 share color limits within each variable and diagnostic.",
        fontsize=13,
    )
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
