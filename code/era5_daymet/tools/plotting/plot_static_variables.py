#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
Plot static Daymet variables from static.npz.

The script creates one overview figure containing:
  land_sea_mask, latitude, orography, and landcover.

It can also save one PNG per variable for closer inspection.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STATIC = "/lustre/orion/atm112/world-shared/patrickfan/daymet/2.5_arcmin_targets_static/static.npz"
DEFAULT_OUT = "static_variable_plots"
EXTENT = [-125.125, -65.125, 23.625, 53.625]
VARS = ["land_sea_mask", "latitude", "orography", "landcover"]


def load_2d(npz, name):
    """Load a variable as a 2D array."""
    arr = np.asarray(npz[name])
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{name} is not 2D after squeezing: shape={arr.shape}")
    return arr


def masked_for_display(name, arr, land_mask):
    """Mask ocean pixels for static variables where that helps visual inspection."""
    out = arr.astype(np.float32, copy=True)
    if name not in ("land_sea_mask", "latitude") and land_mask is not None:
        out[land_mask <= 0.5] = np.nan
    return out


def color_settings(name, arr):
    """Return colormap and robust limits for a static variable."""
    if name == "land_sea_mask":
        return "gray_r", 0.0, 1.0
    if name == "latitude":
        return "viridis", float(np.nanmin(arr)), float(np.nanmax(arr))
    if name == "orography":
        vals = arr[np.isfinite(arr)]
        return "terrain", float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
    if name == "landcover":
        vals = arr[np.isfinite(arr)]
        return "tab20", float(np.nanmin(vals)), float(np.nanmax(vals))
    vals = arr[np.isfinite(arr)]
    return "viridis", float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))


def plot_panel(ax, name, arr):
    """Draw one static variable panel."""
    cmap, vmin, vmax = color_settings(name, arr)
    im = ax.imshow(arr, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"{name}  shape={arr.shape}", fontsize=10)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return im


def main():
    parser = argparse.ArgumentParser(description="Plot static variables from a Daymet static.npz file.")
    parser.add_argument("--static-file", default=DEFAULT_STATIC)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--vars", nargs="+", default=VARS)
    parser.add_argument("--single", action="store_true", help="Also save one PNG per variable.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with np.load(args.static_file, allow_pickle=True) as npz:
        missing = [name for name in args.vars if name not in npz.files]
        if missing:
            raise SystemExit(f"Missing variables in {args.static_file}: {missing}")
        land_mask = load_2d(npz, "land_sea_mask") if "land_sea_mask" in npz.files else None
        arrays = {name: masked_for_display(name, load_2d(npz, name), land_mask) for name in args.vars}

    n = len(args.vars)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.8 * rows), squeeze=False)
    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        if idx >= n:
            ax.axis("off")
            continue
        name = args.vars[idx]
        im = plot_panel(ax, name, arrays[name])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Daymet static variables", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    overview = os.path.join(args.out_dir, "static_variables_overview.png")
    fig.savefig(overview, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {overview}")

    if args.single:
        for name in args.vars:
            fig, ax = plt.subplots(figsize=(9, 5))
            im = plot_panel(ax, name, arrays[name])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = os.path.join(args.out_dir, f"{name}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"-> {out}")


if __name__ == "__main__":
    main()
