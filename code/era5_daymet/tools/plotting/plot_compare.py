#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
Plot a simple side-by-side comparison for one ERA5 field and one Daymet field.

What this file does:
  - Loads one variable for one day from ERA5 and Daymet NPZ files.
  - Masks Daymet ocean pixels with land_sea_mask when available.
  - Saves a two-panel PNG using one shared color scale.

This script is intentionally standalone. It does not import utility.py or any
other project file, so it can be copied and run by itself.

Examples:
  python plot_compare.py --year 2020 --var 2m_temperature_max --day -1
  python plot_compare.py --era5-file /path/to/2020.npz \
                         --daymet-file /path/to/2020_0.npz \
                         --var total_precipitation_24hr --day 10
"""
import argparse
import glob
import os
import struct
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ERA5_DIR = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_Golden_DaymetAligned"
DAYMET_DIR = "/lustre/orion/atm112/world-shared/patrickfan/daymet/2.5_arcmin"
OUT_DIR = "/lustre/orion/atm112/world-shared/patrickfan/paired_era5_daymet"

EXTENT = [-125.125, -65.125, 23.625, 53.625]
LAND_MASK_VAR = "land_sea_mask"


def is_temperature(var):
    """Return True for variables stored as Kelvin temperatures."""
    return "temperature" in var


def is_precip(var):
    """Return True for precipitation variables stored as meters per day."""
    return "precip" in var


def temperature_to_celsius(arr):
    """Convert Kelvin temperature values to Celsius for plotting."""
    return arr - 273.15


def precip_to_log_mm_per_day(arr):
    """Convert meters per day to log1p millimeters per day for plotting."""
    return np.log1p(np.maximum(arr, 0) * 1000)


def find_year_files(base_dir, year):
    """Find year NPZ files under base_dir, including train/val/test subdirectories."""
    found = []
    search_dirs = [os.path.join(base_dir, s) for s in ("train", "val", "test")] + [base_dir]
    for folder in search_dirs:
        year_file = os.path.join(folder, f"{year}.npz")
        if os.path.isfile(year_file):
            found.append(year_file)
        shard_pattern = os.path.join(folder, f"{year}_*.npz")
        found.extend(sorted(glob.glob(shard_pattern), key=_shard_sort_key))

    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _shard_sort_key(path):
    """Sort shard files like 2020_0.npz, 2020_1.npz by shard number."""
    name = os.path.basename(path)
    try:
        return int(name.split("_", 1)[1].split(".", 1)[0])
    except Exception:
        return 0


def _read_npy_header(fp, version):
    """Read a .npy header in a way that works across NumPy versions."""
    fmt = np.lib.format
    if hasattr(fmt, "_read_array_header"):
        return fmt._read_array_header(fp, version)
    reader = getattr(fmt, f"read_array_header_{version[0]}_0", None)
    if reader is None:
        raise ValueError(f"Unsupported .npy version: {version}")
    return reader(fp)


def npz_memmap(path, var):
    """Memory-map an uncompressed NPZ member. Return None for missing/compressed arrays."""
    with zipfile.ZipFile(path) as zf:
        name = var + ".npy"
        if name not in zf.namelist():
            return None
        zip_info = zf.getinfo(name)
        if zip_info.compress_type != zipfile.ZIP_STORED:
            return None

    with open(path, "rb") as fp:
        fp.seek(zip_info.header_offset)
        name_len, extra_len = struct.unpack("<HH", fp.read(30)[26:30])
        fp.seek(zip_info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(fp)
        shape, fortran_order, dtype = _read_npy_header(fp, version)
        data_offset = fp.tell()

    if fortran_order:
        return None
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=data_offset)


def load2d(path, var, day=None):
    """Load one variable from one NPZ file and return a 2D float32 array."""
    memmap = npz_memmap(path, var)
    if memmap is not None:
        arr = np.asarray(memmap[day] if day is not None else memmap)
        while arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.float32)

    with np.load(path, allow_pickle=True) as npz:
        if var not in npz.files:
            raise SystemExit(
                f"[error] {path}\n"
                f"  Variable {var!r} was not found.\n"
                f"  Available variables: {list(npz.files)}"
            )
        arr = npz[var]
        if day is not None:
            arr = arr[day]
        while arr.ndim > 2:
            arr = arr[0]
        return np.asarray(arr, dtype=np.float32)


def resolve_file(label, explicit_file, base_dir, year):
    """Use an explicit file when provided, otherwise pick the first file for year."""
    if explicit_file:
        return explicit_file
    files = find_year_files(base_dir, year)
    if not files:
        raise SystemExit(f"[error] Could not find {label} NPZ files for year {year} under {base_dir}")
    return files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Plot one ERA5 panel and one Daymet panel with a shared color scale."
    )
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--era5-dir", default=ERA5_DIR)
    parser.add_argument("--daymet-dir", default=DAYMET_DIR)
    parser.add_argument("--era5-file", default=None)
    parser.add_argument("--daymet-file", default=None)
    parser.add_argument("--var", default="2m_temperature_max")
    parser.add_argument("--day", type=int, default=-1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    era5_file = resolve_file("ERA5", args.era5_file, args.era5_dir, args.year)
    daymet_file = resolve_file("Daymet", args.daymet_file, args.daymet_dir, args.year)
    era5 = load2d(era5_file, args.var, args.day)
    daymet = load2d(daymet_file, args.var, args.day)

    try:
        land_mask = load2d(daymet_file, LAND_MASK_VAR) > 0.5
        daymet = np.where(land_mask, daymet, np.nan)
    except SystemExit:
        land_mask = None

    if is_precip(args.var):
        era5 = precip_to_log_mm_per_day(era5)
        daymet = precip_to_log_mm_per_day(daymet)
        unit = "log1p(mm/day)"
    elif is_temperature(args.var):
        era5 = temperature_to_celsius(era5)
        daymet = temperature_to_celsius(daymet)
        unit = "degC"
    else:
        unit = "raw"

    finite_values = np.concatenate([era5[np.isfinite(era5)], daymet[np.isfinite(daymet)]])
    if finite_values.size == 0:
        raise SystemExit("[error] No finite values were found for plotting.")
    vmin, vmax = np.percentile(finite_values, 2), np.percentile(finite_values, 98)
    cmap = "Blues" if is_precip(args.var) else "viridis"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    panels = [(axes[0], era5, "ERA5 (120x240)"), (axes[1], daymet, "Daymet (720x1440)")]
    for axis, image, title in panels:
        im = axis.imshow(
            image,
            origin="lower",
            extent=EXTENT,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_title(title, fontsize=11)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{args.var}   year={args.year}   day_idx={args.day}   [{unit}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = args.out or os.path.join(OUT_DIR, f"era5_vs_daymet_{args.var}_day{args.day}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"-> {out}   (vmin/vmax={vmin:.2f}/{vmax:.2f})")
    if land_mask is None:
        print("[warn] Daymet land_sea_mask was not found; Daymet ocean pixels were not masked.")


if __name__ == "__main__":
    main()
