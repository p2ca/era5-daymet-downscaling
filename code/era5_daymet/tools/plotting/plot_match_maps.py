#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
Check whether ERA5 and Daymet maps are spatially aligned.

What this file does:
  - Creates one boundary check image per year.
  - Creates one 2x2 comparison image for each requested variable and sample day.
  - Compares ERA5 low-resolution fields, native Daymet fields, Daymet block-mean
    fields, anomalies, anomaly differences, and Pearson correlation.

This script is intentionally standalone. It does not import utility.py or any
other project file, so it can be copied and run by itself.

Examples:
  python plot_match_maps.py --year 2018 --workers 4
  python plot_match_maps.py --year 2020 --workers 4 --days 15 105 195 285 \
      --vars 2m_temperature_max 2m_temperature_min total_precipitation_24hr
"""
import argparse
import gc
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


USE_CARTOPY = os.environ.get("PLOT_USE_CARTOPY", "0") == "1"
try:
    if USE_CARTOPY:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    else:
        HAS_CARTOPY = False
except Exception:
    HAS_CARTOPY = False


ERA5_DIR = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_Golden_DaymetAligned"
DAYMET_DIR = "/lustre/orion/atm112/world-shared/patrickfan/daymet/2.5_arcmin"
OUT_DIR = "/lustre/orion/atm112/world-shared/patrickfan/paired_era5_daymet"

FACTOR = 6
DAYS_PER_YEAR = 365
LEAP_DROP = "dec31"
LAND_MASK_VAR = "land_sea_mask"
LAND_THRESH = 0.5
FILL_SENTINELS = (-9999.0, -999.0, 9.969209968386869e36)

CHECK_VARS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
DEFAULT_DAYS = [15, 105, 195, 285]

LAT_EDGES_DEFAULT = (23.625, 53.625)
LON_EDGES_DEFAULT = (-125.125, -65.125)


def _shard_sort_key(path):
    """Sort shard files like 2020_0.npz, 2020_1.npz by shard number."""
    name = os.path.basename(path)
    try:
        return int(name.split("_", 1)[1].split(".", 1)[0])
    except Exception:
        return 0


def find_year_files(base_dir, year):
    """Find year NPZ files under base_dir, including train/val/test subdirectories."""
    found = []
    search_dirs = [os.path.join(base_dir, s) for s in ("train", "val", "test")] + [base_dir]
    for folder in search_dirs:
        year_file = os.path.join(folder, f"{year}.npz")
        if os.path.isfile(year_file):
            found.append(year_file)
        found.extend(sorted(glob.glob(os.path.join(folder, f"{year}_*.npz")), key=_shard_sort_key))

    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _all_dates(year):
    """Return all calendar dates in a year."""
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    out = []
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def calendar_365(year, leap_drop=LEAP_DROP):
    """Return a 365-day date list. Leap years drop Dec 31 by default."""
    days = _all_dates(year)
    if len(days) == 366:
        if leap_drop == "feb29":
            days = [d for d in days if not (d.month == 2 and d.day == 29)]
        elif leap_drop == "dec31":
            days = [d for d in days if not (d.month == 12 and d.day == 31)]
        elif leap_drop == "none":
            days = days[:365]
        else:
            raise ValueError(f"Unknown leap_drop={leap_drop!r}")
    if len(days) != DAYS_PER_YEAR:
        raise ValueError(f"{year}: expected 365 days, got {len(days)}")
    return days


def era5_dates(year):
    """Map ERA5 day indices to dates. This script uses direct index alignment."""
    return calendar_365(year, LEAP_DROP)


def _to_thw(arr):
    """Convert arrays to (T,H,W) when possible."""
    if arr.ndim == 4:
        arr = arr[:, 0]
    elif arr.ndim == 2:
        arr = arr[None]
    return arr


def clean_fill(arr):
    """Convert fill values and invalid sentinels to NaN."""
    out = arr.astype(np.float32, copy=True)
    for sentinel in FILL_SENTINELS:
        out[out == np.float32(sentinel)] = np.nan
    out[out <= -9000] = np.nan
    return out


def load_var_stack(files, var):
    """Load a variable from one or more NPZ files and concatenate over time."""
    parts = []
    for path in files:
        with np.load(path, allow_pickle=True) as npz:
            if var not in npz.files:
                continue
            arr = _to_thw(npz[var])
            if arr.ndim < 1:
                return None
            parts.append(arr)
    if not parts:
        return None
    try:
        return clean_fill(np.concatenate(parts, axis=0))
    except ValueError:
        return None


def load_static_2d(files, var):
    """Load a static 2D variable such as land_sea_mask."""
    for path in files:
        with np.load(path, allow_pickle=True) as npz:
            if var not in npz.files:
                continue
            arr = npz[var]
            if arr.ndim == 4:
                arr = arr[0, 0]
            elif arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2:
                return arr.astype(np.float32)
    return None


def block_mean(stack, factor=FACTOR):
    """Block-average a (T,H,W) stack by factor with NaN-safe means."""
    time, height, width = stack.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"{height}x{width} cannot be divided by factor={factor}")
    reshaped = stack.reshape(time, height // factor, factor, width // factor, factor)
    return np.nanmean(reshaped, axis=(2, 4))


def pearson_masked(a, b):
    """Compute Pearson correlation over shared finite pixels."""
    if a is None or b is None:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 30:
        return np.nan
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    return float((av * bv).sum() / denom) if denom != 0 else np.nan


def prep_era5_var(files, var, land_mask):
    """Load an ERA5 variable and mask non-land or invalid-domain pixels."""
    stack = load_var_stack(files, var)
    if stack is None:
        return None
    if land_mask is not None:
        stack[:, ~land_mask] = np.nan
    return stack


def _load_1d(files, names):
    """Load the first matching 1D coordinate array from a set of NPZ files."""
    for path in files:
        with np.load(path, allow_pickle=True) as npz:
            for name in names:
                if name in npz.files and npz[name].ndim == 1:
                    return npz[name].astype(float)
    return None


def geo_extent(files):
    """Infer map extent from lat/lon coordinates, falling back to the CONUS default."""
    lat = _load_1d(files, ["latitude", "lat"])
    lon = _load_1d(files, ["longitude", "lon"])

    if lat is not None and lat.size >= 2:
        dy = abs(lat[1] - lat[0])
        lat_edges = (float(lat.min() - dy / 2), float(lat.max() + dy / 2))
    else:
        lat_edges = LAT_EDGES_DEFAULT

    if lon is not None and lon.size >= 2:
        lonv = np.where(lon > 180, lon - 360, lon)
        dx = abs(lonv[1] - lonv[0])
        lon_edges = (float(lonv.min() - dx / 2), float(lonv.max() + dx / 2))
    else:
        lon_edges = LON_EDGES_DEFAULT

    return (lon_edges[0], lon_edges[1], lat_edges[0], lat_edges[1])


def detect_flip(era5_lr, daymet_lr, sample=(40, 120, 200, 300)):
    """Detect whether Daymet should be flipped in latitude to match ERA5."""
    same_scores = []
    flip_scores = []
    n_days = min(era5_lr.shape[0], daymet_lr.shape[0])
    for idx in sample:
        if idx < n_days:
            same_scores.append(pearson_masked(era5_lr[idx], daymet_lr[idx]))
            flip_scores.append(pearson_masked(era5_lr[idx], daymet_lr[idx][::-1, :]))
    same_mean = np.nanmean(same_scores) if same_scores else np.nan
    flip_mean = np.nanmean(flip_scores) if flip_scores else np.nan
    return bool(np.nan_to_num(flip_mean, nan=-1) > np.nan_to_num(same_mean, nan=-1))


def robust_range(*arrays, lo=2, hi=98):
    """Return robust percentile limits across finite values from multiple arrays."""
    chunks = [arr[np.isfinite(arr)].ravel() for arr in arrays if arr is not None]
    chunks = [chunk for chunk in chunks if chunk.size > 0]
    if not chunks:
        return 0.0, 1.0
    values = np.concatenate(chunks)
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def new_axes(nrow, ncol, figsize):
    """Create map axes, using cartopy projections when cartopy is installed."""
    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    return fig, np.atleast_2d(axes)


def draw(ax, data, extent, title, cmap, vmin, vmax, coast_mask=None):
    """Draw one map panel and optionally add coastlines from cartopy or a mask contour."""
    common = dict(origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    if HAS_CARTOPY:
        im = ax.imshow(data, transform=ccrs.PlateCarree(), **common)
        ax.coastlines(resolution="50m", linewidth=0.6, color="k")
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.4")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="k")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        im = ax.imshow(data, aspect="auto", **common)
        if coast_mask is not None:
            xs = np.linspace(extent[0], extent[1], coast_mask.shape[1])
            ys = np.linspace(extent[2], extent[3], coast_mask.shape[0])
            ax.contour(xs, ys, coast_mask, levels=[0.5], colors="k", linewidths=0.6)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=9)
    return im


def anomaly(arr):
    """Remove the spatial mean over finite pixels."""
    return arr - np.nanmean(arr)


def is_precip(var):
    """Return True for precipitation variables."""
    return "precip" in var


def is_temperature(var):
    """Return True for variables stored as Kelvin temperatures."""
    return "temperature" in var


def transform_temperature(arr):
    """Convert Kelvin temperature values to Celsius for plotting."""
    return arr - 273.15


def transform_precip(arr):
    """Convert meters per day to log1p millimeters per day for plotting."""
    return np.where(np.isfinite(arr), np.log1p(np.maximum(arr, 0) * 1000), np.nan)


def setup_year(year, era5_dir, daymet_dir, out_dir):
    """Load year-level file paths, masks, extent, and output directory."""
    era5_files = find_year_files(era5_dir, year)
    daymet_files = find_year_files(daymet_dir, year)
    if not era5_files or not daymet_files:
        print(f"[skip] {year}: missing ERA5 or Daymet files")
        return None

    fig_dir = os.path.join(out_dir, "figs", str(year))
    os.makedirs(fig_dir, exist_ok=True)
    extent = geo_extent(era5_files)

    era5_land = load_static_2d(era5_files, LAND_MASK_VAR)
    daymet_land = load_static_2d(daymet_files, LAND_MASK_VAR)
    era5_land_bool = (era5_land > LAND_THRESH) if era5_land is not None else None

    valid_mask = load_static_2d(era5_files, "valid_mask")
    if valid_mask is not None:
        valid_bool = valid_mask > 0.5
        era5_land_bool = valid_bool if era5_land_bool is None else (era5_land_bool & valid_bool)

    return {
        "era5_files": era5_files,
        "daymet_files": daymet_files,
        "fig_dir": fig_dir,
        "extent": extent,
        "era5_land": era5_land,
        "daymet_land": daymet_land,
        "era5_land_bool": era5_land_bool,
        "daymet_land_bool": (daymet_land > LAND_THRESH) if daymet_land is not None else None,
    }


def plot_boundary(year, era5_dir, daymet_dir, out_dir):
    """Create one boundary check image using only land/sea masks."""
    state = setup_year(year, era5_dir, daymet_dir, out_dir)
    if state is None or state["era5_land"] is None or state["daymet_land"] is None:
        print(f"  [skip] {year} boundary plot: missing land_sea_mask")
        return

    extent = state["extent"]
    era5_land = state["era5_land"]
    daymet_land = state["daymet_land"]
    era5_land_bool = state["era5_land_bool"]
    daymet_land_bool = state["daymet_land_bool"]

    flip = False
    if era5_land_bool is not None and daymet_land_bool is not None:
        daymet_lr = block_mean(daymet_land_bool.astype(np.float32)[None])[0] > 0.5
        daymet_lr_flip = block_mean(daymet_land_bool[::-1, :].astype(np.float32)[None])[0] > 0.5
        flip = bool(np.mean(era5_land_bool == daymet_lr_flip) > np.mean(era5_land_bool == daymet_lr))

    daymet_plot = daymet_land[::-1, :] if flip else daymet_land
    daymet_lr_mask = block_mean((daymet_plot > LAND_THRESH).astype(np.float32)[None])[0]

    fig, axes = new_axes(1, 3, (16, 4.6))
    draw(axes[0, 0], era5_land.astype(float), extent, "ERA5 land_sea_mask (120x240)",
         "Greens", 0, 1, era5_land)
    draw(axes[0, 1], daymet_plot.astype(float), extent, "Daymet land_sea_mask (720x1440)",
         "Greens", 0, 1, daymet_plot)
    draw(axes[0, 2], era5_land.astype(float), extent,
         "Overlay: ERA5 fill vs Daymet coastline (red)", "Greens", 0, 1, era5_land)

    xs = np.linspace(extent[0], extent[1], daymet_lr_mask.shape[1])
    ys = np.linspace(extent[2], extent[3], daymet_lr_mask.shape[0])
    axes[0, 2].contour(xs, ys, daymet_lr_mask, levels=[0.5], colors="red", linewidths=0.8)

    suffix = "   [Daymet flipped]" if flip else ""
    fig.suptitle(
        f"{year} CONUS boundary check "
        f"(green=land; red=Daymet coastline; coastlines should overlap){suffix}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(state["fig_dir"], "_boundary_check.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}{'  (flip)' if flip else ''}")


def plot_one_var(year, var, era5_dir, daymet_dir, out_dir, days):
    """Create comparison plots for one variable and selected days."""
    state = setup_year(year, era5_dir, daymet_dir, out_dir)
    if state is None:
        return

    extent = state["extent"]
    era5_land = state["era5_land"]
    daymet_land = state["daymet_land"]
    era5_land_bool = state["era5_land_bool"]
    daymet_land_bool = state["daymet_land_bool"]
    fig_dir = state["fig_dir"]

    era5_stack = prep_era5_var(state["era5_files"], var, era5_land_bool)
    daymet_full = load_var_stack(state["daymet_files"], var)
    if era5_stack is None or daymet_full is None:
        print(f"  [warn] {year} {var} is missing; skipped")
        return

    if daymet_land_bool is not None:
        daymet_full[:, ~daymet_land_bool] = np.nan

    daymet_lr = block_mean(daymet_full)
    flip = detect_flip(era5_stack, daymet_lr)
    if flip:
        daymet_full = daymet_full[:, ::-1, :]
        daymet_lr = daymet_lr[:, ::-1, :]

    dates = era5_dates(year)
    coast_for_daymet = daymet_land[::-1, :] if (flip and daymet_land is not None) else daymet_land

    for day_idx in days:
        if day_idx >= era5_stack.shape[0] or day_idx >= daymet_lr.shape[0]:
            continue

        era5_lr = era5_stack[day_idx]
        daymet_hr = daymet_full[day_idx]
        daymet_lr_day = daymet_lr[day_idx]
        corr = pearson_masked(era5_lr, daymet_lr_day)
        date_string = dates[day_idx].isoformat()

        if is_precip(var):
            era5_show, daymet_hr_show, daymet_lr_show = map(
                transform_precip, (era5_lr, daymet_hr, daymet_lr_day)
            )
            cmap, unit = "Blues", "log1p(mm/day)"
        elif is_temperature(var):
            era5_show, daymet_hr_show, daymet_lr_show = map(
                transform_temperature, (era5_lr, daymet_hr, daymet_lr_day)
            )
            cmap, unit = "viridis", "degC"
        else:
            era5_show, daymet_hr_show, daymet_lr_show = era5_lr, daymet_hr, daymet_lr_day
            cmap, unit = "viridis", "raw"

        vmin, vmax = robust_range(era5_show, daymet_lr_show)
        era5_anomaly = anomaly(era5_show)
        daymet_anomaly = anomaly(daymet_lr_show)
        diff = era5_anomaly - daymet_anomaly
        diff_max = (
            max(1e-9, np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98))
            if np.isfinite(diff).any()
            else 1
        )

        fig, axes = new_axes(2, 2, (14, 9))
        images = [
            (
                draw(axes[0, 0], era5_show, extent, f"ERA5 LR 120x240 [{unit}]",
                     cmap, vmin, vmax, era5_land),
                axes[0, 0],
            ),
            (
                draw(axes[0, 1], daymet_hr_show, extent, f"Daymet HR 720x1440 [{unit}]",
                     cmap, vmin, vmax, coast_for_daymet),
                axes[0, 1],
            ),
            (
                draw(axes[1, 0], daymet_lr_show, extent, f"Daymet->LR 6x block-mean [{unit}]",
                     cmap, vmin, vmax, era5_land),
                axes[1, 0],
            ),
            (
                draw(axes[1, 1], diff, extent, f"Anomaly diff (ERA5 - Daymet->LR); Pearson r={corr:.3f}",
                     "RdBu_r", -diff_max, diff_max, era5_land),
                axes[1, 1],
            ),
        ]
        for image, axis in images:
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        suffix = "   [Daymet flipped]" if flip else ""
        fig.suptitle(
            f"{year}  {var}  day_idx={day_idx}  ({date_string})   Pearson r={corr:.3f}{suffix}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(fig_dir, f"{var}_day{day_idx:03d}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path}  (corr={corr:.3f})")

    del daymet_full, daymet_lr
    gc.collect()


def plot_task(task):
    """Process-pool entry point for one boundary or variable plotting task."""
    kind, year, var, era5_dir, daymet_dir, out_dir, days = task
    try:
        if kind == "boundary":
            plot_boundary(year, era5_dir, daymet_dir, out_dir)
        else:
            plot_one_var(year, var, era5_dir, daymet_dir, out_dir, days)
        return f"{year} {var or 'boundary'}: ok"
    except Exception as exc:
        return f"{year} {var or 'boundary'}: ERROR {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="Check ERA5-Daymet spatial alignment with boundary and pattern plots.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--year", type=int, nargs="+", required=True)
    parser.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS)
    parser.add_argument("--vars", nargs="+", default=CHECK_VARS)
    parser.add_argument("--era5-dir", default=ERA5_DIR)
    parser.add_argument("--daymet-dir", default=DAYMET_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel processes. Each year-variable task may use about 1.5 GB of memory.",
    )
    args = parser.parse_args()

    workers = max(1, args.workers)
    print("=" * 64)
    print(f"Alignment plots  years={args.year}  days={args.days}  vars={args.vars}  workers={workers}")
    print(f"ERA5  : {args.era5_dir}")
    print(f"Daymet: {args.daymet_dir}")
    print(f"Output: {args.out}/figs/<year>/")
    print(f"cartopy={'yes' if HAS_CARTOPY else 'no (using land-mask contours as coastlines)'}")
    print("=" * 64)

    tasks = []
    for year in args.year:
        tasks.append(("boundary", year, None, args.era5_dir, args.daymet_dir, args.out, args.days))
        for var in args.vars:
            tasks.append(("var", year, var, args.era5_dir, args.daymet_dir, args.out, args.days))

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            for result in executor.map(plot_task, tasks):
                print(f"[done] {result}")
    else:
        for task in tasks:
            print(f"[done] {plot_task(task)}")

    print("\nDone. Review PNG files under figs/<year>/.")


if __name__ == "__main__":
    main()
