#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""完整 CONUS 降雨大图 —— 只画 ViT crop60, 全帧 720x1440, 修正美国比例(aspect=1/cos lat), 高分辨率可放大看小格子。"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUT = f"{W}/runs/exp/20260720-eda-unified/full_precip_map_vit.png"
VIT_CK = f"{W}/runs/exp/20260712-vit-d384-b16-ep12/ckpt.pt"
STATS = f"{W}/runs/stats/train_dayofyear"
EXTENT = [-125.125, -65.125, 23.625, 53.625]
ASPECT = 1.0 / np.cos(np.deg2rad(0.5 * (EXTENT[2] + EXTENT[3])))   # 修正经纬比例
DAY = 9
dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cv = torch.load(VIT_CK, map_location=dev, weights_only=False); a = cv["args"]
    iv, ov = a["in_vars"], a["out_vars"]; Cin = len(iv) + 3 + len(ov); pi = ov.index(TD.PRECIP)
    s = TD.Stats(STATS, iv, ov)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [2020], iv, ov, s)
    vit = TV.ViT(Cin, len(ov), img=a["patch"], patch=a["vit_patch"], dim=a["dim"],
                 depth=a["depth"], heads=a["heads"], mlp=a.get("mlp", 4.0)).to(dev)
    sd = cv["model"]
    if not any(k.endswith("blocks.0.mlp.3.weight") for k in sd):
        sd = {k.replace(".mlp.2.", ".mlp.3."): x for k, x in sd.items()}
    vit.load_state_dict(sd); vit.eval()

    cond, _, m, hr = d.full(2020, DAY)
    land = (m[0] if m.ndim == 3 else m) > 0.5
    cb = torch.from_numpy(cond[None]).float().to(dev)
    with torch.no_grad():
        pv = TD.det_predict(vit, cb, a["patch"], dev)      # ViT 羽化分块, 全帧
    p = pv[pi] * s.d_std[pi] + s.d_mean[pi]
    pr = (TD.precip_inv(p, s.precip_scale) if s.precip_log else np.maximum(p, 0.0))
    arr = np.where(land, pr * 1000.0, np.nan)              # mm/day, 海洋置 nan
    vmax = np.nanpercentile(np.where(land, hr[pi]*1000.0, np.nan), 99.0)

    cmap = LinearSegmentedColormap.from_list("pr", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#3f007d"])
    cmap.set_bad("#eeeeee")
    from datetime import date, timedelta
    ds = (date(2020, 1, 1) + timedelta(days=DAY)).isoformat()
    fig, ax = plt.subplots(figsize=(13, 7.2))
    im = ax.imshow(arr, origin="lower", extent=EXTENT, cmap=cmap, vmin=0, vmax=vmax, aspect=ASPECT)
    ax.set_title(f"ViT (crop60) — precipitation over CONUS,  {ds}", fontsize=15)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="precip (mm/day)")
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"vmax={vmax:.1f} mm/day  aspect={ASPECT:.3f}  saved -> {OUT}")


if __name__ == "__main__":
    main()
