#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""给三个完整跑过的 ViT 各出一张全 CONUS 降雨大图(修正美国比例, 同天同色标, 高分辨率可放大)。"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
ODIR = f"{W}/runs/exp/20260720-eda-unified"
STATS = f"{W}/runs/stats/train_dayofyear"
EXTENT = [-125.125, -65.125, 23.625, 53.625]
ASPECT = 1.0 / np.cos(np.deg2rad(0.5 * (EXTENT[2] + EXTENT[3])))
DAY = 9
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    ("crop60 (main, 15ch)",      "vit_crop60",      f"{W}/runs/exp/20260712-vit-d384-b16-ep12/ckpt.pt"),
    ("R0 crop132 (23ch)",        "vit_R0_crop132",  f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/ckpt.pt"),
    ("R1 crop60 +reg (23ch)",    "vit_R1_crop60reg", f"{W}/runs/exp/20260718-vit-R1-crop60-reg/ckpt.pt"),
]
cmap = LinearSegmentedColormap.from_list("pr", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#3f007d"])
cmap.set_bad("#eeeeee")


def build_vit(ck, Cin, Cout):
    a = ck["args"]
    net = TV.ViT(Cin, Cout, img=a["patch"], patch=a["vit_patch"], dim=a["dim"],
                 depth=a["depth"], heads=a["heads"], mlp=a.get("mlp", 4.0)).to(dev)
    sd = ck["model"]
    if not any(k.endswith("blocks.0.mlp.3.weight") for k in sd):    # 旧 ckpt(dropout前) mlp.2 -> mlp.3
        sd = {k.replace(".mlp.2.", ".mlp.3."): x for k, x in sd.items()}
    net.load_state_dict(sd); net.eval(); return net


def main():
    from datetime import date, timedelta
    ds = (date(2020, 1, 1) + timedelta(days=DAY)).isoformat()
    data_cache = {}; vmax = None
    for label, tag, ckp in MODELS:
        ck = torch.load(ckp, map_location=dev, weights_only=False); a = ck["args"]
        iv, ov = a["in_vars"], a["out_vars"]; key = tuple(iv); pi = ov.index(TD.PRECIP)
        Cin = len(iv) + 3 + len(ov)
        if key not in data_cache:
            s = TD.Stats(STATS, iv, ov)
            d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [2020], iv, ov, s)
            data_cache[key] = (s, d)
            print(f"[maps] loaded data for {len(iv)}-var input", flush=True)
        s, d = data_cache[key]
        cond, _, m, hr = d.full(2020, DAY)
        land = (m[0] if m.ndim == 3 else m) > 0.5
        if vmax is None:
            vmax = float(np.nanpercentile(np.where(land, hr[pi]*1000.0, np.nan), 99.0))
        vit = build_vit(ck, Cin, len(ov))
        cb = torch.from_numpy(cond[None]).float().to(dev)
        with torch.no_grad():
            pv = TD.det_predict(vit, cb, a["patch"], dev)
        p = pv[pi] * s.d_std[pi] + s.d_mean[pi]
        pr = TD.precip_inv(p, s.precip_scale) if s.precip_log else np.maximum(p, 0.0)
        arr = np.where(land, pr * 1000.0, np.nan)

        fig, ax = plt.subplots(figsize=(13, 7.2))
        im = ax.imshow(arr, origin="lower", extent=EXTENT, cmap=cmap, vmin=0, vmax=vmax, aspect=ASPECT)
        ax.set_title(f"ViT {label} — precipitation over CONUS, {ds}", fontsize=14)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="precip (mm/day)")
        out = f"{ODIR}/full_precip_{tag}.png"
        fig.savefig(out, dpi=210, bbox_inches="tight"); plt.close(fig)
        print(f"[maps] {label:26s} crop={a['patch']} in={len(iv)}ch  -> {out}", flush=True)
    print(f"[maps] shared vmax={vmax:.1f} mm/day  aspect={ASPECT:.3f}  DONE", flush=True)


if __name__ == "__main__":
    main()
