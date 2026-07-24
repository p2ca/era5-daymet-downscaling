#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""crop60 ViT 还有没有"小格子"(tile拼缝60px / patch棋盘2px)? 高通显形 + 缝隙比值。"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUT = f"{W}/runs/exp/20260720-eda-unified/vit_grid_check.png"
VIT_CK = f"{W}/runs/exp/20260712-vit-d384-b16-ep12/ckpt.pt"
STATS = f"{W}/runs/stats/train_dayofyear"
DAY = 200; TILE = 60
dev = "cuda" if torch.cuda.is_available() else "cpu"


def laplace(a):
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    from scipy.ndimage import convolve
    return convolve(a, k, mode="reflect")


def main():
    ck = torch.load(VIT_CK, map_location=dev, weights_only=False); a = ck["args"]
    iv, ov = a["in_vars"], a["out_vars"]; Cin = len(iv) + 3 + len(ov)
    s = TD.Stats(STATS, iv, ov)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [2020], iv, ov, s)
    vit = TV.ViT(Cin, len(ov), img=a["patch"], patch=a["vit_patch"], dim=a["dim"],
                 depth=a["depth"], heads=a["heads"], mlp=a.get("mlp", 4.0)).to(dev)
    sd = ck["model"]
    if not any(k.endswith("blocks.0.mlp.3.weight") for k in sd):
        sd = {k.replace(".mlp.2.", ".mlp.3."): x for k, x in sd.items()}
    vit.load_state_dict(sd); vit.eval()

    cond, _, m, hr = d.full(2020, DAY)
    land = (m[0] if m.ndim == 3 else m) > 0.5
    by, bx, bs = TD.pick_land_box(land, 240)
    cb = torch.from_numpy(cond[None]).float().to(dev)
    with torch.no_grad():
        pred = TD.det_predict(vit, cb, TILE, dev)          # 羽化重叠(production)
    K = pred[0] * s.d_std[0] + s.d_mean[0]                 # tmax 物理 K
    truthK = hr[0]
    P = K[by:by+bs, bx:bx+bs] - 273.15
    T = truthK[by:by+bs, bx:bx+bs] - 273.15
    Lp = laplace(P); Lt = laplace(T)

    # 缝隙比值: tile 边界行/列 上的 |高通| 均值 vs 内部
    ay = np.arange(bs); seam = ((by + ay) % TILE == 0)      # 全局 tile 边界所在的行
    ax_ = np.arange(bs); seamx = ((bx + ax_) % TILE == 0)
    edge = np.abs(Lp)[seam, :].mean() if seam.any() else np.nan
    edgex = np.abs(Lp)[:, seamx].mean() if seamx.any() else np.nan
    inner = np.abs(Lp)[~seam, :][:, ~seamx].mean()
    ratio = (0.5*(edge+edgex))/max(inner, 1e-9)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    im0 = ax[0].imshow(P, origin="lower", cmap="RdYlBu_r"); fig.colorbar(im0, ax=ax[0], fraction=.046)
    ax[0].set_title(f"ViT crop60 pred tmax  day{DAY} (°C)")
    for g in range(0, bs, TILE):
        ax[0].axhline(g-0.5, color="cyan", lw=.6, alpha=.55); ax[0].axvline(g-0.5, color="cyan", lw=.6, alpha=.55)
    vmax = np.percentile(np.abs(Lt), 99)
    im1 = ax[1].imshow(Lp, origin="lower", cmap="gray_r", vmin=0, vmax=vmax); fig.colorbar(im1, ax=ax[1], fraction=.046)
    ax[1].set_title(f"ViT high-pass (Laplacian)\nseam/center |HP| ratio = {ratio:.2f}  (1.0 = no seam)")
    for g in range(0, bs, TILE):
        ax[1].axhline(g-0.5, color="cyan", lw=.6, alpha=.5); ax[1].axvline(g-0.5, color="cyan", lw=.6, alpha=.5)
    im2 = ax[2].imshow(np.abs(Lt), origin="lower", cmap="gray_r", vmin=0, vmax=vmax); fig.colorbar(im2, ax=ax[2], fraction=.046)
    ax[2].set_title("Truth high-pass (reference texture)")
    fig.suptitle(f"crop60 ViT grid-artifact check — cyan lines = 60px tile boundaries;  "
                 f"if seams/checkerboard remained, they'd light up on the tile grid in the middle panel", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"seam/center |HP| ratio = {ratio:.3f}  (edge_row={edge:.4f} edge_col={edgex:.4f} inner={inner:.4f})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
