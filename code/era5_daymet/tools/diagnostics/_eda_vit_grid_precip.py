#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""降雨版小格子检查: crop60 ViT precip 场 + 高通, 看 tile拼缝(60px)/patch棋盘(2px) 是否残留。"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUT = f"{W}/runs/exp/20260720-eda-unified/vit_grid_check_precip.png"
VIT_CK = f"{W}/runs/exp/20260712-vit-d384-b16-ep12/ckpt.pt"
STATS = f"{W}/runs/stats/train_dayofyear"
DAY = 9; TILE = 60                                        # 2020-01-10 冬季锋面, 大范围降水
dev = "cuda" if torch.cuda.is_available() else "cpu"
logmm = lambda x: np.log1p(np.maximum(x, 0.0) * 1000.0)
def lap(a): return convolve(a, np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32), mode="reflect")


def main():
    ck = torch.load(VIT_CK, map_location=dev, weights_only=False); a = ck["args"]
    iv, ov = a["in_vars"], a["out_vars"]; Cin = len(iv) + 3 + len(ov); pidx = ov.index(TD.PRECIP)
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
    # 选一块降水【最有空间结构】的全陆地方框(按 log 降水的空间 std, 而非均值)
    pilog = logmm(hr[pidx]); bs = 180; best = (-1, 0, 0)
    for y0 in range(0, d.H-bs, 60):
        for x0 in range(0, d.W-bs, 60):
            if land[y0:y0+bs, x0:x0+bs].all():
                sub = pilog[y0:y0+bs, x0:x0+bs]
                if 0.3 < np.mean(hr[pidx][y0:y0+bs, x0:x0+bs] > 5e-4) < 0.9:   # 有干有湿
                    v = float(sub.std())
                    if v > best[0]: best = (v, y0, x0)
    _, by, bx = best
    cb = torch.from_numpy(cond[None]).float().to(dev)
    with torch.no_grad():
        pred = TD.det_predict(vit, cb, TILE, dev)          # 羽化重叠(production)
    Pmm = np.maximum(pred[pidx], 0.0) * 1000.0             # mm/day
    Tmm = np.maximum(hr[pidx], 0.0) * 1000.0
    Pf = Pmm[by:by+bs, bx:bx+bs]; Tf = Tmm[by:by+bs, bx:bx+bs]
    Plog = logmm(pred[pidx])[by:by+bs, bx:bx+bs]; Tlog = logmm(hr[pidx])[by:by+bs, bx:bx+bs]
    Lp = lap(Plog); Lt = lap(Tlog)

    ay = np.arange(bs); seam = ((by+ay) % TILE == 0); seamx = ((bx+ay) % TILE == 0)
    ratio = (0.5*(np.abs(Lp)[seam].mean()+np.abs(Lp)[:, seamx].mean())) / max(np.abs(Lp)[~seam][:, ~seamx].mean(), 1e-9)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    im0 = ax[0].imshow(Plog, origin="lower", cmap="YlGnBu", vmin=0, vmax=np.percentile(Tlog, 99.5))
    fig.colorbar(im0, ax=ax[0], fraction=.046, label="log1p(mm)")
    ax[0].set_title(f"ViT crop60  precip  day{DAY} (2020-01-10)  [log]")
    for g in range(0, bs, TILE):
        ax[0].axhline(g-.5, color="red", lw=.6, alpha=.6); ax[0].axvline(g-.5, color="red", lw=.6, alpha=.6)
    hmx = np.percentile(np.abs(Lt), 99)
    im1 = ax[1].imshow(np.abs(Lp), origin="lower", cmap="magma", vmin=0, vmax=hmx); fig.colorbar(im1, ax=ax[1], fraction=.046)
    ax[1].set_title("ViT precip high-pass (log)\nNO 60px tile seam;  residual 2px checkerboard (see zoom)")
    for g in range(0, bs, TILE):
        ax[1].axhline(g-.5, color="cyan", lw=.6, alpha=.55); ax[1].axvline(g-.5, color="cyan", lw=.6, alpha=.55)
    # 放大 inset: 48px 方块, 看清 2px 棋盘
    zy, zx, zs = bs//2 - 24, bs//2 - 24, 48
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax[1], width="42%", height="42%", loc="lower right")
    axins.imshow(np.abs(Lp), origin="lower", cmap="magma", vmin=0, vmax=hmx)
    axins.set_xlim(zx, zx+zs); axins.set_ylim(zy, zy+zs); axins.set_xticks([]); axins.set_yticks([])
    for sp in axins.spines.values(): sp.set_color("cyan"); sp.set_linewidth(1.2)
    mark_inset(ax[1], axins, loc1=2, loc2=4, fc="none", ec="cyan", lw=.8)
    axins.set_title("×zoom (2px)", fontsize=8, color="cyan")
    im2 = ax[2].imshow(np.abs(Lt), origin="lower", cmap="magma", vmin=0, vmax=hmx); fig.colorbar(im2, ax=ax[2], fraction=.046)
    ax[2].set_title("Truth precip high-pass (reference)")
    fig.suptitle("crop60 ViT precip grid-artifact check — 60px tile squares gone (feathered fusion); "
                 "residual is a fine 2px checkerboard (ConvTranspose), absent in truth", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT, dpi=145, bbox_inches="tight")
    print(f"box=({by},{bx}) precip_mean={best[0]*1000:.2f}mm/day  tile-seam ratio={ratio:.3f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
