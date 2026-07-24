#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""每个方法单独出图: BCSD/UNet/ViT(crop60/R0/R1)/CorrDiff + truth 的 2020 年均场 + bias(单张)。
统一色标, 修正美国比例(不标注), 供 HTML 网格排版。"""
import os, sys, time
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.tools.plotting import plot_model_maps as PM
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
ODIR = f"{W}/runs/exp/20260720-result-maps-annual/single"
STATS = f"{W}/runs/stats/train_dayofyear"
EXTENT = PM.EXTENT; ASPECT = PM.ASPECT
YEAR = 2020; STRIDE = 1          # 1=全年365天(汇报口径)。子采样只用于快速调试,
                                 # 会写进 annual_means.npz 的 _stride 元数据, 且缓存口径不符会强制重算
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(ODIR, exist_ok=True)
VARS = ["2m_temperature_max", "2m_temperature_min", TD.PRECIP]
VS = {"2m_temperature_max": "tmax", "2m_temperature_min": "tmin", TD.PRECIP: "precip"}
prcmap = LinearSegmentedColormap.from_list("pr", ["#f7fbff","#c6dbef","#6baed6","#2171b5","#08306b","#3f007d"])
prcmap.set_bad("#eeeeee"); import copy
tcmap = copy.copy(plt.get_cmap("RdYlBu_r")); tcmap.set_bad("#eeeeee")
bcmap = copy.copy(plt.get_cmap("RdBu_r")); bcmap.set_bad("#eeeeee")
# 方法显示名与画图顺序
METHODS = ["truth", "bcsd", "unet", "vit_c60", "vit_cos30", "vit_r0", "vit_r1", "corrdiff"]
NAME = {"truth":"Truth (Daymet)","bcsd":"BCSD","unet":"UNet","vit_c60":"ViT-crop60",
        "vit_cos30":"ViT-crop60-30ep","vit_r0":"ViT-crop132","vit_r1":"ViT-crop60+reg","corrdiff":"CorrDiff"}


def load_vit(ckp):
    ck = torch.load(ckp, map_location=dev, weights_only=False); a = ck["args"]
    iv, ov = a["in_vars"], a["out_vars"]; Cin = len(iv)+3+len(ov)
    net = TV.ViT(Cin, len(ov), img=a["patch"], patch=a["vit_patch"], dim=a["dim"],
                 depth=a["depth"], heads=a["heads"], mlp=a.get("mlp",4.0)).to(dev)
    sd = ck["model"]
    if not any(k.endswith("blocks.0.mlp.3.weight") for k in sd):
        sd = {k.replace(".mlp.2.",".mlp.3."): x for k,x in sd.items()}
    net.load_state_dict(sd); net.eval(); return net, a["patch"], iv, ov


def render(arr2d, land, var, mode, name, vmin, vmax, tag):
    a = np.where(land, arr2d, np.nan)
    if mode == "field":
        cmap = tcmap if var != TD.PRECIP else prcmap
        unit = "°C" if var != TD.PRECIP else "mm/day"
    else:
        cmap = bcmap; unit = "Δ°C" if var != TD.PRECIP else "Δmm/day"
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(a, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, aspect=ASPECT)
    ax.set_title(name + (" − truth" if mode == "bias" else ""), fontsize=15)
    ax.set_xlabel("lon", fontsize=10); ax.set_ylabel("lat", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=unit)
    fig.savefig(f"{ODIR}/{tag}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def _render_all(acc, land):
    for vi, var in enumerate(VARS):
        disp = lambda a: (a[vi]-273.15) if var != TD.PRECIP else (np.maximum(a[vi],0)*1000.0)
        tr = disp(acc["truth"])[land]
        vmin, vmax = (np.percentile(tr,2), np.percentile(tr,98)) if var != TD.PRECIP else (0, np.percentile(tr,99))
        biasmax = 0
        for k in METHODS[1:]:
            b = (disp(acc[k]) - disp(acc["truth"]))[land]; biasmax = max(biasmax, np.percentile(np.abs(b),98))
        for k in METHODS:
            render(disp(acc[k]), land, var, "field", NAME[k], vmin, vmax, f"{VS[var]}_field_{k}")
            if k != "truth":
                render(disp(acc[k]) - disp(acc["truth"]), land, var, "bias", NAME[k], -biasmax, biasmax, f"{VS[var]}_bias_{k}")
        print(f"[amaps] rendered {VS[var]}  field vmax={vmax:.2f} biasmax={biasmax:.2f}", flush=True)


def main():
    t0 = time.time()
    # --- P15: bcsd/unet/vit_c60/corrdiff ---
    a15 = SimpleNamespace(year=YEAR, stats_dir=STATS, era5_dir=M.ERA5_DIR, daymet_dir=M.DAYMET_DIR,
        unet_dir=f"{W}/runs/exp/20260711-unet-b64", vit_dir=f"{W}/runs/exp/20260712-vit-d384-b16-ep12",
        bcsd_coef_dir=f"{W}/runs/bcsd_coefs", corrdiff_dir=f"{W}/runs/exp/20260714-corrdiff-b64",
        regressor_ckpt=f"{W}/runs/exp/20260711-unet-b64/ckpt.pt")
    device, stats15, test15, out_vars, det_preds, corrdiff_pred = PM.build(a15)
    # --- P15 追加: vit_cos30(15ch, crop60, 30轮长调度) ---
    vit_cos30, tile_cos30, _, _ = load_vit(f"{W}/runs/exp/20260712-vit-d384-b16-cos30/ckpt.pt")
    # --- P17: vit_r0/vit_r1 ---
    vit_r0, tile_r0, iv17, ov = load_vit(f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/ckpt.pt")
    vit_r1, tile_r1, _, _ = load_vit(f"{W}/runs/exp/20260718-vit-R1-crop60-reg/ckpt.pt")
    stats17 = TD.Stats(STATS, iv17, ov); test17 = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [YEAR], iv17, ov, stats17)
    pidx = out_vars.index(TD.PRECIP)
    def denorm(x, s):
        p = x * s.d_std[:,None,None] + s.d_mean[:,None,None]
        p[pidx] = TD.precip_inv(p[pidx], s.precip_scale) if s.precip_log else np.maximum(p[pidx],0)
        return p

    NPZ = f"{ODIR}/annual_means.npz"
    if os.path.exists(NPZ):                     # 已算过年均 -> 直接重渲染, 跳过推理
        z = np.load(NPZ)
        cached_stride = int(z["_stride"]) if "_stride" in z.files else -1
        if cached_stride == STRIDE:             # ★口径一致才复用缓存, 否则(如旧 stride=10)强制重算
            land = z["land"]; acc = {k: z[k] for k in METHODS}
            nd = int(z["_ndays"]) if "_ndays" in z.files else -1
            print(f"[amaps] 载入已存年均(stride={cached_stride}, {nd}天), 跳过推理", flush=True)
            _render_all(acc, land); print(f"[amaps] DONE(render-only) {time.time()-t0:.0f}s", flush=True); return
        print(f"[amaps] ⚠ 缓存口径(stride={cached_stride})≠当前(stride={STRIDE}) -> 忽略旧缓存, 重新推理", flush=True)
    days = list(range(0, test15.ndays[YEAR], STRIDE))
    acc = {k: None for k in METHODS}; land = None
    for t in days:
        cond, _, m, hr = test15.full(YEAR, t)
        if land is None: land = (m[0] if m.ndim==3 else m) > 0.5
        cb15 = torch.from_numpy(cond[None]).float().to(dev)
        cond17, _, _, _ = test17.full(YEAR, t)
        cb17 = torch.from_numpy(cond17[None]).float().to(dev)
        vals = {
            "truth": hr,
            "bcsd": det_preds["bcsd"](cond, t)[0],
            "unet": det_preds["unet"](cond, t)[0],
            "vit_c60": det_preds["vit"](cond, t)[0],
            "vit_cos30": denorm(TD.det_predict(vit_cos30, cb15, tile_cos30, dev), stats15),
            "corrdiff": corrdiff_pred(cond, m, 0)[0],
            "vit_r0": denorm(TD.det_predict(vit_r0, cb17, tile_r0, dev), stats17),
            "vit_r1": denorm(TD.det_predict(vit_r1, cb17, tile_r1, dev), stats17),
        }
        for k in METHODS:
            acc[k] = vals[k].astype(np.float64) if acc[k] is None else acc[k] + vals[k]
        print(f"[amaps] day {t} ({time.time()-t0:.0f}s)", flush=True)
    for k in METHODS: acc[k] = acc[k] / len(days)
    np.savez_compressed(f"{ODIR}/annual_means.npz", land=land,
                        _stride=np.int64(STRIDE), _ndays=np.int64(len(days)), _year=np.int64(YEAR),
                        **{k: acc[k].astype(np.float32) for k in METHODS})
    _render_all(acc, land)
    print(f"[amaps] DONE {time.time()-t0:.0f}s -> {ODIR}", flush=True)


if __name__ == "__main__":
    main()
