#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
EDA ② 误差分解 (证据加固, 登录节点 CPU 推理, 无训练作业)
------------------------------------------------------------------
对照最公平的一对: UNet(15ch, 整帧) vs 旧crop60 ViT(15ch, tile=60 羽化) —— 同 9 变量输入。
在同一 384px 陆地方框内:
  (A) 误差功率谱: radial_psd(pred - truth)  UNet vs ViT, 逐变量 -> ViT 到底在哪个尺度输?
      预期: 若 ViT 输在高频(子tile) => 过平滑/锐度短板(与①一致); 若输在低频 => 全局上下文问题。
  (B) ViT 窗内位置误差: 非重叠 60px 分块, 按 (i%60,j%60) 累计平方误差 -> 边缘 vs 中心。
      若边缘误差不升 => 切窗无害(坐实①的"切窗不是主因"); 若边缘升 => 存在切窗惩罚(羽化重叠可缓解)。
输出: runs/exp/20260720-eda-error-decomp/{err_psd.png, tile_pos.png, summary.json}
"""
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV
from era5_daymet.training.train_downscale import precip_inv
from era5_daymet.paths import PROJECT_ROOT

# 路径一律相对仓库根锚定, 从任何 cwd 运行都写入正式 runs/ (不再在 code/ 下误建目录)
STATS  = str(PROJECT_ROOT / "runs/stats/train_dayofyear")
OUTDIR = str(PROJECT_ROOT / "runs/exp/20260720-eda-error-decomp")
UNET_CK = str(PROJECT_ROOT / "runs/exp/20260711-unet-b64/ckpt.pt")
VIT_CK  = str(PROJECT_ROOT / "runs/exp/20260712-vit-d384-b16-ep12/ckpt.pt")
YEAR = 2020; STRIDE = 1; BOX = 384; TILE = 60; FACTOR = 6   # STRIDE=1=全年(汇报口径); 调大只为快速探索
VARS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
LAB  = {"2m_temperature_max": "tmax (K)", "2m_temperature_min": "tmin (K)", "total_precipitation_24hr": "precip (m/day)"}
os.makedirs(OUTDIR, exist_ok=True)
dev = "cpu"; torch.set_num_threads(os.cpu_count() or 8)


def radial_mean(P):
    c0, c1 = np.array(P.shape) // 2; Y, X = np.indices(P.shape)
    r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def err_psd(err):
    w = np.hanning(err.shape[0])[:, None] * np.hanning(err.shape[1])[None, :]
    return radial_mean(np.abs(np.fft.fftshift(np.fft.fft2((err - err.mean()) * w))) ** 2)


def denorm(pred_norm, s):
    p = pred_norm * s.d_std[:, None, None] + s.d_mean[:, None, None]
    p[2] = precip_inv(p[2], s.precip_scale) if s.precip_log else np.maximum(p[2], 0.0)
    return p


def main():
    t0 = time.time()
    ck_u = torch.load(UNET_CK, map_location=dev, weights_only=False)
    ck_v = torch.load(VIT_CK, map_location=dev, weights_only=False)
    au, av = ck_u["args"], ck_v["args"]
    iv = au["in_vars"]; ov = au["out_vars"]
    assert iv == av["in_vars"], "in_vars mismatch!"
    Cin = len(iv) + 3 + len(ov)
    print(f"[eda2] Cin={Cin} (should be 15)  in_vars={len(iv)}", flush=True)
    s = TD.Stats(STATS, iv, ov)

    unet = TD.UNet(Cin, len(ov), base=au.get("base", 64), temb=0).to(dev)
    unet.load_state_dict(ck_u["model"]); unet.eval()
    vit = TV.ViT(Cin, len(ov), img=av["patch"], patch=av["vit_patch"], dim=av["dim"],
                 depth=av["depth"], heads=av["heads"], mlp=av.get("mlp", 4.0)).to(dev)
    # 旧 ckpt(加 dropout 前) MLP 第二个 Linear 在 mlp.2; 当前结构(多了 Dropout)在 mlp.3 -> 重映射
    sd = {k.replace(".mlp.2.", ".mlp.3."): x for k, x in ck_v["model"].items()}
    vit.load_state_dict(sd); vit.eval()
    print(f"[eda2] models loaded  UNet={sum(p.numel() for p in unet.parameters())/1e6:.1f}M  "
          f"ViT={sum(p.numel() for p in vit.parameters())/1e6:.1f}M", flush=True)

    data = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [YEAR], iv, ov, s)
    mask = data.mask[YEAR]; by, bx, bs = TD.pick_land_box(mask, min(BOX, data.H, data.W))
    days = list(range(0, data.ndays[YEAR], STRIDE))
    print(f"[eda2] box=({by},{bx},{bs}) all-land={mask[by:by+bs,bx:bx+bs].all()}  days={days}", flush=True)

    eps_u = {v: None for v in VARS}; eps_v = {v: None for v in VARS}
    # 窗内位置误差累计 (ViT, 非重叠), 3 变量
    nt = bs // TILE                              # 每行/列整 tile 数
    pos_se = np.zeros((len(ov), TILE, TILE), np.float64); pos_n = 0

    with torch.no_grad():
        for t in days:
            cond, _, m, hr = data.full(YEAR, t)          # cond(15,H,W) 归一; hr(3,H,W) 物理真值
            cb = torch.from_numpy(cond[None]).float()
            condbox = cb[:, :, by:by+bs, bx:bx+bs]
            truth = hr[:, by:by+bs, bx:bx+bs]            # (3,bs,bs) 物理
            # --- UNet 整块 ---
            pu = denorm(unet(condbox)[0].numpy(), s)
            # --- ViT tile=60 羽化(production 口径) ---
            pv = denorm(TD.det_predict(vit, condbox, TILE, dev), s)
            for i, v in enumerate(VARS):
                eu = err_psd(pu[i] - truth[i]); ev = err_psd(pv[i] - truth[i])
                eps_u[v] = eu if eps_u[v] is None else eps_u[v] + eu
                eps_v[v] = ev if eps_v[v] is None else eps_v[v] + ev
            # --- ViT 非重叠分块 -> 窗内位置误差 ---
            for iy in range(nt):
                for ix in range(nt):
                    y0, x0 = by + iy*TILE, bx + ix*TILE
                    tp = denorm(vit(cb[:, :, y0:y0+TILE, x0:x0+TILE])[0].numpy(), s)
                    tt = hr[:, y0:y0+TILE, x0:x0+TILE]
                    pos_se += (tp - tt) ** 2; pos_n += 1
            print(f"[eda2]  day {t} done  ({time.time()-t0:.0f}s)", flush=True)

    nd = len(days)
    for v in VARS: eps_u[v] /= nd; eps_v[v] /= nd
    pos_rmse = np.sqrt(pos_se / max(pos_n, 1))          # (3,60,60)

    # ---- 图A: 误差功率谱 ----
    figA, axA = plt.subplots(1, 3, figsize=(16, 4.6))
    k_tile = bs / TILE
    band = {}
    for j, v in enumerate(VARS):
        eu = eps_u[v]; ev = eps_v[v]; k = np.arange(1, len(eu))
        ax = axA[j]
        ax.loglog(k, eu[1:], label="UNet (full-frame)", lw=2)
        ax.loglog(k, ev[1:], label="ViT crop60 (tiled)", lw=2)
        ax.axvline(k_tile, color="green", ls="--", alpha=.7, label="60px tile scale")
        ax.set_title(f"error PSD  {LAB[v]}"); ax.set_xlabel("radial wavenumber k (cyc/box)")
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
        # 分频段误差功率占比 (low: k<=tile尺度; high: k>tile尺度)
        kt = int(round(k_tile))
        band[v] = dict(unet_low=float(eu[1:kt].sum()), unet_high=float(eu[kt:].sum()),
                       vit_low=float(ev[1:kt].sum()), vit_high=float(ev[kt:].sum()))
    figA.suptitle(f"EDA-2A  error power spectrum  UNet vs ViT-crop60 (same 15ch input)   "
                  f"year={YEAR} {nd} days  box={bs}px", fontsize=12)
    figA.tight_layout(); figA.savefig(os.path.join(OUTDIR, "err_psd.png"), dpi=130, bbox_inches="tight")

    # ---- 图B: ViT 窗内位置误差 ----
    figB, axB = plt.subplots(1, len(VARS)+1, figsize=(5*(len(VARS)+1), 4.4))
    prof = {}
    for j, v in enumerate(VARS):
        hm = pos_rmse[j]
        im = axB[j].imshow(hm, origin="lower", cmap="magma"); figB.colorbar(im, ax=axB[j], fraction=.046)
        axB[j].set_title(f"ViT within-tile RMSE\n{LAB[v]}"); axB[j].set_xlabel("col % 60"); axB[j].set_ylabel("row % 60")
        # 边缘环 vs 中心: 距最近边界的距离
        Y, X = np.indices((TILE, TILE)); edge = np.minimum(np.minimum(Y, TILE-1-Y), np.minimum(X, TILE-1-X))
        rings = [float(hm[edge == d].mean()) for d in range(TILE//2)]
        prof[v] = rings
        axB[-1].plot(range(len(rings)), rings, label=LAB[v], lw=2)
    axB[-1].set_title("within-tile RMSE vs dist-to-edge"); axB[-1].set_xlabel("distance to nearest tile edge (px)")
    axB[-1].set_ylabel("RMSE (phys units)"); axB[-1].grid(True, alpha=.3); axB[-1].legend(fontsize=8)
    figB.suptitle(f"EDA-2B  ViT non-overlap within-tile error (raw tiling artifact test)  "
                  f"{pos_n} tiles", fontsize=12)
    figB.tight_layout(); figB.savefig(os.path.join(OUTDIR, "tile_pos.png"), dpi=130, bbox_inches="tight")

    json.dump(dict(year=YEAR, days=days, box=bs, tile=TILE, band_err_power=band,
                   within_tile_rmse_vs_edge=prof), open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1)
    # 打印分频段结论
    for v in VARS:
        b = band[v]
        print(f"[eda2] {v:28s} err-power high/low: UNet {b['unet_high']/max(b['unet_low'],1e-9):.2f}  "
              f"ViT {b['vit_high']/max(b['vit_low'],1e-9):.2f}  | ViT/UNet high={b['vit_high']/max(b['unet_high'],1e-9):.2f} low={b['vit_low']/max(b['unet_low'],1e-9):.2f}", flush=True)
    for v in VARS:
        r = prof[v]; print(f"[eda2] {v:28s} within-tile edge/center RMSE ratio = {r[0]/max(r[-1],1e-9):.3f}  (edge={r[0]:.4f} center={r[-1]:.4f})", flush=True)
    print(f"[eda2] DONE {time.time()-t0:.0f}s -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
