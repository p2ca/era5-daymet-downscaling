"""R0 阶梯感诊断: 用 ep6 best ckpt 在一个 test 日上做分块推理(det_predict 已含加权融合),
画 tmax 场 + 拉普拉斯高通(分块缝隙/阶梯会在高通图上现出亮线)。只跑 ViT, 不碰 corrdiff。"""
# Packaged implementation; the original code/ path remains compatible.
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as VM

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
CK = f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/ckpt.pt"
DEV = "cuda"
DAY = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # 仲夏, tmax 平滑, 阶梯最易见

a = torch.load(CK, map_location="cpu")["args"]
in_vars, out_vars = a["in_vars"], a["out_vars"]
stats = TD.Stats(a["stats_dir"], in_vars, out_vars)
test = TD.DownscaleData(a["era5_dir"], a["daymet_dir"], [a["test_year"]], in_vars, out_vars, stats)
Cin, Cout = len(in_vars) + 3 + len(out_vars), len(out_vars)
tile = a["patch"]                                       # 132

net = VM.ViT(Cin, Cout, img=tile, patch=a["vit_patch"], dim=a["dim"],
             depth=a["depth"], heads=a["heads"], mlp=a["mlp"]).to(DEV)
net.load_state_dict(torch.load(CK, map_location=DEV)["model"]); net.eval()
print(f"[r0] loaded ep6 ckpt, tile={tile}, Cin={Cin}", flush=True)

cond, _, m, hr = test.full(a["test_year"], DAY)
land = (m[0] if m.ndim == 3 else m) > 0.5
with torch.no_grad():
    o = TD.det_predict(net, torch.from_numpy(cond[None]).float().to(DEV), tile, DEV)  # (Cout,H,W) 归一
pred = o * stats.d_std[:, None, None] + stats.d_mean[:, None, None]                    # 物理
ti = out_vars.index("2m_temperature_max")
pt, tt = pred[ti], hr[ti]
pt = np.where(land, pt, np.nan); tt = np.where(land, tt, np.nan)
lap = np.where(land, np.abs(laplace(np.nan_to_num(pred[ti]))), np.nan)                 # 高通: 缝隙现亮线

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
vmin, vmax = np.nanpercentile(tt, 2), np.nanpercentile(tt, 98)
ax[0].imshow(tt, cmap="RdYlBu_r", vmin=vmin, vmax=vmax); ax[0].set_title(f"Truth tmax  day={DAY}")
ax[1].imshow(pt, cmap="RdYlBu_r", vmin=vmin, vmax=vmax); ax[1].set_title("R0 ViT (23ch, crop132, feathered)")
im = ax[2].imshow(lap, cmap="magma", vmin=0, vmax=np.nanpercentile(lap, 99))
ax[2].set_title("|Laplacian| (阶梯/缝隙 = 亮线)")
for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
# 标出 132-tile 边界(det_predict step = tile - tile//4 = 99)
step = tile - max(tile // 4, 1)
H, Wd = pt.shape
for b in range(step, max(H, Wd), step):
    if b < H: ax[2].axhline(b, color="cyan", lw=0.4, alpha=0.5)
    if b < Wd: ax[2].axvline(b, color="cyan", lw=0.4, alpha=0.5)
plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout()
out = f"{W}/runs/exp/20260717-vit-R0-23ch-crop132/r0_staircase_day{DAY}.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); print(f"[r0] saved {out}", flush=True)
print(f"[r0] tmax 高通均值(陆地)={np.nanmean(lap):.4f}  峰值={np.nanmax(lap):.4f}", flush=True)
