#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""可行性实测(改造后版本): 整幅 720x1440 ViT 单样本前向+反向, 量峰值显存 + 单步耗时。

与旧版区别: 直接构建新结构 ViT(2D sin-cos 固定位置编码 + PixelShuffle 上采样头, full_frame=True),
不再加载老 crop ckpt / 插值 pos。输入 = 17 变量(DEFAULT_IN) -> Cin=23。目的是实测
17 变量整幅的显存/耗时, 确认 bf16 塞得下单卡 64G。
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
STATS = f"{W}/runs/stats/train_dayofyear"
dev = "cuda"
# 鼓励内存高效/flash SDPA 后端(nn.MultiheadAttention need_weights=False 会走 SDPA)
try:
    torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception as e:
    print("sdp toggle:", e)


def main():
    iv, ov = TD.DEFAULT_IN, TD.TARGETS                       # 17 变量输入 / 3 目标
    Cin = len(iv) + 3 + len(ov)                              # 17 + (dz,lc,lsm) + 3 clim = 23
    patch, dim, depth, heads = 2, 384, 8, 6
    gh, gw = 720 // patch, 1440 // patch
    print(f"[probe] 整幅 ViT: Cin={Cin}(17变量+3静态+3气候态) patch={patch} dim={dim} depth={depth} "
          f"heads={heads}  tokens={gh}x{gw}={gh*gw}  pos=sincos head_up=pixelshuffle", flush=True)

    model = TV.ViT(Cin, len(ov), img=64, patch=patch, dim=dim, depth=depth, heads=heads, mlp=4.0,
                   pos_type="sincos", head_up="pixelshuffle", full_frame=True).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[probe] params={n_par/1e6:.1f}M (sincos 无位置参数, 与 crop 权重同规模)", flush=True)

    # 真取一帧完整输入(2020 第 200 天)
    s = TD.Stats(STATS, iv, ov)
    d = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [2020], iv, ov, s)
    cond, tgt, m, _ = d.full(2020, 200)
    x = torch.from_numpy(cond[None]).float().to(dev)         # (1,23,720,1440)
    t = torch.from_numpy(tgt[None]).float().to(dev)          # (1,3,720,1440)
    print(f"[probe] input {tuple(x.shape)}  target {tuple(t.shape)}", flush=True)

    for amp in [False, True]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x)
                loss = F.mse_loss(out.float(), t)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            model.zero_grad(set_to_none=True)
            print(f"[probe] {'bf16' if amp else 'fp32'}:  OK  峰值显存={peak:.1f} GB  "
                  f"单步(fwd+bwd)={dt:.1f}s  out={tuple(out.shape)}  loss={loss.item():.4f}", flush=True)
        except RuntimeError as e:
            print(f"[probe] {'bf16' if amp else 'fp32'}:  OOM/ERR -> {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
