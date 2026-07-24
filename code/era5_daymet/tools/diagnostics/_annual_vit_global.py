#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""给已有年均图追加 ViT-global(整幅): 载缓存 annual_means.npz(其他方法不重跑),
只算 vit_global 的 2020 年均(口径跟随 _annual_individual_maps.STRIDE, 默认全年 stride=1,
整幅 tile=0),用与现有图**相同的 vmin/vmax/biasmax** 渲染 field + bias 单张图,
并把 vit_global 写回缓存。风格/色标与 _annual_individual_maps.py 一致。
★口径护栏: 会校验缓存 npz 的 _stride 与当前 STRIDE 一致, 否则拒绝(避免把不同口径的
方法混进同一张年均图)。"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from era5_daymet.data import match_era5_daymet as M
from era5_daymet.tools.diagnostics import _annual_individual_maps as AIM
from era5_daymet.training import train_downscale as TD
from era5_daymet.training import train_vit as TV

W = AIM.W; ODIR = AIM.ODIR; STATS = AIM.STATS; YEAR = AIM.YEAR; STRIDE = AIM.STRIDE
dev = "cuda" if torch.cuda.is_available() else "cpu"
GCKPT = f"{W}/runs/exp/20260721-vit-fullframe-16node/ckpt.pt"
NPZ = f"{ODIR}/annual_means.npz"


def main():
    t0 = time.time()
    assert os.path.exists(NPZ), "缺 annual_means.npz(先跑 _annual_individual_maps.py)"
    z = np.load(NPZ); land = z["land"]
    cached_stride = int(z["_stride"]) if "_stride" in z.files else -1
    if cached_stride != STRIDE:                         # ★口径护栏: 不许把不同采样口径的方法混进同一张年均图
        raise SystemExit(f"[gmap] ✗ 缓存 annual_means.npz 口径(stride={cached_stride})≠当前(stride={STRIDE})。"
                         f"请先用相同 STRIDE 重跑 _annual_individual_maps.py 生成全年缓存, 再追加 vit_global。")
    base_methods = list(AIM.METHODS)                    # truth + 7 方法(已缓存)
    acc = {k: z[k] for k in base_methods}

    # --- 载 ViT-global(sincos+pixelshuffle, 单进程整幅前向 tile=0)---
    ck = torch.load(GCKPT, map_location=dev, weights_only=False); a = ck["args"]
    iv, ov = a["in_vars"], a["out_vars"]; Cin = len(iv) + 3 + len(ov)
    net = TV.ViT(Cin, len(ov), img=64, patch=a["vit_patch"], dim=a["dim"], depth=a["depth"],
                 heads=a["heads"], mlp=a.get("mlp", 4.0),
                 pos_type=a.get("pos_type", "sincos"), head_up=a.get("head_up", "pixelshuffle"),
                 full_frame=False).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    stats = TD.Stats(STATS, iv, ov); test = TD.DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, [YEAR], iv, ov, stats)
    pidx = ov.index(TD.PRECIP)
    def denorm(x):
        p = x * stats.d_std[:, None, None] + stats.d_mean[:, None, None]
        p[pidx] = TD.precip_inv(p[pidx], stats.precip_scale) if stats.precip_log else np.maximum(p[pidx], 0)
        return p

    days = list(range(0, test.ndays[YEAR], STRIDE))
    gm = None
    with torch.no_grad():
        for t in days:
            cond, _, _, _ = test.full(YEAR, t)
            cb = torch.from_numpy(cond[None]).float().to(dev)
            pred = denorm(TD.det_predict(net, cb, 0, dev))     # tile=0 整幅一次前向
            gm = pred.astype(np.float64) if gm is None else gm + pred
            print(f"[gmap] day {t} ({time.time()-t0:.0f}s)", flush=True)
    gm /= len(days)
    acc["vit_global"] = gm.astype(np.float32)

    # --- 渲染: vmin/vmax 用 truth, biasmax 用**原 7 方法**(与现有图同标度)---
    AIM.NAME["vit_global"] = "ViT-global"
    for vi, var in enumerate(AIM.VARS):
        disp = (lambda arr: (arr[vi] - 273.15)) if var != TD.PRECIP else (lambda arr: np.maximum(arr[vi], 0) * 1000.0)
        tr = disp(acc["truth"])[land]
        vmin, vmax = (np.percentile(tr, 2), np.percentile(tr, 98)) if var != TD.PRECIP else (0, np.percentile(tr, 99))
        biasmax = 0
        for k in base_methods[1:]:                          # 只用原方法定 biasmax -> 与已有 bias 图同标度
            b = (disp(acc[k]) - disp(acc["truth"]))[land]; biasmax = max(biasmax, np.percentile(np.abs(b), 98))
        AIM.render(disp(acc["vit_global"]), land, var, "field", "ViT-global", vmin, vmax, f"{AIM.VS[var]}_field_vit_global")
        AIM.render(disp(acc["vit_global"]) - disp(acc["truth"]), land, var, "bias", "ViT-global", -biasmax, biasmax, f"{AIM.VS[var]}_bias_vit_global")
        print(f"[gmap] {AIM.VS[var]} rendered  field[{vmin:.2f},{vmax:.2f}] biasmax={biasmax:.2f}", flush=True)

    np.savez_compressed(NPZ, land=land,
                        _stride=np.int64(STRIDE), _ndays=np.int64(len(days)), _year=np.int64(YEAR),
                        **{k: acc[k].astype(np.float32) for k in base_methods + ["vit_global"]})
    print(f"[gmap] DONE {time.time()-t0:.0f}s -> {ODIR}/{{tmax,tmin,precip}}_{{field,bias}}_vit_global.png", flush=True)


if __name__ == "__main__":
    main()
