#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
jit_smoke_check.py — JiT/JiTMoE checkpoint 的快速采样体检
============================================================================
从训练目录读 checkpoint(结构按其中保存的 args 重建, 默认叠加 EMA1 权重), 在固定的
几个验证日上整幅采样若干集合成员, 输出:

  - metrics.json     逐日逐成员的陆地 RMSE(z 空间)、集合均值 RMSE、spread
  - map_*.png        真值 / 成员 0 / 集合均值(单图一张, 同日共享色标)
  - psd_*.png        径向平均功率谱: 真值 vs 成员均值(小尺度能量是否被生成出来)
  - scales.json      各日色标范围

指标全在 z-score 空间, 只用于配置间相对比较, 不与物理单位口径的正式评测混用。
============================================================================
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from era5_daymet.models.jit_sampler import generate
from era5_daymet import contract as C
from era5_daymet.data import dataset as DS
from era5_daymet.training.train_jit import build_model


def radial_psd(field):
    """(H, W) -> (波数中心, 径向平均功率)。波数单位: cycles/pixel。"""
    H, W = field.shape
    F = np.fft.rfft2(field)
    p = F.real ** 2 + F.imag ** 2
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    bins = np.linspace(0.0, 0.5, 101)
    idx = np.digitize(kr.ravel(), bins)
    ps = np.bincount(idx, weights=p.ravel(), minlength=bins.size + 1)
    cnt = np.bincount(idx, minlength=bins.size + 1)
    return (bins[:-1] + bins[1:]) / 2, ps[1:-1] / np.maximum(cnt[1:-1], 1)


def save_map(path, field, land, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    im = ax.imshow(np.where(land > 0.5, field, np.nan), cmap="RdYlBu_r",
                   vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="JiT checkpoint 采样体检")
    p.add_argument("--run", required=True)
    p.add_argument("--which", choices=["last", "ckpt"], default="last")
    p.add_argument("--ema", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--days", type=int, nargs="+",
                   default=[2018, 15, 2018, 196, 2019, 105, 2019, 288],
                   help="成对给出 年 日")
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run = Path(args.run)
    ck = torch.load(run / f"{args.which}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    days = list(zip(args.days[0::2], args.days[1::2]))
    years = sorted({y for y, _ in days})
    stats = DS.Stats(a["stats_dir"], C.DEFAULT_IN, C.TARGETS)
    d = DS.DownscaleData(a["era5_dir"], a["daymet_dir"], years,
                         C.DEFAULT_IN, C.TARGETS, stats)
    ti = C.TARGETS.index(a["target"])

    net = build_model(a, (d.H, d.W))
    sd = dict(ck["model"])
    if args.ema:
        for n, v in ck[f"ema{args.ema}"].items():
            sd[n] = v                      # 参数用 EMA, buffer(含路由偏置)保持 model 值
    net.load_state_dict(sd)
    net = net.to(device).eval()

    out = run / "smoke_check"
    out.mkdir(exist_ok=True)
    metrics, scales = {}, {}
    for y, t in days:
        cond, tgt, mask, _ = d.full(y, t)
        cond_t = torch.from_numpy(cond)[None].to(device)
        land_t = torch.from_numpy((mask[0] > 0.5).astype("float32"))[None, None].to(device)
        truth = torch.from_numpy(tgt[ti:ti + 1])[None].to(device) * land_t

        members = []
        for m in range(args.members):
            g = torch.Generator(device=device)
            g.manual_seed(args.seed * 10_000 + m * 100 + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                s = generate(net, cond_t, steps=args.steps,
                             noise_scale=a["noise_scale"], t_eps=a["t_eps"],
                             land=land_t, generator=g)
            # 海洋清零后再进指标与功率谱: 采样钳制只把海洋收敛到近 0(末端 clamp 残
            # 差), 而真值海洋精确为 0, 不清零会给成员的高波数谱垫一层噪声底
            members.append(s.float() * land_t)
        ens = torch.cat(members, dim=0)                        # (M, 1, H, W)
        lsum = land_t.sum()

        def lrmse(x):
            return float((((x - truth) ** 2 * land_t).sum() / lsum).sqrt())

        key = f"{y}d{t:03d}"
        metrics[key] = {
            "member_rmse": [lrmse(m_) for m_ in members],
            "ensmean_rmse": lrmse(ens.mean(0, keepdim=True)),
            "spread": float(((ens.std(0, unbiased=False) * land_t).sum() / lsum)),
        }

        tr = truth[0, 0].cpu().numpy()
        em = ens.mean(0)[0, 0].cpu().numpy()
        m0 = members[0][0, 0].cpu().numpy()
        land_np = land_t[0, 0].cpu().numpy()
        v = tr[land_np > 0.5]
        vmin, vmax = float(np.quantile(v, 0.005)), float(np.quantile(v, 0.995))
        scales[key] = {"vmin": vmin, "vmax": vmax}
        save_map(out / f"map_truth_{key}.png", tr, land_np, vmin, vmax,
                 f"truth {key} (z)")
        save_map(out / f"map_member0_{key}.png", m0, land_np, vmin, vmax,
                 f"member0 {key} (z)")
        save_map(out / f"map_ensmean_{key}.png", em, land_np, vmin, vmax,
                 f"ens-mean {key} (z)")

        k, pt = radial_psd(tr)
        pm = np.mean([radial_psd(mm[0, 0].cpu().numpy())[1] for mm in members], axis=0)
        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
        ax.loglog(k[1:], pt[1:], label="truth")
        ax.loglog(k[1:], pm[1:], label=f"member mean (n={args.members})")
        ax.set_xlabel("wavenumber (cycles/px)"); ax.set_ylabel("radial PSD")
        ax.set_title(f"{key} (z)", fontsize=9); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"psd_{key}.png"); plt.close(fig)
        metrics[key]["psd_ratio_hi"] = float(pm[60:].mean() / max(pt[60:].mean(), 1e-12))

        print(f"[smoke] {key}: member_rmse={metrics[key]['member_rmse'][0]:.4f} "
              f"ens_rmse={metrics[key]['ensmean_rmse']:.4f} "
              f"spread={metrics[key]['spread']:.4f} "
              f"psd_hi_ratio={metrics[key]['psd_ratio_hi']:.3f}", flush=True)

    meta = {"run": str(run), "which": args.which, "ema": args.ema,
            "members": args.members, "steps": args.steps,
            "samples_trained": int(ck.get("samples", -1)),
            "noise_scale": a["noise_scale"], "space": "z-score",
            "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    (out / "scales.json").write_text(json.dumps(scales, indent=1))
    print(f"[smoke] -> {out}")


if __name__ == "__main__":
    main()
