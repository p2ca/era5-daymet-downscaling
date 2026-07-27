#!/usr/bin/env python
# Packaged implementation; the original code/ path remains compatible.
# -*- coding: utf-8 -*-
"""
EDA ④ CorrDiff 在这批数据上的特点 (登录节点 GPU, 复用 plot_model_maps.build)
------------------------------------------------------------------
CorrDiff = 生成式/集合 (回归器 μ + EDM 扩散残差)。要确认两条特点:
  锐度轴 (解释 RMSE 垫底): 功率谱 truth vs UNet/ViT(确定性,高频衰减) vs CorrDiff 均值(也衰减)
                          vs CorrDiff 单成员(恢复高频、贴合 truth)。
  校准轴 (解释 CRPS 全优): rank histogram(平=校准好) + spread-skill(集合发散度 vs 均值误差)。
在若干事件日的 384px 陆地方框内计算。tmax 用 K, precip 用 log1p(mm)。
输出: runs/exp/20260720-eda-corrdiff/{corrdiff_eda.png, summary.json}
"""
import os, sys, json, time
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from era5_daymet.tools.plotting import plot_model_maps as PM
from era5_daymet.training import train_downscale as TD

W = "/lustre/orion/atm112/scratch/hjsong/downscaling"
OUTDIR = f"{W}/runs/exp/20260720-eda-corrdiff"
# DAYS 覆盖全年(汇报口径, stride=1)。★注意: 每天做 NMEM 成员 corrdiff 集合采样, 全年非常贵;
# 快速探索可临时改回少数代表日(如 [9,100,190,280] 覆盖四季), 但不可作为全年汇报。
YEAR = 2020; DAYS = list(range(0, 365)); NMEM = 12; BOX = 384
PRECIP = TD.PRECIP
os.makedirs(OUTDIR, exist_ok=True)
logmm = lambda x: np.log1p(np.maximum(x, 0.0) * 1000.0)


def radial_mean(P):
    c0, c1 = np.array(P.shape) // 2; Y, X = np.indices(P.shape)
    r = np.hypot(Y - c0, X - c1).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def psd(img):
    f = img - img.mean(); w = np.hanning(f.shape[0])[:, None] * np.hanning(f.shape[1])[None, :]
    return radial_mean(np.abs(np.fft.fftshift(np.fft.fft2(f * w))) ** 2)


def main():
    t0 = time.time()
    args = SimpleNamespace(
        year=YEAR, stats_dir=f"{W}/runs/stats/train_dayofyear",
        era5_dir=None, daymet_dir=None,
        unet_dir=f"{W}/runs/exp/20260711-unet-b64",
        vit_dir=f"{W}/runs/exp/20260712-vit-d384-b16-ep12",
        bcsd_coef_dir=f"{W}/runs/bcsd_coefs",
        corrdiff_dir=f"{W}/runs/exp/20260714-corrdiff-b64",
        regressor_ckpt=f"{W}/runs/exp/20260711-unet-b64/ckpt.pt")
    from era5_daymet.data import match_era5_daymet as M
    args.era5_dir = M.ERA5_DIR; args.daymet_dir = M.DAYMET_DIR
    device, stats, test, out_vars, det_preds, corrdiff_pred = PM.build(args)
    ti = {v: i for i, v in enumerate(out_vars)}
    imax, imin, ip = ti["2m_temperature_max"], ti["2m_temperature_min"], ti[PRECIP]

    # 变量 -> (index, 变换, 标签)
    specs = [("tmax", imax, lambda a: a, "tmax (K)"),
             ("precip", ip, logmm, "precip log1p(mm)")]
    acc = {name: {"truth": None, "unet": None, "vit": None, "cd_mean": None, "cd_mem": None} for name, *_ in specs}
    ranks_all = []           # tmax truth 在成员中的秩
    spread_skill = []        # (集合发散度, 均值RMSE) per day, tmax

    for t in DAYS:
        cond, _, m, hr = test.full(YEAR, t)
        land = (m[0] if m.ndim == 3 else m) > 0.5
        by, bx, bs = TD.pick_land_box(land, min(BOX, test.H, test.W))
        sl = (slice(by, by+bs), slice(bx, bx+bs))
        cond_b = cond[:, by:by+bs, bx:bx+bs]
        m_b = m[:, by:by+bs, bx:bx+bs] if m.ndim == 3 else (m[by:by+bs, bx:bx+bs][None])
        unet = det_preds["unet"](cond_b, t)[0]              # (Cout,bs,bs) 物理
        vit = det_preds["vit"](cond_b, t)[0]
        mem = corrdiff_pred(cond_b, m_b, NMEM)              # (N,Cout,bs,bs) 物理
        cd_mean = mem.mean(0)
        truth = hr[:, by:by+bs, bx:bx+bs]

        for name, idx, tf, _ in specs:
            fields = dict(truth=tf(truth[idx]), unet=tf(unet[idx]), vit=tf(vit[idx]),
                          cd_mean=tf(cd_mean[idx]))
            fields["cd_mem"] = np.mean([psd(tf(mem[k, idx])) for k in range(NMEM)], 0)  # 成员PSD均值
            for key in ["truth", "unet", "vit", "cd_mean"]:
                p = psd(fields[key]); acc[name][key] = p if acc[name][key] is None else acc[name][key] + p
            acc[name]["cd_mem"] = fields["cd_mem"] if acc[name]["cd_mem"] is None else acc[name]["cd_mem"] + fields["cd_mem"]

        # 校准 (tmax, K, 陆地像素)
        lb = land[sl]
        mt = mem[:, imax][:, lb]                            # (N, Npix)
        yt = truth[imax][lb]                                # (Npix,)
        rk = (mt < yt[None]).sum(0) + (np.random.rand(yt.size) * ((mt == yt[None]).sum(0))).astype(int)
        ranks_all.append(rk)
        spread = float(mem[:, imax].std(0)[lb].mean())      # 集合发散度(逐像素std的均值)
        skill = float(np.sqrt(((cd_mean[imax] - truth[imax])[lb] ** 2).mean()))  # 均值RMSE
        spread_skill.append((spread, skill))
        print(f"[cd-eda] day {t} done ({time.time()-t0:.0f}s)  spread={spread:.3f} skill={skill:.3f}", flush=True)

    nd = len(DAYS)
    for name, *_ in specs:
        for k in acc[name]:
            acc[name][k] = acc[name][k] / nd

    # ---- 图 ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    colors = dict(truth="k", unet="#1f77b4", vit="#ff7f0e", cd_mean="#2ca02c", cd_mem="#d62728")
    labels = dict(truth="Truth", unet="UNet (det)", vit="ViT (det)",
                  cd_mean="CorrDiff ens-mean", cd_mem="CorrDiff member")
    for j, (name, idx, tf, lab) in enumerate(specs):
        a = ax[0, j]; P = acc[name]; k = np.arange(1, len(P["truth"]))
        for key in ["truth", "unet", "vit", "cd_mean", "cd_mem"]:
            a.loglog(k, P[key][1:], color=colors[key], lw=2.2 if key in ("truth", "cd_mem") else 1.6,
                     ls="-" if key != "cd_mean" else "--", label=labels[key])
        a.axvline(BOX/60, color="green", ls=":", alpha=.6, label="60px tile scale")
        a.set_title(f"power spectrum  {lab}"); a.set_xlabel("radial wavenumber k (cyc/box)")
        a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)

    # rank histogram tmax
    ranks = np.concatenate(ranks_all); nb = NMEM + 1
    h, _ = np.histogram(ranks, bins=np.arange(nb + 1)); h = h / h.sum()
    a = ax[1, 0]; a.bar(range(nb), h, color="#8888cc"); a.axhline(1/nb, color="r", ls="--", label="flat = calibrated")
    a.set_title(f"rank histogram tmax (N={NMEM} members)\nU-shape=under-dispersed, dome=over-dispersed")
    a.set_xlabel("rank of truth among members"); a.set_ylabel("freq"); a.legend(fontsize=8)

    # spread-skill
    a = ax[1, 1]; sp = np.array(spread_skill)
    a.scatter(sp[:, 0], sp[:, 1], s=60, c="#d62728", zorder=3)
    lim = [0, max(sp.max()*1.15, 0.1)]
    a.plot(lim, lim, "k--", alpha=.6, label="1:1 (calibrated)")
    for (x, y), t in zip(sp, DAYS): a.annotate(f"d{t}", (x, y), fontsize=8)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("ensemble spread (tmax std, K)")
    a.set_ylabel("ens-mean RMSE (tmax, K)"); a.set_title("spread-skill (tmax)\nspread<skill => under-dispersed")
    a.grid(True, alpha=.3); a.legend(fontsize=8)

    fig.suptitle(f"EDA-4  CorrDiff characteristics on ERA5->Daymet   year={YEAR}  days={DAYS}  N={NMEM}  box={BOX}px\n"
                 f"top: member restores high-freq (sharp) while ens-mean/det roll off;  bottom: ensemble calibration",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "corrdiff_eda.png"), dpi=130, bbox_inches="tight")

    # 高频功率比 (k>tile尺度) member/mean/det 相对 truth
    hf = {}
    for name, *_ in specs:
        P = acc[name]; kt = int(round(BOX/60))
        tr_hf = P["truth"][kt:].sum()
        hf[name] = {key: float(P[key][kt:].sum()/max(tr_hf, 1e-12)) for key in ["unet", "vit", "cd_mean", "cd_mem"]}
    json.dump(dict(days=DAYS, nmem=NMEM, box=BOX, highfreq_power_ratio_vs_truth=hf,
                   spread_skill=[dict(day=t, spread=s, skill=k) for t, (s, k) in zip(DAYS, spread_skill)],
                   rank_hist_tmax=h.tolist()), open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1)
    for name in hf:
        print(f"[cd-eda] {name} high-freq power / truth:  UNet {hf[name]['unet']:.2f}  ViT {hf[name]['vit']:.2f}  "
              f"CD-mean {hf[name]['cd_mean']:.2f}  CD-member {hf[name]['cd_mem']:.2f}", flush=True)
    print(f"[cd-eda] DONE {time.time()-t0:.0f}s -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
