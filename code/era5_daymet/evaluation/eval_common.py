#!/usr/bin/env python
# Packaged implementation; code/eval_common.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
eval_common.py — 所有方法共用的评测/画图(单一真相, 保证同表同图)
============================================================================
谁产生预测都行(插值 / UNet / ViT / SCD-EDM 集合), 都喂给同一个 MultiMethodEval,
于是 RMSE/MAE/bias/corr/CRPS、年平均地图、功率谱、rank histogram 全部口径一致、
画在同一张图里。纯 numpy(不依赖 torch); 调用方负责把预测转成物理量。

约定:
  * 预测以"集合成员"形式给入: members 形状 (N, Cout, H, W), 物理单位。
    确定性方法 N=1(此时 CRPS ≡ MAE, 自动成立)。
  * 真值 hr: (Cout, H, W) 物理量; mask: (H,W) 或 (1,H,W) 陆地布尔。
  * 确定性指标(RMSE/MAE/bias/corr)算在"集合均值"上; CRPS/rank-hist 用整个集合。
============================================================================
"""
import os
import json

import numpy as np

from era5_daymet.data import downscale_baseline as DB
from era5_daymet.training import train_downscale as TD

from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity as _ssim

PRECIP = TD.PRECIP

# --- SSIM 的三个口径, 定死在这里, 免得各处不一致 ---
SSIM_SIGMA = 1.5          # Wang et al. 原始 SSIM: 11x11 高斯窗, sigma=1.5
SSIM_ERODE = 5            # 陆地掩膜腐蚀半径(px): 海岸窗口会吃到填充值, 剔掉


def ssim_masked(pred, truth, mb, mb_er):
    """陆地上的 SSIM。

    ★三个必须交代的口径决定(SSIM 对这些极其敏感, 不写清楚数字就没法复现):

    1) 海洋怎么办: SSIM 是局部窗口算的, 窗口一旦跨过海岸线就会吃到无效值。
       这里把 pred 和 truth 的非陆地区域填成★同一个值★(当日陆地均值) ——
       两边填一样 -> 不引入任何人为的结构差异; 再把 SSIM 图只在★腐蚀过★的陆地
       掩膜上平均(腐蚀半径 5px > 高斯窗半径), 彻底避开被填充值污染的海岸窗口。
       (若填不同值, 或不腐蚀直接在全陆地上平均, 海岸带会凭空产生结构差, 分数失真。)

    2) data_range: SSIM 的 C1/C2 正比于动态范围 L。这里取★当日 truth 在陆地上的
       max-min★。同一天里所有方法共用同一个 L -> ★方法之间的比较是公平的★
       (这正是我们要的); 跨天的 L 不同, 但最后是对天平均, 不影响方法排序。

    3) 公式: 高斯窗(sigma=1.5) + use_sample_covariance=False, 即 Wang 2004 原版,
       不是 skimage 的 7x7 均匀窗默认值。
    """
    fill = float(truth[mb].mean())
    p = np.where(mb, pred, fill).astype(np.float64)
    t = np.where(mb, truth, fill).astype(np.float64)
    dr = float(truth[mb].max() - truth[mb].min())
    if dr <= 0:
        return 1.0                                     # 该日陆地上是常数场(退化), 定义为完全相似
    _, S = _ssim(t, p, data_range=dr, gaussian_weights=True, sigma=SSIM_SIGMA,
                 use_sample_covariance=False, full=True)
    m = mb_er if mb_er.any() else mb
    return float(S[m].mean())


class MultiMethodEval:
    def __init__(self, methods, out_vars, H, W, landmask, precip_scale=1000.0, precip_log=True):
        self.methods = list(methods)
        self.ov = list(out_vars); self.Cout = len(out_vars)
        self.H, self.W = H, W
        self.landmask = landmask                                  # (H,W) bool
        self.pscale = precip_scale; self.plog = precip_log
        self.acc = {m: {v: DB.Acc() for v in out_vars} for m in methods}
        self.crps = {m: {v: [0.0, 0] for v in out_vars} for m in methods}   # [sum, n]
        self.ssim = {m: {v: [0.0, 0] for v in out_vars} for m in methods}   # [sum, n]
        # spread-skill: 集合离散度校准。逐日在陆地上取成员方差(ddof=1)均值累加,
        # finalize 里 spread=sqrt(均值方差), 与 RMSE(集合均值) 比。见 finalize 的推导。
        self.spread = {m: {v: [0.0, 0] for v in out_vars} for m in methods}  # [sum_daily_meanvar, n]
        # 降水额外算一份 log 空间 SSIM: 物理空间里绝大多数格点是 0, 局部方差为 0 -> SSIM 虚高,
        # 分数被"共同的干区"主导而不是被降水结构主导。log1p 压缩后结构信息才看得见。
        self.ssim_log = {m: [0.0, 0] for m in methods}
        # 降水的 RMSE/MAE/bias/corr 同样另算一份 log1p(mm) 空间的: 物理空间里量级由少数强降水
        # 格点主导, 两个空间下方法的排名可以是相反的, 因此两份都要给出。
        self.acc_log = {m: DB.Acc() for m in methods}
        self.msum = {m: np.zeros((self.Cout, H, W), np.float64) for m in methods}
        self.tsum = np.zeros((self.Cout, H, W), np.float64); self.nd = 0
        self.spec = {}                                            # 首日: spec['truth'][v], spec[m][v]={'mean','member'}
        self.rh = {m: [] for m in methods}                       # rank hist(仅集合方法)
        self.N = {m: 1 for m in methods}
        self.box = None
        self.mb_er = None                                         # 腐蚀过的陆地掩膜(SSIM 用)

    def add_day(self, hr, mask, members_by_method):
        mb = (mask[0] if mask.ndim == 3 else mask) > 0.5
        if self.box is None:
            self.box = TD.pick_land_box(mb, min(384, self.H, self.W))
            # ★功率谱改为★全年累加求平均★(不再只取首日快照); 首日把累加器初始化为 0
            self.spec["truth"] = {v: 0.0 for v in self.ov}
        if self.mb_er is None:
            self.mb_er = binary_erosion(mb, iterations=SSIM_ERODE)
        by, bx, bs = self.box; first = (self.nd == 0)
        self.tsum += hr; self.nd += 1
        for i, v in enumerate(self.ov):                          # 累加 truth 功率谱(全年平均, _plots 里除以 nd)
            self.spec["truth"][v] = self.spec["truth"][v] + TD.radial_psd(hr[i][by:by+bs, bx:bx+bs], bs)
        pi = self.ov.index(PRECIP) if PRECIP in self.ov else None
        lg = lambda a: np.log1p(np.maximum(a, 0) * self.pscale)   # 物理量 -> log1p(mm)
        for m in self.methods:
            mem = np.asarray(members_by_method[m])               # (N,Cout,H,W)
            self.N[m] = mem.shape[0]
            ens = mem.mean(0)
            self.msum[m] += ens
            for i, v in enumerate(self.ov):
                self.acc[m][v].add(ens[i][mb], hr[i][mb])
                c = TD.crps_ensemble(mem[:, i:i+1], hr[i:i+1], (mask[0] if mask.ndim == 3 else mask))[0]
                self.crps[m][v][0] += float(c); self.crps[m][v][1] += 1
                s = ssim_masked(ens[i], hr[i], mb, self.mb_er)   # ★SSIM 算在集合均值上(与 RMSE 同口径)
                self.ssim[m][v][0] += s; self.ssim[m][v][1] += 1
                if mem.shape[0] > 1:                              # spread(集合成员方差, ddof=1, 陆地均值)
                    dv = float(mem[:, i].var(axis=0, ddof=1)[mb].mean())
                    self.spread[m][v][0] += dv; self.spread[m][v][1] += 1
            if pi is not None:                                    # 降水: 再算一份 log 空间的 SSIM 与逐点指标
                s = ssim_masked(lg(ens[pi]), lg(hr[pi]), mb, self.mb_er)
                self.ssim_log[m][0] += s; self.ssim_log[m][1] += 1
                self.acc_log[m].add(lg(ens[pi])[mb], lg(hr[pi])[mb])
            if mem.shape[0] > 1:
                self.rh[m].append(TD.rank_hist(mem, hr, (mask[0] if mask.ndim == 3 else mask)))
            if m not in self.spec:                               # 首次见到该方法 -> 初始化功率谱累加器
                self.spec[m] = {v: {"mean": 0.0, "member": 0.0} for v in self.ov}
            for i, v in enumerate(self.ov):                      # 累加方法功率谱(全年平均)
                self.spec[m][v]["mean"] = self.spec[m][v]["mean"] + TD.radial_psd(ens[i][by:by+bs, bx:bx+bs], bs)
                self.spec[m][v]["member"] = self.spec[m][v]["member"] + TD.radial_psd(mem[0, i][by:by+bs, bx:bx+bs], bs)

    # -------------------------------------------------------------------
    def finalize(self, out_dir, test_year, eval_stride=1, tag="all", n_total_days=None,
                 make_plots=True, make_maps=True):
        os.makedirs(out_dir, exist_ok=True)
        res = {}
        for m in self.methods:
            res[m] = {}
            for v in self.ov:
                r = self.acc[m][v].result()
                cs, cn = self.crps[m][v]
                r["crps"] = round(cs / max(cn, 1), 6)
                ss, sn = self.ssim[m][v]
                r["ssim"] = round(ss / max(sn, 1), 4)
                # spread-skill 校准比(仅集合方法): 见下方推导, 目标=1, <1 欠离散/>1 过散
                sp_s, sp_n = self.spread[m][v]
                M = self.N[m]
                if sp_n > 0 and M > 1 and r["rmse"] > 0:
                    spread = (sp_s / sp_n) ** 0.5
                    r["spread"] = round(spread, 4)
                    r["spread_skill_ratio"] = round(spread * ((M + 1) / M) ** 0.5 / r["rmse"], 4)
                if v == PRECIP:
                    ls, ln = self.ssim_log[m]
                    r["ssim_log"] = round(ls / max(ln, 1), 4)
                    for k, x in self.acc_log[m].result().items():
                        r[f"{k}_log"] = x
                res[m][v] = r
            res[m]["_ensemble"] = self.N[m]
        res["_space"] = ("physical (precip native units); precip additionally reported in "
                         "log1p(mm) space with the _log suffix; land-masked")
        res["_note"] = "RMSE/MAE/bias/corr/SSIM on ensemble-mean; CRPS/rank-hist/spread on full ensemble (det CRPS==MAE)"
        res["_spread_skill_spec"] = (
            "spread = sqrt(mean over days of [land-mean of member variance, ddof=1]); "
            "spread_skill_ratio = spread * sqrt((M+1)/M) / RMSE(ensemble-mean), M=n_members. "
            "Calibrated target = 1 (the sqrt((M+1)/M) factor removes the finite-ensemble bias: "
            "for iid members+truth, E[MSE_mean]=sigma^2 (M+1)/M while E[var,ddof=1]=sigma^2). "
            "ratio<1 = under-dispersed (ensemble too confident); >1 = over-dispersed.")
        res["_ssim_spec"] = (f"Wang2004: gaussian window sigma={SSIM_SIGMA}, use_sample_covariance=False; "
                             f"data_range = per-day (max-min) of truth over land (same L for all methods on a day); "
                             f"ocean filled with that day's land-mean in BOTH pred and truth, "
                             f"SSIM map averaged over land mask eroded by {SSIM_ERODE}px to avoid coastline windows. "
                             f"precip also reported as ssim_log = SSIM in log1p(mm) space "
                             f"(in physical space most pixels are 0 -> zero local variance -> SSIM inflated by the shared dry area).")
        res["_n_test_days"] = self.nd; res["_eval_stride"] = eval_stride; res["_methods"] = list(self.methods)
        # ── 覆盖率护栏 ─────────────────────────────────────────────────────────
        # 子采样评测不是完整 test 集得分, 文件名与提示都必须标出覆盖天数, 避免被当成最终结果。
        # 规则: 只有覆盖完整 test 年(stride==1 且天数==全年)才配叫规范文件名 metrics_{tag}.json;
        # 任何子采样都写成带 __SUBSAMPLED 的另一个文件、盖上 _WARNING、并打醒目横幅,
        # 让它无法冒充"完整 test 集得分"。下游(报告/plot_table)只认规范文件名。
        full = (eval_stride == 1) and (n_total_days is None or self.nd >= n_total_days)
        cov = f"{self.nd}/{n_total_days}" if n_total_days else f"{self.nd} days"
        res["_n_total_test_days"] = n_total_days
        if full:
            res["_coverage"] = f"FULL test set ({cov} days, stride 1)"
            fname = f"metrics_{tag}.json"
        else:
            res["_coverage"] = f"SUBSAMPLED ({cov} test days, stride {eval_stride})"
            res["_WARNING"] = (
                f"⚠ 子采样评测: 仅覆盖 {cov} 个 test 天 (stride={eval_stride})。"
                f"这不是完整 test 集得分, 严禁作为最终结果汇报。"
                f"完整评测请用 eval_stride=1 重跑。")
            fname = f"metrics_{tag}__SUBSAMPLED_stride{eval_stride}.json"
        jpath = os.path.join(out_dir, fname)
        json.dump(res, open(jpath, "w"), indent=2, ensure_ascii=False, default=float)

        print("\n" + "=" * 118)
        print(f"{'variable':26s} {'method':9s} {'N':>3s} {'RMSE':>10s} {'MAE':>10s} {'bias':>10s} "
              f"{'corr':>8s} {'CRPS':>10s} {'SSIM':>8s} {'SSIMlog':>8s}")
        print("-" * 118)
        for v in self.ov:
            for m in self.methods:
                r = res[m][v]
                sl = f"{r['ssim_log']:8.4f}" if "ssim_log" in r else " " * 8
                print(f"{v:26s} {m:9s} {self.N[m]:>3d} {r['rmse']:10.4f} {r['mae']:10.4f} "
                      f"{r['bias']:10.4f} {r['corr']:8.4f} {r['crps']:10.4f} {r['ssim']:8.4f} {sl}")
            print("-" * 118)
        # spread-skill 单列出来(仅集合方法): ratio<1 欠离散
        ens_m = [m for m in self.methods if self.N[m] > 1]
        if ens_m:
            print(f"\n{'spread-skill (target=1, <1 under-dispersed)':^60s}")
            print(f"{'variable':26s} {'method':9s} {'spread':>10s} {'RMSE':>10s} {'ratio':>8s}")
            for v in self.ov:
                for m in ens_m:
                    r = res[m][v]
                    if "spread_skill_ratio" in r:
                        print(f"{v:26s} {m:9s} {r['spread']:10.4f} {r['rmse']:10.4f} {r['spread_skill_ratio']:8.4f}")
        print(f"指标 JSON -> {jpath}  ({self.nd} 天, stride={eval_stride})")
        if full:
            print(f"✅ 完整 test 集覆盖 ({cov} 天) — 可作为最终结果汇报。")
        else:
            bar = "!" * 78
            print("\n" + bar)
            print(f"⚠ 子采样评测 — 仅 {cov} 个 test 天 (stride={eval_stride})。")
            print("  这不是完整 test 集得分, 严禁当最终结果汇报。文件名已标 __SUBSAMPLED。")
            print("  完整评测: --eval-stride 1")
            print(bar + "\n")

        if make_plots:
            self._plots(out_dir, test_year, tag, make_maps=make_maps)
        return res

    def dump_sums(self, path):
        """把★可加★的累计量落盘, 供多 shard 精确合并(见 merge_sums_and_finalize)。
        只含表格指标(RMSE/MAE/bias/corr/CRPS/SSIM/spread)所需的和量, 不含谱/年均图
        (那些 CorrDiff 另有专门脚本)。DB.Acc 与各 [sum,n] 都是逐日线性累加, 分片求和精确。"""
        import pickle
        st = {"methods": list(self.methods), "ov": list(self.ov),
              "N": dict(self.N), "nd": self.nd,
              "acc": {m: {v: vars(self.acc[m][v]).copy() for v in self.ov} for m in self.methods},
              "crps": {m: {v: list(self.crps[m][v]) for v in self.ov} for m in self.methods},
              "ssim": {m: {v: list(self.ssim[m][v]) for v in self.ov} for m in self.methods},
              "spread": {m: {v: list(self.spread[m][v]) for v in self.ov} for m in self.methods},
              "ssim_log": {m: list(self.ssim_log[m]) for m in self.methods},
              "acc_log": {m: vars(self.acc_log[m]).copy() for m in self.methods}}
        pickle.dump(st, open(path, "wb"))
        print(f"[shard] 累计量 -> {path} ({self.nd} 天, {list(self.methods)})", flush=True)

    # -------------------------------------------------------------------
    def _plots(self, out_dir, test_year, tag, make_maps=True):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"(无 matplotlib, 跳过画图: {e})"); return
        cmap_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        by, bx, bs = self.box; k = np.arange(1, bs // 2)
        nd = max(self.nd, 1)                                       # 功率谱是全年累加 -> 除以天数得平均

        # (a) 逐变量功率谱(全年平均): truth(黑) + 各方法均值(实线); 集合方法额外画单成员(虚线)
        fig, axes = plt.subplots(1, self.Cout, figsize=(5.2 * self.Cout, 4.3), squeeze=False)
        for i, v in enumerate(self.ov):
            ax = axes[0, i]
            ax.loglog(k, (self.spec["truth"][v] / nd)[1:bs // 2], "k-", lw=2.4, label="truth")
            for j, m in enumerate(self.methods):
                col = cmap_cycle[j % len(cmap_cycle)]
                ax.loglog(k, (self.spec[m][v]["mean"] / nd)[1:bs // 2], "-", color=col, label=f"{m} (mean)")
                if self.N[m] > 1:
                    ax.loglog(k, (self.spec[m][v]["member"] / nd)[1:bs // 2], "--", color=col, alpha=.7, label=f"{m} (member)")
            ax.set_xlabel("radial wavenumber"); ax.set_ylabel("power"); ax.set_title(v)
            ax.grid(True, which="both", alpha=.3)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("RAPSD (closer to truth at high-k = sharper; member should match truth, mean is smoother)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"spectrum_{tag}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

        # (b) 年平均地图: 上排 truth+各方法均值场, 下排 各方法-truth 偏差
        if not make_maps:
            return
        tmean = self.tsum / max(self.nd, 1)
        mmean = {m: self.msum[m] / max(self.nd, 1) for m in self.methods}
        ext = [DB.LON_EDGES[0], DB.LON_EDGES[1], DB.LAT_EDGES[0], DB.LAT_EDGES[1]]
        for i, v in enumerate(self.ov):
            sc = self.pscale if v == PRECIP else 1.0
            unit = "mm/day" if v == PRECIP else "deg"

            def msk(a):
                o = (a * sc).astype(np.float32).copy(); o[~self.landmask] = np.nan; return o
            truth_m = msk(tmean[i])
            vmin = float(np.nanpercentile(truth_m, 2)); vmax = float(np.nanpercentile(truth_m, 98))
            diffs = [msk(mmean[m][i]) - truth_m for m in self.methods]
            dmax = float(np.nanpercentile(np.abs(np.stack(diffs)), 98)) or 1.0
            ncol = 1 + len(self.methods)
            fig, ax = plt.subplots(2, ncol, figsize=(3.3 * ncol, 6.6), squeeze=False)
            for c, (title, img) in enumerate([("truth", truth_m)] + [(m, msk(mmean[m][i])) for m in self.methods]):
                a = ax[0, c]; im = a.imshow(img, origin="lower", extent=ext, cmap="viridis",
                                            vmin=vmin, vmax=vmax, aspect="auto")
                a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            ax[1, 0].axis("off")
            for c, m in enumerate(self.methods):
                a = ax[1, c + 1]; im = a.imshow(diffs[c], origin="lower", extent=ext, cmap="RdBu_r",
                                                vmin=-dmax, vmax=dmax, aspect="auto")
                a.set_title(f"{m} - truth", fontsize=10); a.set_xticks([]); a.set_yticks([])
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            fig.suptitle(f"{v}  annual-mean ({unit})  test {test_year}   [top: fields | bottom: bias vs truth]", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(out_dir, f"maps_{tag}_{v}.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

        # (c) rank histogram(仅集合方法; 平=校准好)
        ens_methods = [m for m in self.methods if self.N[m] > 1 and self.rh[m]]
        if ens_methods:
            fig, axes = plt.subplots(1, len(ens_methods), figsize=(4.2 * len(ens_methods), 3.4), squeeze=False)
            for j, m in enumerate(ens_methods):
                h = np.mean(self.rh[m], 0); n = len(h); a = axes[0, j]
                a.bar(range(n), h); a.axhline(1.0 / n, color="r", ls="--")
                a.set_title(f"rank hist {m} (tmax, flat=calibrated)", fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"rankhist_{tag}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
        print(f"图 -> {out_dir}/spectrum_{tag}.png, maps_{tag}_*.png" + (f", rankhist_{tag}.png" if ens_methods else ""))


def merge_sums_and_finalize(sum_paths, out_dir, test_year, tag="corrdiff",
                            n_total_days=None):
    """把多个 shard 的 dump_sums 精确合并, 复用 finalize 的公式产出规范 metrics_{tag}.json。
    只算表格指标(不出谱/年均图 -> make_plots=False)。分片必须覆盖★不相交★的 test 天。"""
    import pickle
    states = [pickle.load(open(p, "rb")) for p in sum_paths]
    if not states:
        raise ValueError("没有 shard 累计文件可合并")
    base = states[0]
    ev = MultiMethodEval(base["methods"], base["ov"], 2, 2, np.ones((2, 2), bool))
    ev.nd = 0
    for m in ev.methods:
        ev.N[m] = 1
        for v in ev.ov:
            ev.acc[m][v] = DB.Acc()
            ev.crps[m][v] = [0.0, 0]; ev.ssim[m][v] = [0.0, 0]; ev.spread[m][v] = [0.0, 0]
        ev.ssim_log[m] = [0.0, 0]
        ev.acc_log[m] = DB.Acc()
    ACC_FIELDS = ("n", "se", "ae", "be", "sp", "st", "spp", "stt", "spt")
    for st in states:
        ev.nd += st["nd"]
        for m in ev.methods:
            ev.N[m] = max(ev.N[m], st["N"][m])
            for v in ev.ov:
                a = ev.acc[m][v]; sa = st["acc"][m][v]
                for k in ACC_FIELDS:
                    setattr(a, k, getattr(a, k) + sa[k])
                for fld, key in ((ev.crps, "crps"), (ev.ssim, "ssim"), (ev.spread, "spread")):
                    fld[m][v][0] += st[key][m][v][0]; fld[m][v][1] += st[key][m][v][1]
            ev.ssim_log[m][0] += st["ssim_log"][m][0]; ev.ssim_log[m][1] += st["ssim_log"][m][1]
            if "acc_log" in st:
                al = ev.acc_log[m]
                for k in ACC_FIELDS:
                    setattr(al, k, getattr(al, k) + st["acc_log"][m][k])
    print(f"[merge] {len(states)} 个 shard -> 合计 {ev.nd} 天", flush=True)
    return ev.finalize(out_dir, test_year, eval_stride=1, tag=tag,
                       n_total_days=n_total_days, make_plots=False)
