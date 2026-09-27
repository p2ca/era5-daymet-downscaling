#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
residual_routing_precheck.py — 残差驱动的专家容量分配: 训练前核查
============================================================================
问题: 若让 MoE 的专家容量跟着"网络自己的残差幅度"走, 这套分配在阶段 A 有没有可分的东西、
能不能从输入预测出来、跨时间稳不稳定。四项核查全部只读已有落场与 μ 缓存, 纯 CPU:

  concentration   阶段 A 的平方误差在 16×16 token 上的集中度(静态年图 与 逐 token-日),
                  以及 token-日误差方差里"静态 token 效应 / 逐日效应"各占多少
  predictability  逐 token 误差幅度能否从 51 条件通道(token 均值)+ token 内地形粗糙度预测:
                  岭回归与 MLP 探针, 时间划分(奇偶月)与空间划分(棋盘 token)各一套
  persistence     μ 缓存残差(归一化空间)的逐像素 σ_r 图: 动态范围、跨年与跨月的秩相关
  reducible       同一方法 real/oracle 的逐 token 误差之差(可约部分)的分布, 以及按"总误差"
                  排序能否找到可约误差

单位口径: 落场 ens_mean 与 Daymet 真值同为物理单位(温度 K); μ 缓存与 DownscaleData.target
的归一化目标同在归一化空间。两条路径各自闭合, 不混用。有效域一律用 DownscaleData.mask。

用法(从 code/ 目录):
  python -m downscaling_4x.tools.diagnostics.residual_routing_precheck concentration \\
      --exp runs/exp/<jda1> --out runs/exp/<diag>/concentration
  ... predictability --exp <jda1> --out <dir>
  ... persistence --mu-cache runs/mu_cache/<id> --out <dir>
  ... reducible --pair unet=<real_dir>:<oracle_dir> --pair ca=<real>:<oracle> --out <dir>
============================================================================
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.data.mu_cache import MuCache


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def pool_sum(a, P):
    """(H, W) -> (H/P, W/P), 每个 P×P 块求和。"""
    H, W = a.shape
    if H % P or W % P:
        raise ValueError(f"{(H, W)} 不能被 patch {P} 整除")
    return a.reshape(H // P, P, W // P, P).sum(axis=(1, 3))


def pool_mean_std(a, P, cnt=None):
    """块内均值与标准差; cnt 给定时只在掩膜内统计(a 在掩膜外须已置 0)。"""
    n = pool_sum(np.ones_like(a, np.float64), P) if cnt is None else cnt
    s1 = pool_sum(a.astype(np.float64), P)
    s2 = pool_sum(np.square(a, dtype=np.float64), P)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        v = np.maximum(s2 / n - m * m, 0.0)
    return m, np.sqrt(v)


def lorenz(v):
    """非负量的集中度: 前 5/10/20/30/50% 单元占总量的份额, 以及 Gini 系数。"""
    v = np.sort(np.asarray(v, np.float64))[::-1]
    tot = v.sum()
    if tot <= 0:
        return {"error": "total is zero"}
    c = np.cumsum(v) / tot
    n = len(v)

    def share(p):
        return round(float(c[max(int(np.ceil(p * n)) - 1, 0)]), 4)

    asc = v[::-1]
    i = np.arange(1, n + 1)
    gini = float((2 * np.sum(i * asc)) / (n * asc.sum()) - (n + 1) / n)
    return {"n": int(n), "top5": share(.05), "top10": share(.10), "top20": share(.20),
            "top30": share(.30), "top50": share(.50), "gini": round(gini, 4)}


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, np.float64), np.asarray(y, np.float64)).correlation
    return round(float(r), 4)


def pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def month_of(year, day):
    return M.frame_date(year, day).month


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def save_map(arr, path, title, vmin=None, vmax=None, cmap="viridis", label=""):
    """单幅地图; 掩膜外为 NaN 显示为空白。色标范围写回给调用方以便同组共用。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.asarray(arr, np.float64)
    if vmin is None:
        vmin = float(np.nanpercentile(a, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(a, 99))
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    im = ax.imshow(a, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest", origin="lower")  # 第 0 行在南
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if label:
        cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"vmin": vmin, "vmax": vmax, "cmap": cmap}


def save_curve(xs, ys, path, title, xlabel, ylabel, ref_diag=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    for lab, (x, y) in ys.items():
        ax.plot(x, y, label=lab, lw=1.6)
    if ref_diag:
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="uniform")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def lorenz_curve(v, npts=200):
    v = np.sort(np.asarray(v, np.float64))[::-1]
    c = np.cumsum(v) / v.sum()
    n = len(v)
    xs = np.linspace(0, 1, npts)
    idx = np.clip((xs * n).astype(int) - 1, 0, n - 1)
    ys = np.where(xs > 0, c[idx], 0.0)
    return xs, ys


def load_regions(path):
    if not path or not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {"id": z["region_id"].astype(np.int64), "names": [str(s) for s in z["region_names"]]}


def region_shares(se_pix, land, reg):
    """逐区域的平方误差份额与均方误差。区域编号 1..n, 其余为域外。"""
    if reg is None:
        return None
    rid = reg["id"]
    out = {}
    tot = se_pix[land].sum()
    for i, name in enumerate(reg["names"], start=1):
        m = land & (rid == i)
        if not m.any():
            continue
        out[name] = {"pixels": int(m.sum()), "se_share": round(float(se_pix[m].sum() / tot), 4),
                     "mse_rel": round(float(se_pix[m].mean() / se_pix[land].mean()), 3)}
    return out


def field_error_accumulate(fields_dir, dd, ti, year, P, land, limit=None):
    """逐日读落场 ens_mean, 与真值(物理单位)作差, 掩膜外置 0。
    返回逐像素 SE/AE 累计与逐 (日, token) 的 SE/AE 累计, 以及自检信息。"""
    H, W = land.shape
    nd = min(dd.ndays[year], C.DAYS_PER_YEAR)
    if limit:
        nd = min(nd, limit)
    se_pix = np.zeros((H, W), np.float64)
    ae_pix = np.zeros((H, W), np.float64)
    tok_se = np.zeros((nd, H // P, W // P), np.float64)
    tok_ae = np.zeros((nd, H // P, W // P), np.float64)
    nan_outside = None
    for t in range(nd):
        pred = np.load(Path(fields_dir) / f"{year}_d{t}.npy").astype(np.float64)
        if pred.shape != (H, W):
            raise ValueError(f"落场形状 {pred.shape} 与掩膜 {(H, W)} 不符")
        fin = np.isfinite(pred)
        if not fin[land].all():
            raise ValueError(f"{year} 第 {t} 天有效域内含非有限值")
        if nan_outside is None:
            nan_outside = round(float((~fin[~land]).mean()), 4)
        truth = dd.target(year, t)[1][ti].astype(np.float64)
        e = np.where(land, pred - truth, 0.0)
        se = e * e
        ae = np.abs(e)
        se_pix += se
        ae_pix += ae
        tok_se[t] = pool_sum(se, P)
        tok_ae[t] = pool_sum(ae, P)
    return se_pix, ae_pix, tok_se, tok_ae, nd, nan_outside


def setup_truth(a, years, mode="baseline_21", cache_years=1):
    stats = Stats(a.era5_dir, a.daymet_dir)
    dd = DownscaleData(a.era5_dir, a.daymet_dir, list(years), stats, mode=mode,
                       era5_cache_years=cache_years)
    return stats, dd


# ---------------------------------------------------------------------------
# 1. 误差集中度
# ---------------------------------------------------------------------------
def run_concentration(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats, dd = setup_truth(a, [a.year])
    land = dd.mask
    ti = C.TARGETS.index(a.target)
    P = a.patch
    fields = Path(a.exp) / f"fields{a.year}" / a.target / "ens_mean"
    t0 = time.time()
    se_pix, ae_pix, tok_se, tok_ae, nd, nan_outside = field_error_accumulate(
        fields, dd, ti, a.year, P, land, a.limit_days)
    print(f"  读 {nd} 天用时 {time.time() - t0:.0f}s", flush=True)

    tok_cnt = pool_sum(land.astype(np.float64), P)
    landtok = tok_cnt > 0
    fulltok = tok_cnt == P * P
    # 全域自检: 与 LEDGER 里的 RMSE/MAE 对得上才说明单位与掩膜没错
    rmse = float(np.sqrt(se_pix[land].sum() / (land.sum() * nd)))
    mae = float(ae_pix[land].sum() / (land.sum() * nd))

    tok_tot = tok_se.sum(0)                     # 全年逐 token 平方误差
    tok_mse = np.full(tok_tot.shape, np.nan)
    tok_mse[landtok] = tok_tot[landtok] / (tok_cnt[landtok] * nd)
    static = lorenz(tok_tot[landtok])
    dynamic = lorenz(tok_se[:, landtok].ravel())
    per_day_top20 = float(np.mean([lorenz(tok_se[t][landtok])["top20"] for t in range(nd)]))

    # token-日 log(MSE) 的方差分解: 静态 token 效应 / 逐日效应 / 可加双效应
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(tok_se[:, landtok] / tok_cnt[landtok][None] + 1e-6)   # (nd, K)
    yc = y - y.mean()
    tot_var = float((yc ** 2).mean())
    tok_eff = y.mean(0, keepdims=True) - y.mean()
    day_eff = y.mean(1, keepdims=True) - y.mean()
    r2_tok = 1 - float(((yc - tok_eff) ** 2).mean()) / tot_var
    r2_day = 1 - float(((yc - day_eff) ** 2).mean()) / tot_var
    r2_add = 1 - float(((yc - tok_eff - day_eff) ** 2).mean()) / tot_var
    # 用静态年图排序取前 20% token, 全年捕获的 SE 份额(= 静态 Lorenz top20); 逐日先知取前 20% 的份额(上界)
    # 地形粗糙度与 token 误差的关系
    oro = M.load_static_2d(a.daymet_dir, "orography").astype(np.float64)
    oro0 = np.where(land, oro, 0.0)
    _, elev_std = pool_mean_std(oro0, P, cnt=tok_cnt)
    rel = {"spearman_tokmse_vs_elev_std": spearman(tok_mse[landtok], elev_std[landtok]),
           "spearman_tokmse_vs_land_frac": spearman(tok_mse[landtok], tok_cnt[landtok] / (P * P))}

    reg = load_regions(a.regions)
    res = {
        "exp": os.path.abspath(a.exp), "target": a.target, "year": a.year, "days": nd,
        "patch": P, "tokens_total": int(landtok.size), "tokens_with_land": int(landtok.sum()),
        "tokens_full_land": int(fulltok.sum()),
        "selfcheck": {"rmse": round(rmse, 4), "mae": round(mae, 4),
                      "nan_fraction_outside_domain": nan_outside,
                      "note": "与 LEDGER 同实验的 RMSE/MAE 对拍; 不一致说明单位或掩膜出错"},
        "static_concentration_over_tokens": static,
        "tokenday_concentration": dynamic,
        "per_day_top20_share_mean": round(per_day_top20, 4),
        "log_mse_variance_decomposition": {"r2_static_token": round(r2_tok, 4),
                                           "r2_day": round(r2_day, 4),
                                           "r2_token_plus_day": round(r2_add, 4)},
        "token_mse_percentiles_K2": {f"p{p}": round(float(np.nanpercentile(tok_mse[landtok], p)), 4)
                                     for p in (5, 10, 25, 50, 75, 90, 95, 99)},
        "terrain_relation": rel,
        "regions": region_shares(se_pix / nd, land, reg),
    }
    dump_json(res, out / "summary.json")
    np.save(out / "pixel_mse.npy", np.where(land, se_pix / nd, np.nan).astype(np.float32))
    np.save(out / "token_mse.npy", tok_mse.astype(np.float32))
    np.save(out / "token_se_daily.npy", tok_se.astype(np.float32))

    scales = {}
    scales["pixel_rmse"] = save_map(np.where(land, np.sqrt(se_pix / nd), np.nan),
                                    out / "pixel_rmse_map.png",
                                    f"{Path(a.exp).name} {a.year} per-pixel RMSE [K]", cmap="magma", label="K")
    scales["token_rmse"] = save_map(np.sqrt(tok_mse), out / "token_rmse_map.png",
                                    f"{Path(a.exp).name} {a.year} per-token ({P}x{P}) RMSE [K]",
                                    cmap="magma", label="K")
    xs, ys_s = lorenz_curve(tok_tot[landtok])
    _, ys_d = lorenz_curve(tok_se[:, landtok].ravel())
    save_curve(xs, {"static: tokens ranked by annual SE": (xs, ys_s), "token-days ranked by SE": (xs, ys_d)},
               out / "lorenz.png", "Squared-error concentration", "share of tokens (or token-days)", "cumulative share of squared error",
               ref_diag=True)
    dump_json(scales, out / "scales.json")
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# 2. 误差可预测性
# ---------------------------------------------------------------------------
def _ridge_fit(X, y, lam):
    # X 已标准化并含常数列
    A = X.T @ X
    A[np.diag_indices_from(A)] += lam
    A[-1, -1] -= lam                       # 常数项不正则
    return np.linalg.solve(A, X.T @ y)


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    f = lambda X: np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    return f(Xtr), f(Xte)


def _metrics(y, p, se):
    """回归指标 + 路由视角指标: 前 20% 命中率, 以及按预测排序取前 20% 捕获的真实 SE 份额。"""
    ss = float(((y - p) ** 2).sum()); st = float(((y - y.mean()) ** 2).sum())
    k = max(int(0.2 * len(y)), 1)
    top_true = set(np.argsort(-y)[:k].tolist())
    top_pred = np.argsort(-p)[:k]
    hit = len([i for i in top_pred if i in top_true]) / k
    se_cap_pred = float(se[top_pred].sum() / se.sum())
    se_cap_oracle = float(np.sort(se)[::-1][:k].sum() / se.sum())
    return {"r2": round(1 - ss / st, 4), "pearson": pearson(y, p), "spearman": spearman(y, p),
            "top20_hit": round(hit, 4), "se_share_top20_by_pred": round(se_cap_pred, 4),
            "se_share_top20_oracle": round(se_cap_oracle, 4)}


def _ridge_eval(Xtr, ytr, Xte, yte, se_te):
    Xs, Xt = _standardize(Xtr, Xte)
    n = len(Xs); cut = int(0.8 * n)
    perm = np.random.RandomState(0).permutation(n)
    itr, iva = perm[:cut], perm[cut:]
    best = None
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        w = _ridge_fit(Xs[itr], ytr[itr], lam * len(itr) / 1e4)
        v = float(((Xs[iva] @ w - ytr[iva]) ** 2).mean())
        if best is None or v < best[0]:
            best = (v, lam)
    w = _ridge_fit(Xs, ytr, best[1] * n / 1e4)
    m = _metrics(yte, Xt @ w, se_te); m["lambda"] = best[1]
    return m


def _mlp_eval(Xtr, ytr, Xte, yte, se_te, epochs=20, seed=0):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(min(32, os.cpu_count() or 8))
    Xs, Xt = _standardize(Xtr, Xte)
    Xs = torch.tensor(Xs[:, :-1], dtype=torch.float32); Xt = torch.tensor(Xt[:, :-1], dtype=torch.float32)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    n = len(Xs); cut = int(0.9 * n)
    perm = torch.randperm(n)
    itr, iva = perm[:cut], perm[cut:]
    net = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], 128), torch.nn.SiLU(),
                              torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bs = 8192
    best_state, best_v = None, np.inf
    for ep in range(epochs):
        net.train()
        sh = itr[torch.randperm(len(itr))]
        for i in range(0, len(sh), bs):
            b = sh[i:i + bs]
            loss = torch.nn.functional.mse_loss(net(Xs[b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(Xs[iva]).squeeze(1), yt[iva]))
        if v < best_v:
            best_v, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        p = net(Xt).squeeze(1).numpy() * ys + ym
    m = _metrics(yte, p, se_te); m["epochs"] = epochs
    return m


def run_predictability(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mode = "history_51"
    need, lags = C.pairing_history_days(mode), C.history_lags(mode)
    fi = FrameIndex([a.year], [a.year - 1, a.year], need, lags, split="precheck")
    ds_years = sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
    stats, dd = setup_truth(a, ds_years, mode=mode, cache_years=2)
    layout = list(dd.layout)
    if len(layout) != C.cond_channels(mode):
        raise ValueError(f"条件通道 {len(layout)} 与合同 {C.cond_channels(mode)} 不符")
    land = dd.mask
    ti = C.TARGETS.index(a.target)
    P = a.patch
    H, W = land.shape
    fields = Path(a.exp) / f"fields{a.year}" / a.target / "ens_mean"
    tok_cnt = pool_sum(land.astype(np.float64), P)
    landtok = tok_cnt > 0
    K = int(landtok.sum())

    # token 内地形粗糙度: 原始高程的块内标准差与均值(只在有效域内统计)
    oro = np.where(land, M.load_static_2d(a.daymet_dir, "orography").astype(np.float64), 0.0)
    elev_mean, elev_std = pool_mean_std(oro, P, cnt=tok_cnt)
    dz_idx = layout.index(C.DZ)
    static_extra_names = ["elev_mean", "elev_std", "dz_std", "land_frac"]

    nfr = len(fi) if not a.limit_days else min(len(fi), a.limit_days)
    F = len(layout) + len(static_extra_names)
    X = np.zeros((nfr, K, F), np.float32)
    Y = np.zeros((nfr, K), np.float32)          # log 逐 token MAE
    SE = np.zeros((nfr, K), np.float32)         # 逐 token 平方误差和(路由视角的份额用)
    days = np.zeros(nfr, np.int64)
    t0 = time.time()
    dz_std = None
    for k in range(nfr):
        y, d = fi.frames[k]
        cond, _, _, raw = dd.full(y, d, fi.history_of(k))
        if cond.shape[0] != F - len(static_extra_names):
            raise ValueError(f"cond 通道数 {cond.shape[0]} 异常")
        if dz_std is None:
            dzf = np.where(land, cond[dz_idx].astype(np.float64), 0.0)
            _, dz_std = pool_mean_std(dzf, P, cnt=tok_cnt)
        pred = np.load(fields / f"{y}_d{d}.npy").astype(np.float64)
        if not np.isfinite(pred[land]).all():
            raise ValueError(f"{y} 第 {d} 天有效域内含非有限值")
        e = np.where(land, pred - raw[ti].astype(np.float64), 0.0)
        tok_ae = pool_sum(np.abs(e), P)[landtok] / tok_cnt[landtok]
        tok_se = pool_sum(e * e, P)[landtok]
        feats = cond.reshape(cond.shape[0], H // P, P, W // P, P).mean(axis=(2, 4))   # (Cc, h, w)
        X[k, :, :cond.shape[0]] = feats[:, landtok].T
        X[k, :, cond.shape[0]:] = np.stack([elev_mean[landtok], elev_std[landtok],
                                            dz_std[landtok], tok_cnt[landtok] / (P * P)], 1)
        Y[k] = np.log(tok_ae + 1e-3)
        SE[k] = tok_se
        days[k] = d
        if k % 60 == 0:
            print(f"  帧 {k}/{nfr}  {time.time() - t0:.0f}s", flush=True)

    months = np.array([month_of(a.year, int(d)) for d in days])
    tok_ids = np.arange(K)
    ti_r, ti_c = np.nonzero(landtok)
    checker = ((ti_r + ti_c) % 2 == 0)          # 棋盘划分 token
    names = layout + static_extra_names
    static_cols = [names.index(n) for n in static_extra_names] + \
                  [names.index(C.DOY_SIN), names.index(C.DOY_COS)]

    def flat(sel_days, sel_tok):
        xs = X[sel_days][:, sel_tok].reshape(-1, F)
        ys = Y[sel_days][:, sel_tok].ravel()
        ss = SE[sel_days][:, sel_tok].ravel()
        tk = np.broadcast_to(tok_ids[sel_tok][None], (int(sel_days.sum()), int(sel_tok.sum()))).ravel()
        return xs, ys, ss, tk

    results = {}
    # 时间划分: 奇数月训练, 偶数月测试; 静态气候态基线 = 训练日里每个 token 的均值
    tr_d, te_d = months % 2 == 1, months % 2 == 0
    all_tok = np.ones(K, bool)
    Xtr, ytr, _, tktr = flat(tr_d, all_tok)
    Xte, yte, sete, tkte = flat(te_d, all_tok)
    clim = np.zeros(K); cnt = np.zeros(K)
    np.add.at(clim, tktr, ytr); np.add.at(cnt, tktr, 1)
    clim = clim / np.maximum(cnt, 1)
    results["temporal_split"] = {
        "train_days": int(tr_d.sum()), "test_days": int(te_d.sum()), "tokens": K,
        "baseline_static_token_climatology": _metrics(yte, clim[tkte], sete),
        "ridge_static_terrain_season_only": _ridge_eval(Xtr[:, static_cols], ytr, Xte[:, static_cols], yte, sete),
        "ridge_all_features": _ridge_eval(Xtr, ytr, Xte, yte, sete),
        "ridge_all_plus_token_climatology": _ridge_eval(
            np.concatenate([Xtr, clim[tktr][:, None]], 1), ytr,
            np.concatenate([Xte, clim[tkte][:, None]], 1), yte, sete),
        "mlp_all_features": _mlp_eval(Xtr, ytr, Xte, yte, sete),
    }
    print("  时间划分完成", flush=True)
    # 空间划分: 棋盘 token, 全部日期; 检验"残差头学到的是物理还是位置"
    all_d = np.ones(nfr, bool)
    Xtr, ytr, _, _ = flat(all_d, checker)
    Xte, yte, sete, _ = flat(all_d, ~checker)
    results["spatial_split_checkerboard"] = {
        "train_tokens": int(checker.sum()), "test_tokens": int((~checker).sum()), "days": nfr,
        "ridge_static_terrain_season_only": _ridge_eval(Xtr[:, static_cols], ytr, Xte[:, static_cols], yte, sete),
        "ridge_all_features": _ridge_eval(Xtr, ytr, Xte, yte, sete),
        "mlp_all_features": _mlp_eval(Xtr, ytr, Xte, yte, sete),
    }
    print("  空间划分完成", flush=True)
    res = {"exp": os.path.abspath(a.exp), "target": a.target, "year": a.year, "frames": nfr,
           "patch": P, "features": names, "target_def": "log(逐 token MAE + 1e-3), token 只含有效域像素",
           "results": results}
    dump_json(res, out / "summary.json")
    np.save(out / "token_ids_rc.npy", np.stack([ti_r, ti_c], 1))
    print(json.dumps(res["results"], ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# 3. μ 缓存残差图的持续性与 σ_r 动态范围
# ---------------------------------------------------------------------------
def _persistence_accumulate(a, years, out):
    """逐年逐日累计残差的一阶/二阶矩; 结束时整体落盘, 统计量可从落盘结果反复重算。"""
    stats = Stats(a.era5_dir, a.daymet_dir)
    cache = MuCache(a.mu_cache, [a.target])
    ti = C.TARGETS.index(a.target)
    P = a.patch
    land = None
    S1 = S2 = None
    N = 0
    tok_s2, n_ym = {}, {}
    glob_s2 = np.zeros(12); glob_n = np.zeros(12)
    skipped = 0
    t0 = time.time()
    for y in years:
        dd = DownscaleData(a.era5_dir, a.daymet_dir, [y], stats, mode="baseline_21", era5_cache_years=1)
        if land is None:
            land = dd.mask; H, W = land.shape
            S1 = np.zeros((H, W)); S2 = np.zeros((H, W))
            tok_cnt = pool_sum(land.astype(np.float64), P)
        elif not np.array_equal(land, dd.mask):
            raise ValueError("不同年份的有效域不一致")
        arr = cache._arr(a.target, y)
        nd = min(dd.ndays[y], C.DAYS_PER_YEAR)
        if a.limit_days:
            nd = min(nd, a.limit_days)
        for d in range(nd):
            mu = np.asarray(arr[d], np.float32)
            if not np.isfinite(mu[land]).all():
                skipped += 1
                continue
            truth = dd.target(y, d)[0][ti]
            r = np.where(land, truth.astype(np.float64) - mu, 0.0)
            m = month_of(y, d)
            S1 += r; S2 += r * r; N += 1
            key = (y, m)
            if key not in tok_s2:
                tok_s2[key] = np.zeros(tok_cnt.shape); n_ym[key] = 0
            tok_s2[key] += pool_sum(r * r, P); n_ym[key] += 1
            glob_s2[m - 1] += float((r * r)[land].sum()); glob_n[m - 1] += float(land.sum())
        print(f"  {y} 完成, 累计 {N} 帧, 跳过 {skipped}, {time.time() - t0:.0f}s", flush=True)
    keys = list(tok_s2.keys())
    np.savez(out / "accum.npz", S1=S1, S2=S2, N=N, land=land, tok_cnt=tok_cnt,
             keys=np.array([f"{y}-{m}" for (y, m) in keys]),
             tok_s2=np.stack([tok_s2[k] for k in keys]), n_ym=np.array([n_ym[k] for k in keys]),
             glob_s2=glob_s2, glob_n=glob_n, years=np.array(years), skipped=skipped)
    return dict(S1=S1, S2=S2, N=N, land=land, tok_cnt=tok_cnt, tok_s2=tok_s2, n_ym=n_ym,
                glob_s2=glob_s2, glob_n=glob_n, years=years, skipped=skipped)


def _persistence_load(path):
    z = np.load(path, allow_pickle=False)
    keys = [tuple(int(v) for v in k.split("-")) for k in z["keys"]]
    return dict(S1=z["S1"], S2=z["S2"], N=int(z["N"]), land=z["land"], tok_cnt=z["tok_cnt"],
                tok_s2={k: z["tok_s2"][i] for i, k in enumerate(keys)},
                n_ym={k: int(z["n_ym"][i]) for i, k in enumerate(keys)},
                glob_s2=z["glob_s2"], glob_n=z["glob_n"], years=[int(v) for v in z["years"]],
                skipped=int(z["skipped"]))


def run_persistence(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.from_accum:
        acc = _persistence_load(a.from_accum)
    else:
        cache = MuCache(a.mu_cache, [a.target])
        yrs = cache.years()
        train = [y for y in M.splits["train"] if y in yrs][::a.year_stride]
        extra = [y for y in (2018, 2019, 2020) if y in yrs]
        years = sorted(set(train) | set(extra))
        if a.limit_years:
            years = years[:a.limit_years]
        acc = _persistence_accumulate(a, years, out)
    S1, S2, N, land, tok_cnt = acc["S1"], acc["S2"], acc["N"], acc["land"], acc["tok_cnt"]
    tok_s2, n_ym, glob_s2, glob_n = acc["tok_s2"], acc["n_ym"], acc["glob_s2"], acc["glob_n"]
    years, skipped = acc["years"], acc["skipped"]
    P = a.patch
    landtok = tok_cnt > 0
    extra = [y for y in (2018, 2019, 2020) if y in years]

    mean = S1 / N
    sig = np.sqrt(np.maximum(S2 / N - mean ** 2, 0.0))
    sig_land = sig[land]
    glob = float(np.sqrt(S2[land].sum() / (N * land.sum()) - (S1[land].sum() / (N * land.sum())) ** 2))
    # 逐年与逐 (年, 月) 的 token σ, 只保留陆地 token 的向量, 避免海洋 token 的 NaN 进入统计
    def tokvec(s2, n):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(s2[landtok] / (tok_cnt[landtok] * n))
    tok_sig_year = {}
    for y in years:
        ms = [m for m in range(1, 13) if (y, m) in tok_s2]
        if not ms:
            continue
        tok_sig_year[y] = tokvec(sum(tok_s2[(y, m)] for m in ms), sum(n_ym[(y, m)] for m in ms))
    tok_sig_month = {k: tokvec(s2, n_ym[k]) for k, s2 in tok_s2.items()}

    def logv(v):
        return np.log(v + 1e-6)

    tr_years = [y for y in years if y in M.splits["train"] and y in tok_sig_year]
    pair_sp, pair_pe = [], []
    for i, y1 in enumerate(tr_years):
        for y2 in tr_years[i + 1:]:
            pair_sp.append(spearman(logv(tok_sig_year[y1]), logv(tok_sig_year[y2])))
            pair_pe.append(pearson(logv(tok_sig_year[y1]), logv(tok_sig_year[y2])))
    train_mean = np.sqrt(np.mean(np.stack([tok_sig_year[y] ** 2 for y in tr_years]), 0))
    vs_heldout = {str(y): spearman(logv(train_mean), logv(tok_sig_year[y])) for y in extra if y in tok_sig_year}
    same_month = {}
    for m in range(1, 13):
        ys_ = [y for y in tr_years if (y, m) in tok_sig_month]
        vals = [spearman(logv(tok_sig_month[(ys_[i], m)]), logv(tok_sig_month[(ys_[j], m)]))
                for i in range(len(ys_)) for j in range(i + 1, len(ys_))]
        if vals:
            same_month[m] = round(float(np.mean(vals)), 4)
    cross_month, jan_jul = [], []
    for y in tr_years:
        ms = [m for m in range(1, 13) if (y, m) in tok_sig_month]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                v = spearman(logv(tok_sig_month[(y, ms[i])]), logv(tok_sig_month[(y, ms[j])]))
                cross_month.append(v)
                if (ms[i], ms[j]) == (1, 7):
                    jan_jul.append(v)
    month_mean = {}
    for m in range(1, 13):
        stack = [tok_sig_month[(y, m)] ** 2 for y in tr_years if (y, m) in tok_sig_month]
        if stack:
            month_mean[m] = np.sqrt(np.mean(np.stack(stack), 0))
    if len(month_mean) == 12:
        mm = np.stack([month_mean[m] for m in range(1, 13)])          # (12, K) 只含陆地 token
        amp = mm.max(0) / np.maximum(mm.min(0), 1e-9)
        season_amp = {f"p{p}": round(float(np.percentile(amp, p)), 3) for p in (10, 50, 90)}
        peak_month = mm.argmax(0) + 1
        peak_hist = {int(m): int((peak_month == m).sum()) for m in range(1, 13)}
    else:
        season_amp, peak_hist = None, None

    def full_map(vec):
        m_ = np.full(tok_cnt.shape, np.nan, np.float32); m_[landtok] = vec
        return m_

    rs_path = Path(a.mu_cache) / "residual_scale.json"
    ref = json.loads(rs_path.read_text()) if rs_path.exists() else {}
    res = {
        "mu_cache": os.path.abspath(a.mu_cache), "target": a.target, "years": years,
        "frames": int(N), "skipped_no_history": int(skipped), "patch": P,
        "space": "normalized target space",
        "selfcheck": {"global_sigma_r": round(glob, 5),
                      "residual_scale_json": ref.get("residual_std"),
                      "note": "训练年全量时应与 residual_scale.json 一致(这里还含 2018-2020)"},
        "pixel_sigma_r_percentiles": {f"p{p}": round(float(np.percentile(sig_land, p)), 5)
                                      for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        "pixel_sigma_r_ratio": {"p90_over_p10": round(float(np.percentile(sig_land, 90) / np.percentile(sig_land, 10)), 3),
                                "p99_over_p50": round(float(np.percentile(sig_land, 99) / np.percentile(sig_land, 50)), 3),
                                "frac_below_half_global": round(float((sig_land < 0.5 * glob).mean()), 4),
                                "frac_above_twice_global": round(float((sig_land > 2 * glob).mean()), 4)},
        "token_sigma_r_ratio_train_mean": {"p90_over_p10": round(float(np.percentile(train_mean, 90) / np.percentile(train_mean, 10)), 3),
                                           "p99_over_p50": round(float(np.percentile(train_mean, 99) / np.percentile(train_mean, 50)), 3)},
        "monthly_global_sigma_r": {int(m + 1): round(float(np.sqrt(glob_s2[m] / glob_n[m])), 5)
                                   for m in range(12) if glob_n[m] > 0},
        "token_map_persistence": {
            "train_years": tr_years,
            "pairwise_annual_spearman": {"mean": round(float(np.mean(pair_sp)), 4), "min": round(float(np.min(pair_sp)), 4)} if pair_sp else None,
            "pairwise_annual_pearson_log": {"mean": round(float(np.mean(pair_pe)), 4), "min": round(float(np.min(pair_pe)), 4)} if pair_pe else None,
            "train_mean_vs_heldout_spearman": vs_heldout,
            "same_month_across_years_spearman": same_month,
            "cross_month_within_year_spearman": {"mean": round(float(np.mean(cross_month)), 4),
                                                 "min": round(float(np.min(cross_month)), 4)} if cross_month else None,
            "jan_vs_jul_spearman_mean": round(float(np.mean(jan_jul)), 4) if jan_jul else None,
            "seasonal_amplitude_max_over_min_per_token": season_amp,
            "peak_month_histogram": peak_hist,
        },
    }
    dump_json(res, out / "summary.json")
    np.save(out / "pixel_sigma_r.npy", np.where(land, sig, np.nan).astype(np.float32))
    np.save(out / "token_sigma_r_train_mean.npy", full_map(train_mean))
    if len(month_mean) == 12:
        np.save(out / "token_sigma_r_monthly_train_mean.npy",
                np.stack([full_map(month_mean[m]) for m in range(1, 13)]))
    scales = {"pixel_sigma_r_rel": save_map(np.where(land, sig / glob, np.nan), out / "pixel_sigma_r_rel_map.png",
                                            f"per-pixel sigma_r / global sigma_r ({Path(a.mu_cache).name})",
                                            vmin=0.0, vmax=3.0, cmap="viridis", label="sigma_r / global")}
    dump_json(scales, out / "scales.json")
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# 4. 可约部分: real 与 oracle 的逐 token 误差之差
# ---------------------------------------------------------------------------
def run_reducible(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats, dd = setup_truth(a, [a.year])
    land = dd.mask
    ti = C.TARGETS.index(a.target)
    P = a.patch
    tok_cnt = pool_sum(land.astype(np.float64), P)
    landtok = tok_cnt > 0
    all_res = {}
    scales = {}
    for spec in a.pair:
        label, dirs = spec.split("=", 1)
        real_dir, orc_dir = dirs.split(":", 1)
        r = {}
        maps = {}
        for kind, dpath in (("real", real_dir), ("oracle", orc_dir)):
            fields = Path(dpath) / f"fields{a.year}" / a.target / "ens_mean"
            se_pix, ae_pix, tok_se, _, nd, _ = field_error_accumulate(fields, dd, ti, a.year, P, land, a.limit_days)
            r[f"{kind}_rmse"] = round(float(np.sqrt(se_pix[land].sum() / (land.sum() * nd))), 4)
            r[f"{kind}_mae"] = round(float(ae_pix[land].sum() / (land.sum() * nd)), 4)
            tot = tok_se.sum(0)
            maps[kind] = np.where(landtok, tot / np.maximum(tok_cnt * nd, 1), np.nan)
        mr, mo = maps["real"][landtok], maps["oracle"][landtok]
        red_se = np.maximum(mr - mo, 0.0)
        red_frac = np.clip(1 - mo / np.maximum(mr, 1e-9), 0, 1)
        k = max(int(0.2 * len(mr)), 1)
        by_total = np.argsort(-mr)[:k]
        by_red = np.argsort(-red_se)[:k]
        r.update({
            "reducible_share_of_real_se": round(float((red_se * tok_cnt[landtok]).sum() / (mr * tok_cnt[landtok]).sum()), 4),
            "reducible_fraction_percentiles": {f"p{p}": round(float(np.percentile(red_frac, p)), 4) for p in (10, 25, 50, 75, 90)},
            "spearman_total_mse_vs_reducible_se": spearman(mr, red_se),
            "spearman_total_mse_vs_reducible_fraction": spearman(mr, red_frac),
            "reducible_se_lorenz": lorenz(red_se * tok_cnt[landtok]),
            "top20_by_total_captures_reducible_share": round(float((red_se * tok_cnt[landtok])[by_total].sum() / (red_se * tok_cnt[landtok]).sum()), 4),
            "top20_by_reducible_captures_reducible_share": round(float((red_se * tok_cnt[landtok])[by_red].sum() / (red_se * tok_cnt[landtok]).sum()), 4),
            "real_dir": os.path.abspath(real_dir), "oracle_dir": os.path.abspath(orc_dir), "days": nd,
        })
        all_res[label] = r
        full = np.full(tok_cnt.shape, np.nan); full[landtok] = red_frac
        np.save(out / f"token_reducible_fraction_{label}.npy", full.astype(np.float32))
        np.save(out / f"token_mse_real_{label}.npy", maps["real"].astype(np.float32))
        np.save(out / f"token_mse_oracle_{label}.npy", maps["oracle"].astype(np.float32))
        scales[f"reducible_fraction_{label}"] = save_map(full, out / f"token_reducible_fraction_{label}.png",
                                                         f"{label}: per-token reducible error fraction 1 - MSE_oracle/MSE_real",
                                                         vmin=0.0, vmax=1.0, cmap="cividis")
        print(f"  {label}: {json.dumps(r, ensure_ascii=False)}", flush=True)
    dump_json({"target": a.target, "year": a.year, "patch": P, "pairs": all_res}, out / "summary.json")
    dump_json(scales, out / "scales.json")


# ---------------------------------------------------------------------------
# 5. 块内误差分解: 地形渲染头的杠杆
# ---------------------------------------------------------------------------
def run_patchdecomp(a):
    """把逐像素误差按 P×P 块拆成块均值分量与块内分量, 并量块内分量能被块内高程异常的
    线性项解释多少: 逐 token-日各自最优斜率是上限, 全年固定斜率是保底。同时给出块内的
    "细尺度技能" = 1 − SE_within / Var_within(truth)。"""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats, dd = setup_truth(a, [a.year])
    land = dd.mask
    ti = C.TARGETS.index(a.target)
    P = a.patch
    H, W = land.shape
    landf = land.astype(np.float64)
    n_k = pool_sum(landf, P)
    landtok = n_k > 0
    z = np.where(land, M.load_static_2d(a.daymet_dir, "orography").astype(np.float64), 0.0)
    Sz = pool_sum(z, P); Sz2 = pool_sum(z * z, P)
    with np.errstate(invalid="ignore", divide="ignore"):
        Zk = np.where(landtok, Sz2 - Sz * Sz / np.maximum(n_k, 1), 0.0)   # 块内高程平方和(去均值)
    Zk = np.maximum(Zk, 0.0)
    zvar_tok = np.where(landtok, Zk / np.maximum(n_k, 1), np.nan)
    results = {}
    nd_all = None
    for spec in a.exp:
        label, d = spec.split("=", 1)
        fields = Path(d) / f"fields{a.year}" / a.target / "ens_mean"
        nd = min(dd.ndays[a.year], C.DAYS_PER_YEAR)
        if a.limit_days:
            nd = min(nd, a.limit_days)
        nd_all = nd
        SE_tot = np.zeros(n_k.shape); SE_mean = np.zeros(n_k.shape); SE_within = np.zeros(n_k.shape)
        SE_expl_daily = np.zeros(n_k.shape); Csum = np.zeros(n_k.shape)
        VarW_truth = np.zeros(n_k.shape)
        for t in range(nd):
            pred = np.load(fields / f"{a.year}_d{t}.npy").astype(np.float64)
            if not np.isfinite(pred[land]).all():
                raise ValueError(f"{label} {a.year} 第 {t} 天有效域内含非有限值")
            truth = dd.target(a.year, t)[1][ti].astype(np.float64)
            e = np.where(land, pred - truth, 0.0)
            y = np.where(land, truth, 0.0)
            Se = pool_sum(e, P); Se2 = pool_sum(e * e, P)
            Sy = pool_sum(y, P); Sy2 = pool_sum(y * y, P)
            with np.errstate(invalid="ignore", divide="ignore"):
                se_mean = np.where(landtok, Se * Se / np.maximum(n_k, 1), 0.0)
                vw = np.where(landtok, Sy2 - Sy * Sy / np.maximum(n_k, 1), 0.0)
                Cez = pool_sum(e * z, P) - Se * Sz / np.maximum(n_k, 1)      # Σ e_within z_within
                expl = np.where(Zk > 0, Cez * Cez / np.maximum(Zk, 1e-12), 0.0)
            SE_tot += Se2; SE_mean += se_mean; SE_within += Se2 - se_mean
            SE_expl_daily += expl; Csum += np.where(landtok, Cez, 0.0); VarW_truth += vw
        SE_expl_static = np.where(Zk > 0, Csum * Csum / (nd * np.maximum(Zk, 1e-12)), 0.0)
        L = landtok
        tot = float(SE_tot[L].sum())
        rmse = float(np.sqrt(tot / (land.sum() * nd)))
        within_share = float(SE_within[L].sum() / tot)
        skill_within = 1.0 - float(SE_within[L].sum() / VarW_truth[L].sum())
        expl_daily_share = float(SE_expl_daily[L].sum() / SE_within[L].sum())
        expl_static_share = float(SE_expl_static[L].sum() / SE_within[L].sum())
        # 若块内误差被逐日/固定斜率修掉, 全域 RMSE 会变成多少
        rmse_if_daily = float(np.sqrt((tot - SE_expl_daily[L].sum()) / (land.sum() * nd)))
        rmse_if_static = float(np.sqrt((tot - SE_expl_static[L].sum()) / (land.sum() * nd)))
        rmse_if_no_within = float(np.sqrt(SE_mean[L].sum() / (land.sum() * nd)))
        with np.errstate(invalid="ignore", divide="ignore"):
            within_mse_tok = np.where(L, SE_within / (np.maximum(n_k, 1) * nd), np.nan)
        rho = spearman(within_mse_tok[L], zvar_tok[L])
        # 块内误差份额随地形粗糙度分层(按块内高程标准差四分位)
        zs = np.sqrt(zvar_tok[L]); q = np.nanpercentile(zs, [25, 50, 75])
        bins = np.digitize(zs, q)
        strata = {}
        for b, name in enumerate(["q1_flat", "q2", "q3", "q4_rough"]):
            m = bins == b
            strata[name] = {"tokens": int(m.sum()),
                            "within_share": round(float(SE_within[L][m].sum() / SE_tot[L][m].sum()), 4),
                            "se_share_of_total": round(float(SE_tot[L][m].sum() / tot), 4),
                            "skill_within": round(1 - float(SE_within[L][m].sum() / VarW_truth[L][m].sum()), 4)}
        results[label] = {"dir": os.path.abspath(d), "rmse_check": round(rmse, 4),
                          "se_share_patch_mean": round(1 - within_share, 4),
                          "se_share_within_patch": round(within_share, 4),
                          "within_patch_skill_r2": round(skill_within, 4),
                          "within_se_explained_by_daily_lapse_slope": round(expl_daily_share, 4),
                          "within_se_explained_by_static_lapse_slope": round(expl_static_share, 4),
                          "rmse_if_within_removed": round(rmse_if_no_within, 4),
                          "rmse_if_daily_slope_fixed": round(rmse_if_daily, 4),
                          "rmse_if_static_slope_fixed": round(rmse_if_static, 4),
                          "spearman_within_mse_vs_elev_var": rho,
                          "strata_by_within_patch_elev_std": strata}
        np.save(out / f"token_within_mse_{label}.npy", within_mse_tok.astype(np.float32))
        print(f"  {label}: {json.dumps(results[label], ensure_ascii=False)}", flush=True)
    res = {"target": a.target, "year": a.year, "days": nd_all, "patch": P,
           "definitions": {"se_share_within_patch": "块内(去块均值)误差平方和 / 总误差平方和",
                           "within_patch_skill_r2": "1 − SE_within / Var_within(truth); 块内细尺度被解释的比例",
                           "explained_by_lapse_slope": "块内误差在块内高程异常上的线性投影所占 SE_within 的比例; daily=逐 token-日最优斜率(上限), static=全年固定斜率(保底)"},
           "elev_var_tok_percentiles_m2": {f"p{p}": round(float(np.nanpercentile(zvar_tok[landtok], p)), 1) for p in (10, 50, 90)},
           "results": results}
    dump_json(res, out / "summary.json")
    np.save(out / "token_elev_var.npy", zvar_tok.astype(np.float32))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--target", default="2m_temperature_max", choices=C.TARGETS)
        p.add_argument("--year", type=int, default=2020)
        p.add_argument("--patch", type=int, default=16)
        p.add_argument("--era5-dir", default=M.ERA5_DIR)
        p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
        p.add_argument("--out", required=True)
        p.add_argument("--limit-days", type=int, default=0, help="冒烟用: 只跑前 N 天")

    p = sub.add_parser("concentration"); common(p)
    p.add_argument("--exp", required=True)
    p.add_argument("--regions", default="", help="regions_v1.npz, 给了就按区域分份额")
    p = sub.add_parser("predictability"); common(p)
    p.add_argument("--exp", required=True)
    p = sub.add_parser("persistence"); common(p)
    p.add_argument("--mu-cache", required=True)
    p.add_argument("--year-stride", type=int, default=1, help="训练年抽样步长")
    p.add_argument("--limit-years", type=int, default=0, help="冒烟用: 只跑前 N 年")
    p.add_argument("--from-accum", default="", help="从 accum.npz 直接重算统计量, 不再遍历数据")
    p = sub.add_parser("reducible"); common(p)
    p.add_argument("--pair", action="append", required=True, help="label=<real_exp_dir>:<oracle_exp_dir>")
    p = sub.add_parser("patchdecomp"); common(p)
    p.add_argument("--exp", action="append", required=True, help="label=<exp_dir>, 可多次")
    a = ap.parse_args()
    {"concentration": run_concentration, "predictability": run_predictability,
     "persistence": run_persistence, "reducible": run_reducible,
     "patchdecomp": run_patchdecomp}[a.cmd](a)


if __name__ == "__main__":
    main()
