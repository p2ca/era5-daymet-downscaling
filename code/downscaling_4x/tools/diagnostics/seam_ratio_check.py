# -*- coding: utf-8 -*-
"""patch 拼接缝验收: seam ratio = 边界梯度 / 邻近非边界梯度, 目标 ≤ 1.10。

读 stageb_dump 的落场目录(ens_mean/<年>_d<日>.npy, 域外 NaN), 缝的位置取自
GridPatching2D overlap_count 的跳变处; 物理单位、仅有效域、逐日算, 取 x/y 两向较大者;
按全年 max 判 PASS/FAIL, 结果写 <fields>/seam_ratio.json。

用法:
  python -m downscaling_4x.tools.diagnostics.seam_ratio_check --fields runs/exp/<id>-eval2020 [--figs]
patch/overlap/boundary 缺省读落场 meta.json 的 config。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from downscaling_4x.models.patching import GridPatching2D

TARGET_RATIO = 1.10


def seam_positions(patch, overlap, boundary, H, W):
    """overlap_count 跳变处 = 缝所在的列/行(去 padding 后的图像坐标)。"""
    oc = GridPatching2D.get_overlap_count((patch, patch), (H, W), overlap, boundary)[0, 0].numpy()
    oc = oc[boundary:boundary + H, boundary:boundary + W]
    bx = np.where(np.diff(oc[H // 2, :]) != 0)[0]
    by = np.where(np.diff(oc[:, W // 2]) != 0)[0]
    return bx, by


def _dir_ratio(line_mean, bnd):
    """一维方向: 各缝处梯度 / 紧邻(±2..3)非缝处梯度, 对缝取平均。"""
    r, n = [], len(line_mean)
    for c in bnd:
        if c < 3 or c > n - 4:
            continue
        base = np.nanmean(np.r_[line_mean[c - 3:c - 1], line_mean[c + 2:c + 4]])
        if np.isfinite(base) and base > 0 and np.isfinite(line_mean[c]):
            r.append(line_mean[c] / base)
    return float(np.nanmean(r)) if r else float("nan")


def seam_ratio(field, land, bx, by):
    """max(x 向, y 向); 有效域上逐列/逐行平均一阶差分后取缝/邻近比。"""
    field = np.asarray(field, np.float64)
    dx, dy = np.abs(np.diff(field, axis=1)), np.abs(np.diff(field, axis=0))
    lx, ly = land[:, :-1] & land[:, 1:], land[:-1, :] & land[1:, :]
    with np.errstate(invalid="ignore"):
        cm = np.where(lx.sum(0) > 0, np.nansum(np.where(lx, dx, 0), 0) / np.maximum(lx.sum(0), 1), np.nan)
        rm = np.where(ly.sum(1) > 0, np.nansum(np.where(ly, dy, 0), 1) / np.maximum(ly.sum(1), 1), np.nan)
    return max(_dir_ratio(cm, bx), _dir_ratio(rm, by))


def overlay_fig(field, land, bx, by, title, out_path, box=600):
    """代表日 ens_mean 局部放大 + patch 网格叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from downscaling_4x.evaluation.metrics import pick_land_box
    r0, c0, bs = pick_land_box(land, box)
    sub = np.where(land, field, np.nan)[r0:r0 + bs, c0:c0 + bs]
    vmin, vmax = np.nanpercentile(sub, [2, 98])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(sub, cmap="turbo", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    for x in bx:
        if c0 < x < c0 + bs:
            ax.axvline(x - c0, color="w", lw=0.5, alpha=0.6)
    for yy in by:
        if r0 < yy < r0 + bs:
            ax.axhline(yy - r0, color="w", lw=0.5, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="stageb_dump 落场的 patch 拼接缝验收")
    ap.add_argument("--fields", required=True, help="落场目录(含 ens_mean/ 与 meta.json)")
    ap.add_argument("--sub", default="ens_mean", help="检查哪个场(ens_mean 或 spread)")
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--boundary", type=int, default=None)
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--figs", action="store_true", help="代表日(全年中位)加 patch 网格叠加图")
    a = ap.parse_args()
    fields = Path(a.fields)
    meta = json.load(open(fields / "meta.json"))
    cfg = meta.get("config", {})
    patch = a.patch or int(cfg["patch"]); ov = a.overlap if a.overlap is not None else int(cfg["overlap"])
    bd = a.boundary if a.boundary is not None else int(cfg["boundary"])
    files = sorted(p for y in a.years for p in (fields / a.sub).glob(f"{y}_d*.npy"))
    if not files:
        raise SystemExit(f"{fields / a.sub} 下没有场文件")
    first = np.load(files[0]); land = np.isfinite(first); H, W = land.shape
    bx, by = seam_positions(patch, ov, bd, H, W)
    per_day = {}
    for p in files:
        f = np.load(p)
        per_day[p.stem] = seam_ratio(np.where(land, f, np.nan), land, bx, by)
    vals = np.array(list(per_day.values()))
    res = {"fields": str(fields), "sub": a.sub, "patch": patch, "overlap": ov, "boundary": bd,
           "seams_x": [int(x) for x in bx], "seams_y": [int(y) for y in by],
           "n_days": len(vals), "min": float(np.nanmin(vals)), "mean": float(np.nanmean(vals)),
           "max": float(np.nanmax(vals)), "target": TARGET_RATIO,
           "verdict": "PASS" if np.nanmax(vals) <= TARGET_RATIO else "FAIL",
           "definition": "缝处陆地平均一阶差分 / 紧邻(±2..3 px)非缝处, x/y 取大; 物理单位, 逐日",
           "per_day": {k: round(v, 4) for k, v in per_day.items()}}
    out = fields / f"seam_ratio{'' if a.sub == 'ens_mean' else '_' + a.sub}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[seam] {fields.name} {a.sub}: patch {patch} ov {ov} bd {bd}, 缝 x{len(bx)}/y{len(by)}, {len(vals)} 天 "
          f"ratio min {res['min']:.3f} mean {res['mean']:.3f} max {res['max']:.3f} (目标 ≤ {TARGET_RATIO}) -> {res['verdict']}; {out}")
    if a.figs:
        p = files[len(files) // 2]
        fig_dir = fields / "figs"; fig_dir.mkdir(exist_ok=True)
        fp = fig_dir / f"seam_grid_{a.sub}__{fields.name}.png"
        overlay_fig(np.load(p), land, bx, by, f"{a.sub} {meta.get('target', '')}  patch{patch}/ov{ov}/bd{bd}  {p.stem}  + patch grid", fp)
        print(f"[seam] 叠加图 -> {fp}")


if __name__ == "__main__":
    main()
