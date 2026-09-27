# -*- coding: utf-8 -*-
"""渲染驱动: 以 stageb_dump 落盘的场为输入, 热插拔地跑启用的图模块。

  列出模块:  python -m downscaling_4x.evaluation.render.cli --list
  出图:      python -m downscaling_4x.evaluation.render.cli \
               --fields <dump_dir> --regions runs/exp/<regions>/regions_v1.npz \
               --target 2m_temperature_max [--only a,b | --skip c] [--out <dir>]

缺 --out 时默认落 <fields>/figs。某模块所需场缺失则跳过并记入 manifest; 单模块抛错不影响其余。
"""
import argparse
import json
from pathlib import Path

from downscaling_4x.evaluation.render import figures as _figures  # noqa: F401  触发自动发现
from downscaling_4x.evaluation.render.registry import REGISTRY, resolve


def main():
    ap = argparse.ArgumentParser(description="阶段 B 场渲染(热插拔单图)")
    ap.add_argument("--fields", help="stageb_dump 输出目录(含 crps/ ens_mean/ ...)")
    ap.add_argument("--regions", help="regions_v1.npz")
    ap.add_argument("--target")
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--out", default=None, help="缺省 <fields>/figs")
    ap.add_argument("--mu-cache", default=None)
    ap.add_argument("--failure-regions", default="",
                    help="逗号分隔的失败地区名(人工从贡献率排名里选), 供失败图/逐月图使用")
    ap.add_argument("--days", type=int, nargs="+", default=[],
                    help="单日图的日索引(0 起), 可给多个; 不给则不出单日图。缺省值会让不同模型"
                         "的单日图落在不同天气上, 因此不设缺省")
    ap.add_argument("--box", action="append", default=[], metavar="lon0,lon1,lat0,lat1",
                    help="在每张地图上叠一个经纬矩形标注框, 可重复; 与场共用 imshow 的 extent, "
                         "同一组框在各模型图上落在同一片地面")
    ap.add_argument("--scales", default="",
                    help="共享色标 JSON 的路径; 多个模型指向同一份即落在同一把尺子上。"
                         "缺省各自写 <out>/scales.json")
    ap.add_argument("--model-tag", default="",
                    help="进图注与文件名的模型标识; 缺省从落场 meta.json 的 checkpoint 路径推出")
    ap.add_argument("--interp", default="nearest",
                    help="imshow 的重采样方式(纯显示): nearest(缺省, 逐格硬边) / bilinear / antialiased")
    ap.add_argument("--display-smooth", type=float, default=0.0,
                    help="画前在有效域内做高斯平滑的 σ(格), 纯显示用; >0 时图标题会标出来。"
                         "指标一律取自未平滑的场")
    ap.add_argument("--only", default="", help="逗号分隔, 只出这些模块")
    ap.add_argument("--skip", default="", help="逗号分隔, 跳过这些模块")
    ap.add_argument("--list", action="store_true", help="列出已注册模块并退出")
    a = ap.parse_args()

    if a.list:
        for n in REGISTRY:
            f = REGISTRY[n]
            print(f"{n:24s} needs=[{','.join(f.needs)}]  {f.doc}")
        return

    for req in ("fields", "regions", "target"):
        if getattr(a, req) is None:
            ap.error(f"--{req} 必填(除非 --list)")

    from downscaling_4x.evaluation.render.context import RenderContext
    boxes = []
    for spec in a.box:
        v = [float(x) for x in spec.split(",")]
        if len(v) != 4:
            ap.error(f"--box 需要 lon0,lon1,lat0,lat1 四个数, 得到 {spec!r}")
        boxes.append((min(v[0], v[1]), max(v[0], v[1]), min(v[2], v[3]), max(v[2], v[3])))
    out = a.out or str(Path(a.fields) / "figs")
    ctx = RenderContext(a.fields, a.regions, a.target, out, years=a.years,
                        mu_cache=a.mu_cache,
                        failure_regions=[s for s in a.failure_regions.split(",") if s],
                        days=a.days, scales_path=(a.scales or None),
                        model_tag=(a.model_tag or None), boxes=boxes, interp=a.interp,
                        display_smooth=a.display_smooth)

    only = [s for s in a.only.split(",") if s]
    skip = [s for s in a.skip.split(",") if s]
    names = resolve(only, skip)
    manifest = {"target": a.target, "fields": str(Path(a.fields)), "out": str(ctx.out),
                "days": list(ctx.days), "scales": str(ctx._scales_path),
                "model_tag": ctx.model_tag, "model_label": ctx.model_label,
                "members": ctx.members,
                # 纯显示参数也记一笔: 只看 PNG 分不出原生与平滑过的图, 而指标恒取未平滑场
                "interp": ctx.interp, "display_smooth": ctx.display_smooth,
                "enabled": names, "produced": {}, "skipped_missing_field": {},
                "skipped_partial_year": {}, "errors": {}}
    have, want = ctx.coverage()
    full = ctx.is_full_year()
    manifest["coverage"] = {"days_available": have, "days_expected": want, "full_year": full}
    if not full:
        print(f"[render] 落场只覆盖 {have}/{want} 天: 依赖整年的模块将被跳过", flush=True)
    for n in names:
        f = REGISTRY[n]
        if f.full_year and not full:
            manifest["skipped_partial_year"][n] = f"{have}/{want}"
            print(f"[render] 跳过 {n}: 该图按整年定义, 落场只有 {have}/{want} 天")
            continue
        miss = [fld for fld in f.needs if not ctx.has_field(fld)]
        if miss:
            manifest["skipped_missing_field"][n] = miss
            print(f"[render] 跳过 {n}: 缺场 {miss}")
            continue
        try:
            arts = f.fn(ctx) or []
            arts = [str(p) for p in (arts if isinstance(arts, (list, tuple)) else [arts])]
            manifest["produced"][n] = arts
            print(f"[render] {n} -> {arts}")
        except Exception as e:
            manifest["errors"][n] = repr(e)
            print(f"[render] {n} 失败: {e!r}")

    ctx.write_scales()
    json.dump(manifest, open(ctx.out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"[render] 完成 -> {ctx.out} (出 {len(manifest['produced'])} 组, "
          f"跳 {len(manifest['skipped_missing_field'])}, 错 {len(manifest['errors'])})")


if __name__ == "__main__":
    main()
