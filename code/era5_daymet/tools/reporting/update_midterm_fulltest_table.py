#!/usr/bin/env python3
# Packaged implementation; the original code/ path remains compatible.
"""Update the single maintained project-report metric table from full-test JSON files.

The script deliberately refuses subsampled inputs.  Every source must report
2020 coverage of exactly 365/365 days with eval_stride=1 before the HTML report
is changed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile


from era5_daymet.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_RESULTS = ROOT / "runs/exp/20260723-fulltest-eval"
DEFAULT_REPORT = ROOT / "docs/reports/report.html"
LEGACY_REPORT_DIR = ROOT / "runs/exp/20260720-midterm-report"
VARIABLES = (
    ("2m_temperature_max", ("rmse", "ssim", "crps")),
    ("2m_temperature_min", ("rmse", "ssim", "crps")),
    ("total_precipitation_24hr", ("rmse", "ssim_log", "crps")),
)
METHODS = (
    ("BCSD", "baseline/metrics_all.json", "bcsd", ""),
    ("UNet", "baseline/metrics_all.json", "unet", ""),
    ("ViT-crop60", "vit-crop60/metrics_all.json", "vit", ""),
    ("ViT-crop60-30ep", "vit-crop60-30ep/metrics_all.json", "vit", ""),
    ("ViT-crop132", "vit-crop132/metrics_all.json", "vit", "nw"),
    ("ViT-crop60+reg", "vit-crop60reg/metrics_all.json", "vit", "nw"),
    ("ViT-global", "vit-global/metrics_all.json", "vit", "nw"),
    ("CorrDiff", "corrdiff/metrics_corrdiff.json", "corrdiff", ""),
)


def load_full_metrics(
    result_root: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, object]]]:
    rows: dict[str, dict[str, dict[str, float]]] = {}
    sources: dict[str, dict[str, object]] = {}
    cache: dict[Path, dict[str, object]] = {}
    errors: list[str] = []

    for label, relative, key, _row_class in METHODS:
        path = result_root / relative
        try:
            payload = cache.setdefault(path, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: cannot read {path}: {exc}")
            continue

        nd = payload.get("_n_test_days")
        nt = payload.get("_n_total_test_days")
        stride = payload.get("_eval_stride")
        coverage = str(payload.get("_coverage", ""))
        if nd != 365 or nt != 365 or stride != 1 or not coverage.startswith("FULL"):
            errors.append(
                f"{label}: not canonical full-test metrics "
                f"(n={nd}, total={nt}, stride={stride}, coverage={coverage!r})"
            )
            continue
        if key not in payload:
            errors.append(f"{label}: method key {key!r} missing from {path}")
            continue

        method_metrics = payload[key]
        missing_vars = [name for name, _ in VARIABLES if name not in method_metrics]
        if missing_vars:
            errors.append(f"{label}: variables missing from {path}: {missing_vars}")
            continue

        rows[label] = method_metrics
        sources[str(path.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "coverage": coverage,
            "n_test_days": nd,
        }

    if errors:
        raise RuntimeError("Full-test input validation failed:\n  - " + "\n  - ".join(errors))
    return rows, sources


def best_values(rows: dict[str, dict[str, dict[str, float]]]) -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    for variable, metrics in VARIABLES:
        for metric in metrics:
            values = [float(row[variable][metric]) for row in rows.values()]
            best[(variable, metric)] = max(values) if metric in ("ssim", "ssim_log") else min(values)
    return best


def formatted(value: float, metric: str, variable: str, is_best: bool) -> str:
    if variable == "total_precipitation_24hr" and metric in ("rmse", "crps"):
        text = f"{value:.4f}"
    elif metric in ("ssim", "ssim_log"):
        text = f"{value:.3f}"
    else:
        text = f"{value:.2f}"
    if text.startswith("0."):
        text = text[1:]
    return f'<span class="win">{text}</span>' if is_best else text


def build_table(rows: dict[str, dict[str, dict[str, float]]]) -> str:
    best = best_values(rows)
    body: list[str] = []
    for label, _relative, _key, row_class in METHODS:
        cells: list[str] = []
        for variable, metrics in VARIABLES:
            parts = []
            for metric in metrics:
                value = float(rows[label][variable][metric])
                parts.append(formatted(value, metric, variable, value == best[(variable, metric)]))
            cells.append("/".join(parts))
        klass = f' class="{row_class}"' if row_class else ""
        body.append(
            f"        <tr{klass}><td>{label}</td>"
            + "".join(f'<td class="n">{cell}</td>' for cell in cells)
            + "</tr>"
        )

    return "\n".join(
        (
            "    <table>",
            "      <caption>2020 全年 365 天口径,物理空间,陆地掩膜。"
            "RMSE↓ / SSIM↑ / CRPS↓,每项加粗最优。"
            "CorrDiff 为生成式(16 成员),RMSE 用集合均值;其余确定性方法 CRPS=MAE。</caption>",
            "      <thead><tr><th>方法</th><th>tmax RMSE/SSIM/CRPS</th>"
            "<th>tmin RMSE/SSIM/CRPS</th><th>precip RMSE/SSIM<sub>log</sub>/CRPS</th></tr></thead>",
            "      <tbody>",
            *body,
            "      </tbody>",
            "    </table>",
        )
    )


def replace_metric_table(html: str, table: str) -> str:
    heading = "<h2>指标与年均结果图</h2>"
    heading_at = html.find(heading)
    if heading_at < 0:
        raise ValueError("metric section heading not found")
    table_at = html.find("<table", heading_at)
    if table_at < 0:
        raise ValueError("metric table start not found")
    table_end = html.find("</table>", table_at)
    if table_end < 0:
        raise ValueError("metric table end not found")
    table_end += len("</table>")
    updated = html[:table_at] + table + html[table_end:]

    new_table_end = table_at + len(table)
    footer_start = updated.find("    数据源 ·", new_table_end)
    if footer_start < 0:
        raise ValueError("data-source footer not found")
    footer_end = updated.find("<br>", footer_start)
    if footer_end < 0:
        raise ValueError("data-source footer terminator not found")
    footer_end += len("<br>")
    footer = (
        "    指标数据源 · runs/exp/20260723-fulltest-eval (2020 FULL 365/365); "
        "年均图数据源 · 20260720-result-maps-annual<br>"
    )
    return updated[:footer_start] + footer + updated[footer_end:]


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=None)
    # Compatibility for a login-node pipeline started before the report move.
    parser.add_argument("--report-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.report is not None and args.report_dir is not None:
        parser.error("--report and the legacy --report-dir cannot be used together")

    if args.report is not None:
        report_path = args.report
    elif args.report_dir is not None:
        report_dir = args.report_dir.resolve()
        report_path = (
            DEFAULT_REPORT
            if report_dir == LEGACY_REPORT_DIR.resolve()
            else report_dir / DEFAULT_REPORT.name
        )
    else:
        report_path = DEFAULT_REPORT

    rows, sources = load_full_metrics(args.results)
    table = build_table(rows)
    if args.check_only:
        print("All full-test metric inputs passed the 365/365 coverage guard.")
        return

    original = report_path.read_text(encoding="utf-8")
    updated = replace_metric_table(original, table)
    if updated.count("2020 全年 365 天口径") != 1:
        raise RuntimeError(
            f"{report_path}: updated full-test caption count is not exactly one"
        )
    atomic_write(report_path, updated)
    print(f"updated {report_path}")

    manifest = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_root": str(args.results),
        "coverage_guard": "all sources: 365/365 days, stride=1, _coverage starts FULL",
        "sources": sources,
        "table_rows": rows,
        "report_file": str(report_path),
    }
    manifest_path = args.results / "fulltest_table_manifest.json"
    atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
