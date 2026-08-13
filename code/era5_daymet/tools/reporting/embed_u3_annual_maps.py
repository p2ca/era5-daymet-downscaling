#!/usr/bin/env python
"""Embed six verified U3 annual-map PNGs into the canonical HTML report."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path


MAPS = (
    ("tmax", "field", "tmax · 2020 年均", "年均场(Truth + 各方法)"),
    ("tmax", "bias", "tmax · 2020 年均", "偏差 bias(方法 − truth)"),
    ("tmin", "field", "tmin · 2020 年均", "年均场(Truth + 各方法)"),
    ("tmin", "bias", "tmin · 2020 年均", "偏差 bias(方法 − truth)"),
    ("precip", "field", "precip (mm/day) · 2020 年均", "年均场(Truth + 各方法)"),
    ("precip", "bias", "precip (mm/day) · 2020 年均", "偏差 bias(方法 − truth)"),
)


def _block(variable: str, kind: str, image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    marker = f"U3_MAP:{variable}:{kind}"
    alt = f"U3 {variable} 2020 annual {'mean' if kind == 'field' else 'bias'}"
    return (
        f"      <!-- {marker}:start -->\n"
        f'      <img src="data:image/png;base64,{encoded}" alt="{alt}" loading="lazy">\n'
        f"      <!-- {marker}:end -->\n"
    )


def _upsert(
    html: str,
    *,
    variable: str,
    kind: str,
    heading: str,
    group_label: str,
    image_path: Path,
) -> str:
    marker = f"U3_MAP:{variable}:{kind}"
    start_marker = f"      <!-- {marker}:start -->"
    end_marker = f"      <!-- {marker}:end -->"
    replacement = _block(variable, kind, image_path)
    if start_marker in html:
        start = html.index(start_marker)
        end = html.index(end_marker, start) + len(end_marker)
        if end < len(html) and html[end] == "\n":
            end += 1
        return html[:start] + replacement + html[end:]

    heading_token = f"<h3>{heading}</h3>"
    heading_start = html.index(heading_token)
    next_heading = html.find("<h3>", heading_start + len(heading_token))
    section_end = next_heading if next_heading >= 0 else len(html)
    group_token = f'<div class="grouplab">{group_label}</div>'
    group_start = html.index(group_token, heading_start, section_end)
    grid_start = html.index('<div class="imgrid">', group_start, section_end)
    grid_end = html.index("</div>", grid_start, section_end)
    return html[:grid_end] + replacement + html[grid_end:]


def embed(report_path: Path, maps_dir: Path) -> None:
    report_path = report_path.resolve()
    maps_dir = maps_dir.resolve()
    html = report_path.read_text()
    for variable, kind, heading, group_label in MAPS:
        image_path = maps_dir / f"{variable}_{kind}_u3.png"
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing U3 report image: {image_path}")
        html = _upsert(
            html,
            variable=variable,
            kind=kind,
            heading=heading,
            group_label=group_label,
            image_path=image_path,
        )

    for variable, kind, _, _ in MAPS:
        marker = f"U3_MAP:{variable}:{kind}"
        if html.count(f"<!-- {marker}:start -->") != 1:
            raise ValueError(f"report marker count is not one: {marker}")
        if html.count(f"<!-- {marker}:end -->") != 1:
            raise ValueError(f"report marker count is not one: {marker}")

    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(html)
        os.replace(temporary, report_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[u3-report] embedded 6 full-year maps -> {report_path}", flush=True)


def main() -> None:
    project = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", default=str(project / "docs/reports/report.html")
    )
    parser.add_argument(
        "--maps-dir",
        default=str(
            project
            / "runs/exp/20260726-unet-U3-fullframe-16node-v3/annual_maps_2020"
        ),
    )
    args = parser.parse_args()
    embed(Path(args.report), Path(args.maps_dir))


if __name__ == "__main__":
    main()
