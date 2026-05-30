#!/usr/bin/env python3
"""Render the simple draw.io shapes used by the M3 coherence figure to SVG."""

from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_style(style: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (style or "").split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
        elif item:
            out[item] = "1"
    return out


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>\s*<div[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def inline_font_size(value: str | None) -> float | None:
    match = re.search(r"font-size:\s*([0-9.]+)px", value or "")
    return float(match.group(1)) if match else None


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def color(value: str | None, default: str = "none") -> str:
    if not value or value == "none":
        return default
    return value


def num(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def geometry(cell: ET.Element) -> tuple[float, float, float, float] | None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return None
    return (
        num(geom.get("x")),
        num(geom.get("y")),
        num(geom.get("width"), 0.0),
        num(geom.get("height"), 0.0),
    )


def text_anchor(align: str | None) -> tuple[str, float]:
    if align == "left":
        return "start", 0.04
    if align == "right":
        return "end", 0.96
    return "middle", 0.5


def emit_text(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    style: dict[str, str],
    raw_value: str | None,
    font_override: str | None,
) -> None:
    if not text:
        return
    font_size = inline_font_size(raw_value) or num(style.get("fontSize"), 12.0)
    font_family = font_override or style.get("fontFamily", "Latin Modern Roman")
    font_color = color(style.get("fontColor"), "#111827")
    anchor, x_factor = text_anchor(style.get("align"))
    base_x = x + w * x_factor
    lines = text.splitlines() or [text]
    line_h = font_size * 1.12
    start_y = y + h / 2.0 - line_h * (len(lines) - 1) / 2.0 + font_size * 0.34
    weight = "600" if "1" in (style.get("fontStyle") or "") else "400"
    parts.append(
        f'<text x="{base_x:.2f}" y="{start_y:.2f}" text-anchor="{anchor}" '
        f'font-family="{esc(font_family)}" font-size="{font_size:.2f}" '
        f'font-weight="{weight}" fill="{esc(font_color)}">'
    )
    for idx, line in enumerate(lines):
        dy = 0.0 if idx == 0 else line_h
        parts.append(f'<tspan x="{base_x:.2f}" dy="{dy:.2f}">{esc(line)}</tspan>')
    parts.append("</text>")


def emit_clock(parts: list[str], x: float, y: float, w: float, h: float, style: dict[str, str]) -> None:
    stroke = color(style.get("strokeColor"), "#3867d6")
    sw = num(style.get("strokeWidth"), 1.2)
    cx = x + w / 2.0
    cy = y + h / 2.0
    r = min(w, h) * 0.42
    parts.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="white" '
        f'stroke="{esc(stroke)}" stroke-width="{sw:.2f}"/>'
    )
    parts.append(
        f'<path d="M {cx:.2f} {cy:.2f} L {cx:.2f} {cy-r*0.55:.2f} '
        f'M {cx:.2f} {cy:.2f} L {cx+r*0.50:.2f} {cy+r*0.35:.2f}" '
        f'fill="none" stroke="{esc(stroke)}" stroke-width="{max(sw*0.75, 0.8):.2f}" '
        f'stroke-linecap="round"/>'
    )


def emit_vertex(parts: list[str], cell: ET.Element, font_override: str | None) -> None:
    geom = geometry(cell)
    if geom is None:
        return
    x, y, w, h = geom
    style = parse_style(cell.get("style"))
    raw_value = cell.get("value")
    value = clean_text(raw_value)

    if "clock" in style.get("shape", ""):
        emit_clock(parts, x, y, w, h, style)
        return

    is_text = "text" in style
    if not is_text:
        fill = color(style.get("fillColor"), "none")
        stroke = color(style.get("strokeColor"), "none")
        sw = num(style.get("strokeWidth"), 1.0)
        if "ellipse" in style:
            parts.append(
                f'<ellipse cx="{x+w/2:.2f}" cy="{y+h/2:.2f}" rx="{w/2:.2f}" ry="{h/2:.2f}" '
                f'fill="{esc(fill)}" stroke="{esc(stroke)}" stroke-width="{sw:.2f}"/>'
            )
        else:
            rx = min(10.0, h / 2.0) if style.get("rounded") == "1" else 0.0
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
                f'fill="{esc(fill)}" stroke="{esc(stroke)}" stroke-width="{sw:.2f}"/>'
            )
    emit_text(parts, x, y, w, h, value, style, raw_value, font_override)


def marker_id(stroke: str) -> str:
    return "arrow_" + re.sub(r"[^0-9A-Za-z]+", "_", stroke).strip("_")


def edge_points(cell: ET.Element, centers: dict[str, tuple[float, float]]) -> tuple[float, float, float, float] | None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return None
    points = {pt.get("as"): pt for pt in geom.findall("mxPoint")}
    source = points.get("sourcePoint")
    target = points.get("targetPoint")
    if source is not None and target is not None:
        return num(source.get("x")), num(source.get("y")), num(target.get("x")), num(target.get("y"))
    source_ref = cell.get("source")
    target_ref = cell.get("target")
    if source_ref in centers and target_ref in centers:
        x1, y1 = centers[source_ref]
        x2, y2 = centers[target_ref]
        return x1, y1, x2, y2
    return None


def emit_edge(
    parts: list[str],
    defs: dict[str, str],
    cell: ET.Element,
    centers: dict[str, tuple[float, float]],
) -> None:
    pts = edge_points(cell, centers)
    if pts is None:
        return
    x1, y1, x2, y2 = pts
    style = parse_style(cell.get("style"))
    stroke = color(style.get("strokeColor"), "#111827")
    sw = num(style.get("strokeWidth"), 1.0)
    marker = ""
    if style.get("endArrow") and style.get("endArrow") != "none":
        mid = marker_id(stroke)
        marker = f' marker-end="url(#{mid})"'
        defs.setdefault(
            mid,
            f'<marker id="{mid}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<path d="M 0 0 L 9 3.5 L 0 7 z" fill="{esc(stroke)}"/></marker>',
        )
    if style.get("curved") == "1":
        dy = abs(y2 - y1)
        cx1, cy1 = x1, y1 + dy * 0.52
        cx2, cy2 = x2, y2 - dy * 0.52
        path = f"M {x1:.2f} {y1:.2f} C {cx1:.2f} {cy1:.2f}, {cx2:.2f} {cy2:.2f}, {x2:.2f} {y2:.2f}"
    else:
        path = f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
    parts.append(
        f'<path d="{path}" fill="none" stroke="{esc(stroke)}" stroke-width="{sw:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker}/>'
    )


def render(infile: Path, outfile: Path, font_override: str | None = None) -> None:
    root = ET.parse(infile).getroot()
    model = root.find(".//mxGraphModel")
    if model is None:
        raise ValueError(f"No mxGraphModel in {infile}")
    width = int(num(model.get("pageWidth"), 640))
    height = int(num(model.get("pageHeight"), 560))
    cells = root.findall(".//mxCell")

    centers: dict[str, tuple[float, float]] = {}
    for cell in cells:
        if cell.get("vertex") == "1":
            geom = geometry(cell)
            if geom is None:
                continue
            x, y, w, h = geom
            centers[cell.get("id", "")] = (x + w / 2.0, y + h / 2.0)

    defs: dict[str, str] = {}
    body: list[str] = []
    for cell in cells:
        if cell.get("edge") == "1":
            emit_edge(body, defs, cell, centers)
        elif cell.get("vertex") == "1":
            emit_vertex(body, cell, font_override)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if defs:
        svg.append("<defs>")
        svg.extend(defs.values())
        svg.append("</defs>")
    svg.extend(body)
    svg.append("</svg>")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--font-family")
    args = parser.parse_args()
    render(args.infile, args.outfile, args.font_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
