#!/usr/bin/env python3
"""Generate the draw.io source and SVG for the M3 coherence mechanism inset."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape


W = 640
H = 560
# ImageMagick's SVG renderer needs the exact font entry, otherwise it may
# silently pick an italic fallback. This is the regular Latin Modern face.
FONT = "LMRoman10-Regular"


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    color: str = "#111827",
    anchor: str = "middle",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="normal" fill="{color}" text-anchor="{anchor}" dominant-baseline="middle">{escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str,
    sw: float = 2.0,
    rx: float = 12.0,
    extra: str = "",
) -> str:
    extra_attr = f" {extra}" if extra else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra_attr}/>'


def path(d: str, *, color: str, width: float, marker: str | None = None) -> str:
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{marker_attr}/>'


def line(x1: float, y1: float, x2: float, y2: float, *, color: str, width: float, marker: str | None = None) -> str:
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{marker_attr}/>'


def pill(x: float, y: float, w: float, h: float, value: str, *, stroke: str, fill: str, color: str) -> str:
    return "\n".join(
        [
            rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.25, rx=h / 2.0, extra='filter="url(#labelShadow)"'),
            text(x + w / 2.0, y + h / 2.0 + 0.4, value, size=12, color=color),
        ]
    )


def badge(x: float, y: float, value: str, *, fill: str, stroke: str, color: str = "#ffffff", r: float = 11.0) -> str:
    return "\n".join(
        [
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
            text(x, y + 0.4, value, size=11, color=color, weight=600),
        ]
    )


def chip_notches(x: float, y: float, w: float, h: float, *, color: str) -> str:
    parts: list[str] = []
    for yy in (y + 32, y + 66, y + 100):
        parts.append(rect(x - 14, yy, 14, 8, fill=color, stroke=color, sw=0, rx=2, extra='opacity="0.92"'))
        parts.append(rect(x + w, yy, 14, 8, fill=color, stroke=color, sw=0, rx=2, extra='opacity="0.92"'))
    return "\n".join(parts)


def cells(x: float, y: float, *, stroke: str, fill: str, state: str) -> str:
    parts: list[str] = []
    cell_w = 22
    cell_h = 19
    gap = 4
    for i in range(5):
        cx = x + i * (cell_w + gap)
        active = i in (2, 3)
        parts.append(rect(cx, y, cell_w, cell_h, fill=fill if active else "#ffffff", stroke=stroke, sw=1.05, rx=5))
    if state == "owned":
        dot_x = x + 3.48 * (cell_w + gap)
        parts.append(f'<circle cx="{dot_x:.1f}" cy="{y + cell_h / 2:.1f}" r="10" fill="#b91c1c" stroke="#ffffff" stroke-width="2.5"/>')
    if state == "invalid":
        x0 = x + 2 * (cell_w + gap) - 2
        x1 = x + 4 * (cell_w + gap) - gap + 2
        parts.append(line(x0, y + 3, x1, y + cell_h - 3, color="#b91c1c", width=2.8))
        parts.append(line(x0, y + cell_h - 3, x1, y + 3, color="#b91c1c", width=2.8))
    return "\n".join(parts)


def core_card(
    x: float,
    y: float,
    *,
    title: str,
    action: str,
    stroke: str,
    fill: str,
    line_fill: str,
    state: str,
    state_label: str,
) -> str:
    return "\n".join(
        [
            chip_notches(x, y, 226, 136, color=stroke),
            rect(x, y, 226, 136, fill=fill, stroke=stroke, sw=2.0, rx=18, extra='filter="url(#cardShadow)"'),
            '<path d="M {0} {1} H {2}" stroke="{3}" stroke-width="1" opacity="0.22"/>'.format(x + 24, y + 45, x + 202, stroke),
            '<path d="M {0} {1} H {2}" stroke="{3}" stroke-width="1" opacity="0.12"/>'.format(x + 24, y + 124, x + 202, stroke),
            text(x + 21, y + 27, title, size=17, anchor="start", weight=500),
            pill(x + 18, y + 82, 68, 28, action, stroke=stroke, fill="#ffffff", color="#334155"),
            text(x + 96, y + 66, "L1 Line", size=11, color="#475569", anchor="start"),
            badge(x + 204, y + 64, "M" if state == "owned" else "I", fill="#b91c1c", stroke="#ffffff", r=8.5),
            cells(x + 96, y + 86, stroke=stroke, fill=line_fill, state=state),
            text(x + 162, y + 121, state_label, size=10, color="#b91c1c"),
        ]
    )


def address_card(x: float, y: float) -> str:
    return "\n".join(
        [
            rect(x, y, 286, 94, fill="#f8fafc", stroke="#94a3b8", sw=2.05, rx=18, extra='filter="url(#cardShadow)"'),
            text(x + 143, y + 30, "Cache-Line Address A", size=17, weight=500),
            rect(x + 74, y + 56, 138, 24, fill="#ffffff", stroke="#94a3b8", sw=1.35, rx=12),
            text(x + 143, y + 68, "Same 64B Address", size=11, color="#475569"),
        ]
    )


def wait_panel(x: float, y: float) -> str:
    return "\n".join(
        [
            rect(x, y, 250, 54, fill="#eff6ff", stroke="#bfdbfe", sw=1.5, rx=22, extra='filter="url(#labelShadow)"'),
            '<circle cx="{0}" cy="{1}" r="18" fill="#ffffff" stroke="#3867d6" stroke-width="2.5"/>'.format(x + 43, y + 27),
            line(x + 43, y + 27, x + 43, y + 14, color="#3867d6", width=2.6),
            line(x + 43, y + 27, x + 55, y + 35, color="#3867d6", width=2.6),
            text(x + 152, y + 31, "Reader Load Wait", size=17, color="#1d4ed8"),
        ]
    )


def write_svg(path_out: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '  <filter id="cardShadow" x="-12%" y="-12%" width="124%" height="130%"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.11"/></filter>',
        '  <filter id="labelShadow" x="-18%" y="-30%" width="136%" height="160%"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.08"/></filter>',
        '  <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b91c1c"/></marker>',
        '  <marker id="arrow-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7048e8"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        text(20, 34, "(b)", size=22, color="#000000", anchor="start", weight=600),
        rect(54, 72, 532, 414, fill="#fbfdff", stroke="#dfe7f1", sw=1.15, rx=24),
        line(110, 279, 530, 279, color="#cbd5e1", width=2.1),
        '<circle cx="176" cy="279" r="5" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>',
        '<circle cx="320" cy="279" r="5" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>',
        '<circle cx="464" cy="279" r="5" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>',
        text(320, 257, "Coherence Fabric", size=11, color="#64748b"),
        core_card(80, 104, title="Writer Core", action="Store", stroke="#d95f59", fill="#fff5f4", line_fill="#fee2e2", state="owned", state_label="Owned"),
        core_card(334, 104, title="Reader Core", action="Load", stroke="#3867d6", fill="#eff6ff", line_fill="#dbeafe", state="invalid", state_label="Invalid"),
        address_card(177, 316),
        path("M 188 240 C 197 262, 221 280, 270 316", color="#ffffff", width=7.2),
        path("M 188 240 C 197 262, 221 280, 270 316", color="#b91c1c", width=3.5, marker="arrow-red"),
        pill(112, 249, 96, 27, "RFO Request", stroke="#d95f59", fill="#fff5f4", color="#b91c1c"),
        badge(174, 236, "1", fill="#b91c1c", stroke="#ffffff", r=10),
        '<circle cx="216" cy="277" r="4" fill="#b91c1c" opacity="0.52"/>',
        '<circle cx="236" cy="292" r="4" fill="#b91c1c" opacity="0.72"/>',
        path("M 405 316 C 430 295, 451 270, 462 240", color="#ffffff", width=7.2),
        path("M 405 316 C 430 295, 451 270, 462 240", color="#7048e8", width=3.5, marker="arrow-purple"),
        pill(428, 249, 104, 27, "Snoop / HITM", stroke="#7048e8", fill="#f4f1ff", color="#5b35d5"),
        badge(468, 236, "3", fill="#7048e8", stroke="#ffffff", r=10),
        '<circle cx="431" cy="292" r="4" fill="#7048e8" opacity="0.52"/>',
        '<circle cx="416" cy="304" r="4" fill="#7048e8" opacity="0.72"/>',
        badge(308, 302, "2", fill="#64748b", stroke="#ffffff", r=10),
        text(333, 302, "Owner Changes", size=11, color="#475569", anchor="start"),
        wait_panel(196, 432),
        "</svg>",
    ]
    path_out.write_text("\n".join(parts), encoding="utf-8")


def mx_cell(cell_id: str, value: str, style: str, x: float, y: float, w: float, h: float) -> str:
    return (
        f'<mxCell id="{cell_id}" value="{escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )


def mx_edge(edge_id: str, color: str, x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<mxCell id="{edge_id}" value="" style="endArrow=block;html=1;curved=1;strokeWidth=3;strokeColor={color};" edge="1" parent="1">'
        f'<mxGeometry width="50" height="50" relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
        f'</mxGeometry></mxCell>'
    )


def mx_rect(
    cell_id: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float = 1.0,
    rounded: bool = True,
    font_size: float = 12.0,
    font_color: str = "#111827",
    align: str = "center",
) -> str:
    style = (
        f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fontFamily=Latin Modern Roman;"
        f"fontSize={font_size};fontColor={font_color};align={align};verticalAlign=middle;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
    )
    return mx_cell(cell_id, value, style, x, y, w, h)


def mx_text_cell(
    cell_id: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float,
    color: str = "#111827",
    align: str = "center",
    bold: bool = False,
) -> str:
    style = (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;"
        f"fontFamily=Latin Modern Roman;fontSize={size};fontColor={color};align={align};"
        f"fontStyle={1 if bold else 0};"
    )
    return mx_cell(cell_id, value, style, x, y, w, h)


def mx_ellipse(
    cell_id: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str,
    font_size: float = 11.0,
    font_color: str = "#ffffff",
) -> str:
    style = (
        "ellipse;whiteSpace=wrap;html=1;fontFamily=Latin Modern Roman;fontStyle=1;"
        f"fontSize={font_size};fillColor={fill};strokeColor={stroke};fontColor={font_color};"
    )
    return mx_cell(cell_id, value, style, x, y, w, h)


def mx_line(cell_id: str, color: str, x1: float, y1: float, x2: float, y2: float, *, width: float = 1.0) -> str:
    return (
        f'<mxCell id="{cell_id}" value="" style="endArrow=none;html=1;rounded=0;'
        f'strokeWidth={width};strokeColor={color};" edge="1" parent="1">'
        '<mxGeometry width="50" height="50" relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
        '</mxGeometry></mxCell>'
    )


def mx_connector(
    cell_id: str,
    color: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: float = 3.0,
    arrow: str = "block",
) -> str:
    return (
        f'<mxCell id="{cell_id}" value="" style="endArrow={arrow};html=1;curved=1;'
        f'strokeWidth={width};strokeColor={color};" edge="1" parent="1">'
        '<mxGeometry width="50" height="50" relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
        '</mxGeometry></mxCell>'
    )


def drawio_cache_cells(prefix: str, x: float, y: float, *, stroke: str, fill: str, state: str) -> list[str]:
    out: list[str] = []
    cell_w = 22
    cell_h = 19
    gap = 4
    for i in range(5):
        cx = x + i * (cell_w + gap)
        active = i in (2, 3)
        out.append(
            mx_rect(
                f"{prefix}_cell_{i}",
                "",
                cx,
                y,
                cell_w,
                cell_h,
                fill=fill if active else "#ffffff",
                stroke=stroke,
                stroke_width=1.05,
                rounded=True,
            )
        )
    if state == "owned":
        dot_x = x + 3.48 * (cell_w + gap)
        out.append(mx_ellipse(f"{prefix}_owner_dot", "", dot_x - 10, y - 0.5, 20, 20, fill="#b91c1c", stroke="#ffffff"))
    if state == "invalid":
        x0 = x + 2 * (cell_w + gap) - 2
        x1 = x + 4 * (cell_w + gap) - gap + 2
        out.append(mx_line(f"{prefix}_invalid_x1", "#b91c1c", x0, y + 3, x1, y + cell_h - 3, width=2.8))
        out.append(mx_line(f"{prefix}_invalid_x2", "#b91c1c", x0, y + cell_h - 3, x1, y + 3, width=2.8))
    return out


def drawio_core(
    prefix: str,
    x: float,
    y: float,
    *,
    title: str,
    action: str,
    stroke: str,
    fill: str,
    line_fill: str,
    state: str,
    state_label: str,
) -> list[str]:
    out: list[str] = []
    for idx, yy in enumerate((y + 32, y + 66, y + 100)):
        out.append(mx_rect(f"{prefix}_pin_l_{idx}", "", x - 14, yy, 14, 8, fill=stroke, stroke=stroke, stroke_width=0, rounded=True))
        out.append(mx_rect(f"{prefix}_pin_r_{idx}", "", x + 226, yy, 14, 8, fill=stroke, stroke=stroke, stroke_width=0, rounded=True))
    out.append(mx_rect(f"{prefix}_card", "", x, y, 226, 136, fill=fill, stroke=stroke, stroke_width=2.0, rounded=True))
    out.append(mx_line(f"{prefix}_rule_top", stroke, x + 24, y + 45, x + 202, y + 45, width=1.0))
    out.append(mx_line(f"{prefix}_rule_bot", stroke, x + 24, y + 124, x + 202, y + 124, width=1.0))
    out.append(mx_text_cell(f"{prefix}_title", title, x + 21, y + 15, 140, 26, size=17, align="left"))
    out.append(mx_rect(f"{prefix}_action", action, x + 18, y + 82, 68, 28, fill="#ffffff", stroke=stroke, stroke_width=1.25, rounded=True, font_size=12, font_color="#334155"))
    out.append(mx_text_cell(f"{prefix}_l1_label", "L1 Line", x + 96, y + 53, 68, 22, size=11, color="#475569", align="left"))
    out.append(
        mx_ellipse(
            f"{prefix}_state_badge",
            "M" if state == "owned" else "I",
            x + 195.5,
            y + 55.5,
            17,
            17,
            fill="#b91c1c",
            stroke="#ffffff",
        )
    )
    out.extend(drawio_cache_cells(f"{prefix}_l1", x + 96, y + 86, stroke=stroke, fill=line_fill, state=state))
    out.append(mx_text_cell(f"{prefix}_state_label", state_label, x + 128, y + 110, 68, 22, size=10, color="#b91c1c"))
    return out


def drawio_address_card(x: float, y: float) -> list[str]:
    out = [
        mx_rect("address_card", "", x, y, 286, 94, fill="#f8fafc", stroke="#94a3b8", stroke_width=2.05, rounded=True),
        mx_text_cell("address_title", "Cache-Line Address A", x + 50, y + 17, 186, 28, size=17),
        mx_rect("address_bar", "Same 64B Address", x + 74, y + 56, 138, 24, fill="#ffffff", stroke="#94a3b8", stroke_width=1.35, rounded=True, font_size=11, font_color="#475569"),
    ]
    return out


def write_drawio(path_out: Path) -> None:
    cells_xml = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>',
        mx_text_cell("b", "(b)", 20, 18, 45, 32, size=22, align="left", bold=True),
        mx_rect("panel", "", 54, 72, 532, 414, fill="#fbfdff", stroke="#dfe7f1", stroke_width=1.15, rounded=True),
        mx_line("fabric_bus", "#cbd5e1", 110, 279, 530, 279, width=2.1),
        mx_ellipse("fabric_node_l", "", 171, 274, 10, 10, fill="#ffffff", stroke="#94a3b8"),
        mx_ellipse("fabric_node_c", "", 315, 274, 10, 10, fill="#ffffff", stroke="#94a3b8"),
        mx_ellipse("fabric_node_r", "", 459, 274, 10, 10, fill="#ffffff", stroke="#94a3b8"),
        mx_text_cell("fabric_label", "Coherence Fabric", 260, 245, 120, 24, size=11, color="#64748b"),
    ]
    cells_xml.extend(
        drawio_core(
            "writer",
            80,
            104,
            title="Writer Core",
            action="Store",
            stroke="#d95f59",
            fill="#fff5f4",
            line_fill="#fee2e2",
            state="owned",
            state_label="Owned",
        )
    )
    cells_xml.extend(
        drawio_core(
            "reader",
            334,
            104,
            title="Reader Core",
            action="Load",
            stroke="#3867d6",
            fill="#eff6ff",
            line_fill="#dbeafe",
            state="invalid",
            state_label="Invalid",
        )
    )
    cells_xml.extend(drawio_address_card(177, 316))
    cells_xml.extend(
        [
            mx_connector("rfo_under", "#ffffff", 188, 240, 270, 316, width=7.2, arrow="none"),
            mx_connector("rfo", "#b91c1c", 188, 240, 270, 316, width=3.5, arrow="block"),
            mx_rect("rfo_text", "RFO Request", 112, 249, 96, 27, fill="#fff5f4", stroke="#d95f59", stroke_width=1.25, rounded=True, font_size=12, font_color="#b91c1c"),
            mx_ellipse("step1", "1", 164, 226, 20, 20, fill="#b91c1c", stroke="#ffffff"),
            mx_ellipse("rfo_dot_1", "", 212, 273, 8, 8, fill="#b91c1c", stroke="#b91c1c"),
            mx_ellipse("rfo_dot_2", "", 232, 288, 8, 8, fill="#b91c1c", stroke="#b91c1c"),
            mx_connector("hitm_under", "#ffffff", 405, 316, 462, 240, width=7.2, arrow="none"),
            mx_connector("hitm", "#7048e8", 405, 316, 462, 240, width=3.5, arrow="block"),
            mx_rect("hitm_text", "Snoop / HITM", 428, 249, 104, 27, fill="#f4f1ff", stroke="#7048e8", stroke_width=1.25, rounded=True, font_size=12, font_color="#5b35d5"),
            mx_ellipse("step3", "3", 458, 226, 20, 20, fill="#7048e8", stroke="#ffffff"),
            mx_ellipse("hitm_dot_1", "", 427, 288, 8, 8, fill="#7048e8", stroke="#7048e8"),
            mx_ellipse("hitm_dot_2", "", 412, 300, 8, 8, fill="#7048e8", stroke="#7048e8"),
            mx_ellipse("step2", "2", 298, 292, 20, 20, fill="#64748b", stroke="#ffffff"),
            mx_text_cell("owner_changes", "Owner Changes", 333, 290, 88, 24, size=11, color="#475569", align="left"),
            mx_rect("wait_panel", "", 196, 432, 250, 54, fill="#eff6ff", stroke="#bfdbfe", stroke_width=1.5, rounded=True),
            mx_ellipse("wait_clock", "", 221, 441, 36, 36, fill="#ffffff", stroke="#3867d6"),
            mx_line("wait_clock_hand_1", "#3867d6", 239, 459, 239, 446, width=2.6),
            mx_line("wait_clock_hand_2", "#3867d6", 239, 459, 251, 467, width=2.6),
            mx_text_cell("wait_text", "Reader Load Wait", 282, 446, 132, 28, size=17, color="#1d4ed8"),
        ]
    )
    body = "".join(cells_xml)
    drawio = (
        '<mxfile host="app.diagrams.net" modified="2026-05-07T00:00:00.000Z" agent="Codex" version="24.7.17">'
        '<diagram name="Page-1">'
        f'<mxGraphModel dx="1000" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
        f'<root>{body}</root></mxGraphModel>'
        "</diagram></mxfile>"
    )
    path_out.write_text(drawio, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    args.drawio.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    write_drawio(args.drawio)
    write_svg(args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
