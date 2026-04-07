from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc="#ffffff",
    ec="#1f2937",
    lw=1.6,
    fontsize=10,
    bold=False,
    zorder=3,
):
    # Soft drop shadow for visual layering.
    shadow = FancyBboxPatch(
        (x + 0.005, y - 0.006),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0,
        edgecolor="none",
        facecolor="#0f172a",
        alpha=0.08,
        zorder=zorder - 1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#111827",
        wrap=True,
        zorder=zorder + 1,
    )


def add_arrow(ax, p1, p2, color="#334155", lw=1.8, style="->", rad=0.0, zorder=5):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_rect(ax, x, y, w, h, text, fc="#ffffff", ec="#1f2937", lw=1.5, fontsize=10, bold=False, zorder=4):
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#111827",
        wrap=True,
        zorder=zorder + 1,
    )


def draw_framework(output_path):
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background canvas.
    bg = FancyBboxPatch(
        (0.015, 0.02),
        0.97,
        0.95,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
    )
    ax.add_patch(bg)

    # Title
    ax.text(
        0.5,
        0.95,
        "StreamSeed Retrieval Full Workflow",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.92,
        "query -> two-tier storage -> seed retrieval -> collect candidate -> validate -> top-k result -> writeback",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
    )

    # Main modules (left-to-right pipeline)
    y = 0.50
    h = 0.15

    # Query
    add_box(
        ax,
        0.03,
        y,
        0.095,
        h,
        "Query",
        fc="#ffffff",
        ec="#334155",
        lw=1.7,
        fontsize=12,
        bold=True,
    )

    # Two-tier storage (rounded container)
    add_box(
        ax,
        0.16,
        0.38,
        0.22,
        0.39,
        "Two-Tier Storage",
        fc="#ecfeff",
        ec="#0e7490",
        lw=1.8,
        fontsize=12,
        bold=True,
    )
    add_rect(
        ax,
        0.182,
        0.57,
        0.176,
        0.13,
        "Hot-Exact\nLayer",
        fc="#cffafe",
        ec="#0e7490",
        fontsize=10,
    )
    add_rect(
        ax,
        0.182,
        0.41,
        0.176,
        0.13,
        "Semantic-Shared\nLayer",
        fc="#ccfbf1",
        ec="#0f766e",
        fontsize=10,
    )

    # Retrieval
    add_box(
        ax,
        0.43,
        y,
        0.14,
        h,
        "Seed\nRetrieval",
        fc="#dcfce7",
        ec="#166534",
        lw=1.7,
        fontsize=11,
        bold=True,
    )

    # Collect candidate
    add_box(
        ax,
        0.62,
        y,
        0.16,
        h,
        "Collect Candidate\n(Bounded Expansion)",
        fc="#bbf7d0",
        ec="#15803d",
        lw=1.7,
        fontsize=10,
        bold=True,
    )

    # Validation
    add_box(
        ax,
        0.81,
        y,
        0.14,
        h,
        "Validation\n(accept/fallback)",
        fc="#ffedd5",
        ec="#c2410c",
        lw=1.7,
        fontsize=10,
        bold=True,
    )

    # Result and writeback (second row)
    add_box(
        ax,
        0.81,
        0.22,
        0.14,
        0.13,
        "Top-k Result",
        fc="#e0e7ff",
        ec="#4338ca",
        lw=1.7,
        fontsize=11,
        bold=True,
    )
    add_box(
        ax,
        0.56,
        0.17,
        0.20,
        0.12,
        "Writeback\nretain / evict / promote / demote",
        fc="#c7d2fe",
        ec="#3730a3",
        lw=1.7,
        fontsize=10,
        bold=True,
    )

    # Main flow arrows
    add_arrow(ax, (0.125, 0.575), (0.16, 0.575), color="#334155", lw=2.0)
    add_arrow(ax, (0.38, 0.575), (0.43, 0.575), color="#0f766e", lw=2.0)
    add_arrow(ax, (0.57, 0.575), (0.62, 0.575), color="#166534", lw=2.0)
    add_arrow(ax, (0.78, 0.575), (0.81, 0.575), color="#15803d", lw=2.0)
    add_arrow(ax, (0.88, 0.50), (0.88, 0.35), color="#c2410c", lw=2.0)
    add_arrow(ax, (0.81, 0.255), (0.76, 0.23), color="#4338ca", lw=2.0)
    add_arrow(ax, (0.56, 0.23), (0.38, 0.44), color="#1d4ed8", lw=2.0, rad=0.15)

    # Labels around arrows
    ax.text(0.49, 0.61, "select seed", ha="center", va="bottom", fontsize=9.5, color="#166534")
    ax.text(0.70, 0.61, "expand from seed", ha="center", va="bottom", fontsize=9.5, color="#15803d")
    ax.text(0.885, 0.43, "accepted", ha="left", va="center", fontsize=9.5, color="#9a3412")
    ax.text(0.47, 0.355, "write back to storage", ha="center", va="center", fontsize=9.5, color="#1d4ed8")

    # Small note near validation
    ax.text(
        0.88,
        0.69,
        "Check candidate reliability\nvia quality + consistency",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#7c2d12",
    )

    # Footer caption
    ax.text(
        0.5,
        0.045,
        "Flow: query -> two-tier storage -> seed retrieval -> collect candidate -> validate -> top-k result -> writeback",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    out = root / "../figures/generated/streamseed_retrieval_full_workflow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_framework(str(out))
    print(f"Saved: {out}")
