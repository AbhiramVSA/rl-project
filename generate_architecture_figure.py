from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#2a3342", lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.03 * width,
        y + 0.68 * height,
        title,
        fontsize=9.5,
        fontweight="bold",
        va="center",
        ha="left",
        color="#0f172a",
    )
    ax.text(
        x + 0.03 * width,
        y + 0.30 * height,
        body,
        fontsize=8.5,
        va="center",
        ha="left",
        color="#1f2937",
        linespacing=1.25,
    )


def add_arrow(ax, start, end, color="#1f2937", style="-|>", lw=1.6, linestyle="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 7.6))
    ax.set_xlim(0, 1.26)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Group regions
    ax.add_patch(
        Rectangle(
            (0.04, 0.44),
            0.44,
            0.36,
            fill=False,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            edgecolor="#94a3b8",
        )
    )
    ax.text(0.26, 0.81, "State Estimation", fontsize=11, ha="center", va="bottom", color="#334155")

    ax.add_patch(
        Rectangle(
            (0.29, 0.12),
            0.61,
            0.67,
            fill=False,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            edgecolor="#94a3b8",
        )
    )
    ax.text(0.60, 0.10, "RL Controller", fontsize=11, ha="center", va="top", color="#334155")

    ax.add_patch(
        Rectangle(
            (0.77, 0.20),
            0.43,
            0.68,
            fill=False,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            edgecolor="#94a3b8",
        )
    )
    ax.text(0.985, 0.18, "Deployment Loop", fontsize=11, ha="center", va="top", color="#334155")

    add_box(
        ax,
        (0.07, 0.49),
        0.20,
        0.14,
        "Dialogue inputs",
        "history\nCRM features\nprospect signals",
        "#f8fafc",
    )
    add_box(
        ax,
        (0.31, 0.49),
        0.20,
        0.14,
        "Belief encoder $f_\\psi$",
        "state estimate $b_t \\in \\mathbb{R}^d$",
        "#eef2ff",
    )
    add_box(
        ax,
        (0.56, 0.60),
        0.24,
        0.14,
        "Actor head $g_\\theta$",
        "$\\pi_\\theta(a_t\\mid b_t)$",
        "#fee2e2",
    )
    add_box(
        ax,
        (0.56, 0.35),
        0.24,
        0.14,
        "Quantile critic $q_{\\phi,1:N}$",
        "return distribution $Z_\\phi^\\pi(b_t)$",
        "#dcfce7",
    )
    add_box(
        ax,
        (0.86, 0.60),
        0.18,
        0.14,
        "Macro-action",
        "discover\nqualify\nclose ...",
        "#fff7ed",
    )
    add_box(
        ax,
        (0.86, 0.35),
        0.18,
        0.14,
        "Risk summary",
        "mean, variance,\nCVaR, tail risk",
        "#fff7ed",
    )
    add_box(
        ax,
        (1.08, 0.60),
        0.12,
        0.14,
        "Environment",
        "sales dialogue",
        "#f3f4f6",
    )
    add_box(
        ax,
        (1.03, 0.82),
        0.18,
        0.12,
        "Optional response layer",
        "frontier LLM\nor template generator",
        "#f8fafc",
    )
    add_box(
        ax,
        (1.03, 0.34),
        0.18,
        0.14,
        "Reward synthesis",
        "$\\Delta I_t,\\;\\Delta T_t,\\;\\Delta S_t$\n$F_t,\\;O_t,\\;y_t,\\;v_t$",
        "#eef2ff",
    )
    add_box(
        ax,
        (0.35, 0.14),
        0.23,
        0.14,
        "Optimization losses",
        "policy loss\ncritic loss\nentropy regularization",
        "#fff7ed",
    )

    add_arrow(ax, (0.27, 0.56), (0.31, 0.56))
    add_arrow(ax, (0.51, 0.56), (0.56, 0.67))
    add_arrow(ax, (0.51, 0.56), (0.56, 0.42))
    add_arrow(ax, (0.80, 0.67), (0.86, 0.67))
    add_arrow(ax, (0.80, 0.42), (0.86, 0.42))
    add_arrow(ax, (1.04, 0.67), (1.08, 0.67))
    add_arrow(ax, (1.12, 0.82), (1.12, 0.74))
    add_arrow(ax, (1.04, 0.67), (1.03, 0.88), linestyle="--")
    add_arrow(ax, (1.14, 0.60), (1.14, 0.48))
    add_arrow(ax, (1.03, 0.41), (0.58, 0.21))
    add_arrow(ax, (1.03, 0.36), (0.58, 0.20))
    add_arrow(ax, (0.47, 0.28), (0.41, 0.49))

    ax.text(0.825, 0.70, "$a_t$", fontsize=9, color="#475569")
    ax.text(0.80, 0.25, "$r_t,\\; d_t,\\; o_{t+1}$", fontsize=9, color="#475569")

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_diagram.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "architecture_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
