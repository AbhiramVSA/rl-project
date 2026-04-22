from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#2a3342", lw=1.3):
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
        fontsize=11,
        fontweight="bold",
        va="center",
        ha="left",
        color="#0f172a",
    )
    ax.text(
        x + 0.03 * width,
        y + 0.30 * height,
        body,
        fontsize=10,
        va="center",
        ha="left",
        color="#1f2937",
        linespacing=1.35,
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

    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Group regions
    ax.add_patch(Rectangle((0.03, 0.50), 0.34, 0.26, fill=False, linestyle=(0, (4, 4)), linewidth=1.2, edgecolor="#6b7280"))
    ax.text(0.20, 0.77, "state estimation", fontsize=12, ha="center", va="bottom")

    ax.add_patch(Rectangle((0.20, 0.18), 0.56, 0.57, fill=False, linestyle=(0, (4, 4)), linewidth=1.2, edgecolor="#6b7280"))
    ax.text(0.48, 0.15, "RL controller", fontsize=12, ha="center", va="top")

    ax.add_patch(Rectangle((0.58, 0.38), 0.37, 0.52, fill=False, linestyle=(0, (4, 4)), linewidth=1.2, edgecolor="#6b7280"))
    ax.text(0.76, 0.35, "deployment loop", fontsize=12, ha="center", va="top")

    add_box(
        ax,
        (0.05, 0.54),
        0.15,
        0.11,
        "Dialogue inputs",
        "history\nCRM features\nprospect signals",
        "#f8fafc",
    )
    add_box(
        ax,
        (0.24, 0.54),
        0.16,
        0.11,
        "Belief encoder $f_\\psi$",
        "state estimate $b_t \\in \\mathbb{R}^d$",
        "#eef2ff",
    )
    add_box(
        ax,
        (0.46, 0.62),
        0.18,
        0.11,
        "Actor head $g_\\theta$",
        "$\\pi_\\theta(a_t\\mid b_t)$",
        "#fee2e2",
    )
    add_box(
        ax,
        (0.46, 0.42),
        0.20,
        0.12,
        "Quantile critic $q_{\\phi,1:N}$",
        "return distribution $Z_\\phi^\\pi(b_t)$",
        "#dcfce7",
    )
    add_box(
        ax,
        (0.71, 0.62),
        0.13,
        0.11,
        "Macro-action",
        "discover\nqualify\nclose ...",
        "#fff7ed",
    )
    add_box(
        ax,
        (0.69, 0.42),
        0.16,
        0.12,
        "Risk summary",
        "mean, variance,\nCVaR, tail risk",
        "#fff7ed",
    )
    add_box(
        ax,
        (0.88, 0.62),
        0.08,
        0.11,
        "Environment",
        "sales dialogue",
        "#f3f4f6",
    )
    add_box(
        ax,
        (0.86, 0.80),
        0.11,
        0.11,
        "Optional response layer",
        "frontier LLM\nor template generator",
        "#f8fafc",
    )
    add_box(
        ax,
        (0.86, 0.40),
        0.12,
        0.12,
        "Reward synthesis",
        "$\\Delta I_t,\\;\\Delta T_t,\\;\\Delta S_t$\n$F_t,\\;O_t,\\;y_t,\\;v_t$",
        "#eef2ff",
    )
    add_box(
        ax,
        (0.28, 0.22),
        0.18,
        0.12,
        "Optimization losses",
        "policy loss\ncritic loss\nentropy regularization",
        "#fff7ed",
    )

    add_arrow(ax, (0.20, 0.595), (0.24, 0.595))
    add_arrow(ax, (0.40, 0.595), (0.46, 0.675))
    add_arrow(ax, (0.40, 0.595), (0.46, 0.48))
    add_arrow(ax, (0.64, 0.675), (0.71, 0.675))
    add_arrow(ax, (0.66, 0.48), (0.69, 0.48))
    add_arrow(ax, (0.84, 0.675), (0.88, 0.675))
    add_arrow(ax, (0.92, 0.80), (0.92, 0.73))
    add_arrow(ax, (0.84, 0.675), (0.86, 0.675), linestyle="--")
    add_arrow(ax, (0.92, 0.62), (0.92, 0.52))
    add_arrow(ax, (0.86, 0.46), (0.46, 0.28))
    add_arrow(ax, (0.69, 0.48), (0.46, 0.28))
    add_arrow(ax, (0.37, 0.34), (0.32, 0.54))

    ax.text(0.675, 0.69, "$a_t$", fontsize=10, color="#374151")
    ax.text(0.61, 0.32, "$r_t,\\; d_t,\\; o_{t+1}$", fontsize=10, color="#374151")

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_diagram.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "architecture_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
