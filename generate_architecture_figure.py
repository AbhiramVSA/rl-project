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


def add_arrow(ax, start, end, color="#1f2937", style="-|>", lw=1.8, linestyle="-"):
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


def add_panel(ax, xy, width, height, title):
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            linestyle=(0, (4, 4)),
            linewidth=1.1,
            edgecolor="#94a3b8",
        )
    )
    ax.text(
        x + width / 2.0,
        y + height + 0.025,
        title,
        fontsize=12,
        ha="center",
        va="bottom",
        color="#334155",
    )


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 1.24)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel(ax, (0.05, 0.18), 0.70, 0.66, "A. Risk-Sensitive RL Controller")
    add_panel(ax, (0.79, 0.18), 0.40, 0.66, "B. Deployment / Response Realization")

    add_box(
        ax,
        (0.09, 0.53),
        0.18,
        0.13,
        "Dialogue State",
        "history\nCRM features\nprospect signals",
        "#f8fafc",
    )
    add_box(
        ax,
        (0.31, 0.53),
        0.20,
        0.13,
        "Belief encoder $f_\\psi$",
        "latent belief\n$b_t \\in \\mathbb{R}^d$",
        "#eef2ff",
    )
    add_box(
        ax,
        (0.56, 0.58),
        0.16,
        0.12,
        "Actor head $g_\\theta$",
        "$\\pi_\\theta(a_t\\mid b_t)$",
        "#fee2e2",
    )
    add_box(
        ax,
        (0.56, 0.36),
        0.16,
        0.12,
        "Quantile critic $q_{\\phi,1:N}$",
        "distributional value\n$Z_\\phi^\\pi(b_t)$",
        "#dcfce7",
    )
    add_box(
        ax,
        (0.33, 0.21),
        0.20,
        0.12,
        "Optimization losses",
        "policy loss\nquantile critic loss\nentropy regularization",
        "#fff7ed",
    )
    add_box(
        ax,
        (0.56, 0.20),
        0.16,
        0.13,
        "Risk summary",
        "mean\nCVaR$_\\alpha$\nuncertainty proxy",
        "#fff7ed",
    )
    add_box(
        ax,
        (0.83, 0.58),
        0.15,
        0.12,
        "Macro-action",
        "discover\nqualify\nclose",
        "#fff7ed",
    )
    add_box(
        ax,
        (1.01, 0.58),
        0.14,
        0.12,
        "Environment",
        "sales dialogue\nsimulator",
        "#f3f4f6",
    )
    add_box(
        ax,
        (0.83, 0.36),
        0.15,
        0.12,
        "Reward synthesis",
        "$\\Delta I_t,\\;\\Delta T_t$\n$\\Delta S_t,\\;F_t,\\;O_t$\n$y_t,\\;v_t$",
        "#eef2ff",
    )
    add_box(
        ax,
        (1.01, 0.78),
        0.17,
        0.11,
        "Response realizer",
        "frontier LLM\nor template policy",
        "#f8fafc",
    )

    add_arrow(ax, (0.27, 0.595), (0.31, 0.595))
    add_arrow(ax, (0.51, 0.595), (0.56, 0.64))
    add_arrow(ax, (0.51, 0.595), (0.56, 0.42))
    add_arrow(ax, (0.72, 0.64), (0.83, 0.64))
    add_arrow(ax, (0.98, 0.64), (1.01, 0.64))
    add_arrow(ax, (1.08, 0.78), (1.08, 0.70))
    add_arrow(ax, (1.08, 0.58), (1.08, 0.48))
    add_arrow(ax, (1.01, 0.42), (0.72, 0.42))
    add_arrow(ax, (0.72, 0.39), (0.72, 0.33))
    add_arrow(ax, (0.64, 0.36), (0.53, 0.27))
    add_arrow(ax, (0.58, 0.36), (0.43, 0.27))
    add_arrow(ax, (0.41, 0.33), (0.41, 0.53))

    ax.text(0.77, 0.665, "$a_t$", fontsize=9, color="#475569")
    ax.text(0.84, 0.44, "$r_t, d_t, o_{t+1}$", fontsize=9, color="#475569")
    ax.text(0.66, 0.335, "risk statistics", fontsize=8.5, color="#475569", ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_diagram.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "architecture_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
