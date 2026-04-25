from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#2a3342",
    lw: float = 1.2,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.09 * width,
        y + 0.68 * height,
        title,
        fontsize=9.8,
        fontweight="bold",
        va="center",
        ha="left",
        color="#0f172a",
    )
    ax.text(
        x + 0.09 * width,
        y + 0.30 * height,
        body,
        fontsize=8.2,
        va="center",
        ha="left",
        color="#1f2937",
        linespacing=1.25,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#1f2937",
    style: str = "-|>",
    lw: float = 1.65,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def add_elbow(
    ax,
    points: list[tuple[float, float]],
    color: str = "#1f2937",
    lw: float = 1.65,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
) -> None:
    """Draw an orthogonal connector with the arrowhead only on the final segment."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            solid_capstyle="round",
        )
    add_arrow(ax, points[-2], points[-1], color=color, lw=lw, linestyle=linestyle)


def add_panel(ax, xy: tuple[float, float], width: float, height: float, title: str) -> None:
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            linestyle=(0, (5, 5)),
            linewidth=1.1,
            edgecolor="#94a3b8",
        )
    )
    ax.text(
        x + width / 2.0,
        y + height + 0.18,
        title,
        fontsize=11.2,
        ha="center",
        va="bottom",
        color="#334155",
    )


def add_label(ax, xy: tuple[float, float], text: str) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        fontsize=9,
        ha="center",
        va="center",
        color="#334155",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "#ffffff",
            "edgecolor": "#cbd5e1",
            "linewidth": 0.6,
        },
    )


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4.45))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.45)
    ax.axis("off")

    add_panel(ax, (0.35, 0.30), 7.3, 3.55, "A. Risk-Sensitive RL Controller")
    add_panel(ax, (8.05, 0.30), 7.55, 3.55, "B. Deployment / Response Realization")

    add_box(
        ax,
        (0.75, 2.58),
        1.85,
        0.78,
        "Dialogue State",
        "history\nCRM features\nprospect signals",
        "#f8fafc",
    )
    add_box(
        ax,
        (3.05, 2.58),
        1.85,
        0.78,
        "Belief encoder $f_\\psi$",
        "latent belief\n$b_t \\in \\mathbb{R}^d$",
        "#eef2ff",
    )
    add_box(
        ax,
        (5.45, 2.58),
        1.55,
        0.78,
        "Actor head $g_\\theta$",
        "$\\pi_\\theta(a_t\\mid b_t)$",
        "#fee2e2",
    )
    add_box(
        ax,
        (5.45, 1.37),
        1.55,
        0.78,
        "Quantile critic $q_\\phi$",
        "distributional value\n$Z_\\phi^\\pi(b_t)$",
        "#dcfce7",
    )
    add_box(
        ax,
        (3.05, 0.47),
        1.85,
        0.70,
        "Optimization losses",
        "actor loss\nquantile Huber loss\nentropy bonus",
        "#fff7ed",
    )
    add_box(
        ax,
        (5.45, 0.47),
        1.55,
        0.70,
        "Risk summary",
        "mean\nCVaR$_\\alpha$\ndispersion",
        "#fff7ed",
    )
    add_box(
        ax,
        (8.55, 2.58),
        1.65,
        0.78,
        "Macro-action",
        "discover\nqualify\nclose",
        "#fff7ed",
    )
    add_box(
        ax,
        (12.65, 2.58),
        1.75,
        0.78,
        "Environment",
        "sales dialogue\nsimulator",
        "#f3f4f6",
    )
    add_box(
        ax,
        (12.65, 1.37),
        1.75,
        0.78,
        "Reward synthesis",
        "$\\Delta I_t, \\Delta T_t, \\Delta S_t$\n$F_t, O_t, y_t, v_t$",
        "#eef2ff",
    )
    add_box(
        ax,
        (10.80, 3.12),
        1.75,
        0.58,
        "Response realizer",
        "frontier LLM\nor template policy",
        "#f8fafc",
    )

    # Main decision path.
    add_arrow(ax, (2.60, 2.97), (3.05, 2.97))
    add_arrow(ax, (4.90, 2.97), (5.45, 2.97))
    add_arrow(ax, (7.00, 2.97), (8.55, 2.97))
    add_label(ax, (7.78, 3.18), "$a_t$")
    add_arrow(ax, (10.20, 2.97), (12.65, 2.97))

    # Response realization is conditioned by the macro-action, then feeds the environment.
    add_elbow(ax, [(9.38, 3.36), (9.38, 3.42), (10.80, 3.42)])
    add_arrow(ax, (11.68, 3.12), (11.68, 2.97))

    # Critic and risk path.
    add_elbow(ax, [(3.98, 2.58), (3.98, 1.76), (5.45, 1.76)])
    add_arrow(ax, (6.22, 1.37), (6.22, 1.17))
    add_arrow(ax, (5.45, 0.82), (4.90, 0.82))

    # Environment feedback has a dedicated middle lane.
    add_elbow(ax, [(13.52, 2.58), (13.52, 2.20), (13.52, 2.15)])
    add_arrow(ax, (12.65, 1.76), (7.00, 1.76))
    add_label(ax, (10.80, 1.98), "$r_t, d_t, o_{t+1}$")

    # Training update is dashed and routed back to the shared encoder lane.
    add_elbow(ax, [(3.98, 1.17), (3.98, 2.40), (3.98, 2.58)], linestyle=(0, (3, 3)))
    add_label(ax, (3.98, 2.18), "update")

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_diagram.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "architecture_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
