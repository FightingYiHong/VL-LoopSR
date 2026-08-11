from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    NATURE_COLORS,
    NEGATIVE,
    POSITIVE,
    label_panel,
    save,
    setup,
)

VARIANT_COLORS = [NATURE_COLORS["blue"], POSITIVE, NATURE_COLORS["sky"], COLOR_NEUTRAL_MID]


def main() -> None:
    setup()
    variants = ["Full", "w/o Observer", "w/o Critic", "w/o Proposer"]
    median_mse = np.array([0.3012, 0.2949, 0.7268, np.nan])
    pass_lt1 = np.array([68, 67, 50, 0]) / 96 * 100
    recall = np.array([0.4852, 0.4193, 0.1858, 0.0]) * 100
    proxy_misuse = np.array([76, 81, 66, np.nan]) / 96 * 100
    valid = np.array([96, 96, 96, 0]) / 96 * 100

    fig = plt.figure(figsize=(9.8, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.40)
    x = np.arange(len(variants))

    ax = fig.add_subplot(gs[0, 0])
    width = 0.18
    metrics = [
        ("Valid", valid, NATURE_COLORS["blue"]),
        ("MSE<1", pass_lt1, POSITIVE),
        ("Recall", recall, NATURE_COLORS["sky"]),
        ("Proxy misuse", proxy_misuse, NEGATIVE),
    ]
    for i, (name, values, color) in enumerate(metrics):
        xpos = x + (i - 1.5) * width
        ax.bar(xpos, np.nan_to_num(values), width, color=color, alpha=0.84, label=name)
    ax.set_xticks(x, variants, rotation=0, ha="center")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Cases (%)")
    ax.set_title("Component-level outcome metrics")
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.34),
        handlelength=1.4,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    _clean_axes(ax)
    label_panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    failure_modes = ["Usable", "Proxy", "Missed", "High MSE", "No expr."]
    failure = np.array(
        [
            [18, 28, 22, 28, 0],
            [10, 35, 22, 29, 0],
            [10, 9, 31, 46, 0],
            [0, 0, 0, 0, 96],
        ],
        dtype=float,
    )
    stack_colors = [
        POSITIVE,
        NEGATIVE,
        NATURE_COLORS["orange"],
        COLOR_NEUTRAL_MID,
        COLOR_NEUTRAL_LIGHT,
    ]
    bottom = np.zeros(len(variants))
    for i, mode in enumerate(failure_modes):
        ax.bar(x, failure[:, i], bottom=bottom, color=stack_colors[i], edgecolor="none", linewidth=0, label=mode)
        bottom += failure[:, i]
    ax.set_xticks(x, variants, rotation=0, ha="center")
    ax.set_ylim(0, 128)
    ax.set_ylabel("Cases")
    ax.set_title("Failure-mode decomposition")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.50), handlelength=1.6, borderaxespad=0.0)
    _clean_axes(ax)
    label_panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    rounds = np.arange(1, 7)
    trajectory = {
        "Full": [50, 65, 68, 69, 69, 69],
        "w/o Observer": [50, 62, 64, 64, 65, 65],
        "w/o Critic": [50, 50, 50, 50, 50, 50],
        "w/o Proposer": [0, 0, 0, 0, 0, 0],
    }
    colors = VARIANT_COLORS
    for i, variant in enumerate(variants):
        ax.plot(rounds, trajectory[variant], marker="o", ms=3.6, lw=1.25, color=colors[i], label=variant)
    _label_trajectory_ends(ax, rounds, trajectory, variants, colors)
    ax.set_xlabel("Refinement round")
    ax.set_ylabel("MSE<1 cases")
    ax.set_ylim(-4, 76)
    ax.set_xlim(0.8, 6.72)
    ax.set_title("Multi-step loop contribution")
    _clean_axes(ax)
    label_panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    safe_median = np.nan_to_num(median_mse, nan=3.0)
    ax.bar(x, safe_median, color=VARIANT_COLORS, alpha=0.86)
    ax.set_yscale("log")
    ax.set_xticks(x, variants, rotation=0, ha="center")
    ax.set_ylabel("Median test MSE")
    ax.set_title("Numerical degradation")
    ax.text(x[-1], safe_median[-1] * 0.82, "No valid\nexpr.", ha="center", va="top", fontsize=6.8, color=COLOR_NEUTRAL_DARK)
    for xi, value in zip(x[:-1], median_mse[:-1]):
        ax.text(xi, value * 1.16, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8, color=COLOR_NEUTRAL_DARK)
    _clean_axes(ax)
    label_panel(ax, "d")

    fig.suptitle("Ablation of the observe-critique-propose loop", y=0.99, fontsize=10.5)
    save(fig, "Fig8_component_ablation")


def _clean_axes(ax) -> None:
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")


def _label_trajectory_ends(ax, rounds, trajectory, variants, colors) -> None:
    offsets = {
        "Full": 1.00,
        "w/o Observer": 0.96,
        "w/o Critic": 1.00,
        "w/o Proposer": 1.00,
    }
    for variant, color in zip(variants, colors):
        ax.text(
            rounds[-1] + 0.12,
            trajectory[variant][-1] * offsets.get(variant, 1.0),
            variant,
            color=color,
            fontsize=6.8,
            va="center",
            ha="left",
            clip_on=False,
        )


if __name__ == "__main__":
    main()
