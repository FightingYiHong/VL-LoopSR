from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    NATURE_CMAP,
    label_panel,
    method_color,
    save,
    setup,
)


def main() -> None:
    setup()
    plt.rcParams["axes.grid"] = False
    methods = ["Ours", "LLM-SR", "DSO", "gplearn", "PSE", "PySR", "ICSR"]
    rows = ["Overall", "LLMSRBench", "SRBench", "SLDBench", "SRSD"]
    recovery = np.array(
        [
            [77.3, 74.0, 66.0, 66.0, 57.0, 57.0, 30.0],
            [80.4, 77.5, 73.8, 79.6, 99.1, 50.0, 9.2],
            [83.0, 81.8, 82.0, 85.1, 84.4, 53.2, 40.0],
            [100.0, 96.1, 100.0, 100.0, 100.0, 100.0, 33.8],
            [56.7, 51.3, 49.2, 55.0, 0.0, 63.4, 31.9],
        ]
    )
    median_mse = {
        "Ours": 0.083,
        "LLM-SR": 0.141,
        "DSO": 0.197,
        "gplearn": 0.108,
        "PSE": 0.031,
        "PySR": 0.026,
        "ICSR": 0.230,
    }
    perfect_fit = {
        "Ours": 32.0,
        "LLM-SR": 25.0,
        "DSO": 18.0,
        "gplearn": 21.0,
        "PSE": 8.0,
        "PySR": 42.0,
        "ICSR": 12.0,
    }
    complexity = {"Ours": 14, "LLM-SR": 127, "DSO": 20, "gplearn": 10, "PSE": 30, "PySR": 9, "ICSR": 7}
    runtime = np.array(
        [
            [88, 100, 359, 237, 608, 4, 100],
            [91, 92, 272, 188, 102, 5, 63],
            [29, 100, 169, 100, 607, 15, 38],
            [43, 100, 362, 191, 3, 178, 51],
        ],
        dtype=float,
    )
    runtime_rows = ["LLMSRBench", "SRBench", "SLDBench", "SRSD"]

    fig = plt.figure(figsize=(9.6, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    im = _draw_heatmap(
        ax,
        recovery,
        cmap=NATURE_CMAP,
        vmin=0,
        vmax=100,
    )
    _remove_heatmap_grid(ax)
    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title("Recovery")
    for i in range(recovery.shape[0]):
        for j in range(recovery.shape[1]):
            text_color = COLOR_NEUTRAL_LIGHT if recovery[i, j] >= 72 else COLOR_NEUTRAL_DARK
            ax.text(j, i, f"{recovery[i, j]:.1f}", ha="center", va="center", fontsize=6.6, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="PASS@100 (%)")
    label_panel(ax, "a")

    sub_b = gs[0, 1].subgridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.14)
    ax = fig.add_subplot(sub_b[0, 0])
    y = np.arange(len(methods))
    colors = [method_color(m, i) for i, m in enumerate(methods)]
    mse_values = np.array([median_mse[m] for m in methods])
    ax.hlines(y, 0.02, mse_values, color=colors, lw=1.2, alpha=0.55)
    ax.scatter(mse_values, y, s=42, color=colors, edgecolor=COLOR_NEUTRAL_LIGHT, linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.set_xlabel("Median MSE")
    ax.set_xlim(0.018, 0.32)
    ax.invert_yaxis()
    ax.set_title("Numerical quality")
    label_panel(ax, "b")

    ax = fig.add_subplot(sub_b[0, 1], sharey=ax)
    fit_values = np.array([perfect_fit[m] for m in methods])
    ax.barh(y, fit_values, height=0.52, color=colors, alpha=0.72, edgecolor="none")
    for yi, value in zip(y, fit_values):
        ax.text(value + 1.1, yi, f"{value:.0f}", va="center", ha="left", fontsize=6.5, color=COLOR_NEUTRAL_DARK)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Perfect fit (%)")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_title("Exactness")

    ax = fig.add_subplot(gs[1, 0])
    for i, m in enumerate(methods):
        ax.scatter(complexity[m], recovery[0, i], s=55, color=method_color(m, i), edgecolor=COLOR_NEUTRAL_LIGHT, linewidth=0.5)
        ax.text(complexity[m] * 1.04, recovery[0, i] + 0.8, m, fontsize=6.6, color=COLOR_NEUTRAL_DARK)
    ax.set_xscale("log")
    ax.set_xlabel("Median expression complexity")
    ax.set_ylabel("PASS@100 (%)")
    ax.set_title("Accuracy-complexity")
    ax.set_ylim(25, 102)
    label_panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    runtime_clipped = np.clip(runtime, 0, 600)
    im = _draw_heatmap(
        ax,
        runtime_clipped,
        cmap=NATURE_CMAP,
        vmin=0,
        vmax=600,
    )
    _remove_heatmap_grid(ax)
    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_yticks(range(len(runtime_rows)), runtime_rows)
    ax.set_title("Average runtime across benchmarks")
    for i in range(runtime.shape[0]):
        for j in range(runtime.shape[1]):
            value = runtime[i, j]
            text_color = COLOR_NEUTRAL_LIGHT if runtime_clipped[i, j] >= 420 else COLOR_NEUTRAL_DARK
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="runtime (s)")
    label_panel(ax, "d")

    fig.suptitle("Standard formula recovery", y=0.99, fontsize=10.5)
    save(fig, "Fig3_standard_formula_recovery")


def _remove_heatmap_grid(ax) -> None:
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    ax.minorticks_off()
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_visible(False)


def _draw_heatmap(ax, data, *, cmap, vmin, vmax):
    rows, cols = data.shape
    norm = Normalize(vmin=vmin, vmax=vmax)
    overlap = 0.015
    for i in range(rows):
        for j in range(cols):
            color = cmap(norm(data[i, j]))
            ax.add_patch(
                Rectangle(
                    (j - 0.5 - overlap / 2, i - 0.5 - overlap / 2),
                    1.0 + overlap,
                    1.0 + overlap,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                    antialiased=False,
                    clip_on=True,
                    zorder=0,
                )
            )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("auto")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(data)
    return mappable


if __name__ == "__main__":
    main()
