from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    NATURE_COLORS,
    label_panel,
    method_color,
    save,
    setup,
)


def main() -> None:
    setup()
    methods = ["Ours", "PySR", "PyOperon", "LLM-SR", "gplearn", "PhySO", "DSO", "PSE"]
    mean_mse = np.array([0.1557, 0.2164, 0.2386, 0.4911, 0.6475, 0.7958, 1.2160, 1.3750])
    recall = np.array([0.6954, 0.3930, 0.4375, 0.5377, 0.1497, 0.0000, 0.0299, 0.0000])
    fdr = np.array([0.4077, 0.4278, 0.3541, 0.9604, 0.4907, 0.1921, 0.9520, 0.8843])
    runtime = np.array([11.75, 96.15, 95.06, 553.3, 70.42, 240.3, 7.65, 22.20])
    valid = np.array([216, 216, 216, 216, 216, 177, 216, 216])

    fig = plt.figure(figsize=(9.8, 6.1))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.42)
    y = np.arange(len(methods))
    colors = [method_color(m, i) for i, m in enumerate(methods)]

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y, mean_mse, color=colors, alpha=0.88, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("MSE on relevant-variable reconstruction")
    ax.set_title("Prediction error under distractors")
    _clean_barh_grid(ax)
    for yi, value in zip(y, mean_mse):
        ax.text(value * 1.08, yi, f"{value:.2f}", va="center", ha="left", fontsize=6.8, color=COLOR_NEUTRAL_DARK)
    label_panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    h = 0.32
    ax.barh(y - h / 2, recall * 100, height=h, color=NATURE_COLORS["blue"], alpha=0.88, label="Recall")
    ax.barh(y + h / 2, fdr * 100, height=h, color=NATURE_COLORS["vermillion"], alpha=0.78, label="FDR")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Variable selection metric (%)")
    ax.set_title("Relevant-variable recovery")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.50), borderaxespad=0.0)
    _clean_barh_grid(ax)
    label_panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    dims = np.array([200, 500, 1000])
    trend = {
        "Ours": [0.16, 0.11, 0.12],
        "PySR": [0.09, 0.18, 0.31],
        "PyOperon": [0.14, 0.18, 0.31],
        "LLM-SR": [0.49, 0.50, 0.51],
        "gplearn": [0.65, 0.70, 0.75],
        "PhySO": [0.80, 1.25, 1.05],
        "DSO": [1.22, 1.27, 1.27],
        "PSE": [1.38, 1.75, 1.08],
    }
    for i, (method, values) in enumerate(trend.items()):
        ax.plot(dims, values, marker="o", ms=3.5, lw=1.2, color=method_color(method, i), label=method)
    _label_line_ends(ax, dims, trend)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(170, 1550)
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Ambient dimension")
    ax.set_ylabel("Mean MSE")
    ax.set_title("Scaling across ambient dimensions")
    label_panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    ax.barh(y, runtime, color=colors, alpha=0.86, height=0.62)
    ax.scatter(runtime, y, s=np.clip(valid / 216 * 46, 18, 46), color=COLOR_NEUTRAL_LIGHT, edgecolor=colors, linewidth=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Runtime per case (s)")
    ax.set_title("Search cost")
    _clean_barh_grid(ax, xgrid=False)
    for yi, value in zip(y, runtime):
        ax.text(value * 1.08, yi, f"{value:.1f}", va="center", ha="left", fontsize=6.8, color=COLOR_NEUTRAL_DARK)
    label_panel(ax, "d")

    fig.suptitle("High-dimensional distractor robustness", y=0.99, fontsize=10.5)
    save(fig, "Fig5_high_dimensional_distractors")


def _clean_barh_grid(ax, *, xgrid: bool = True) -> None:
    ax.set_axisbelow(True)
    ax.grid(False, axis="y", which="both")
    ax.yaxis.grid(False, which="both")
    ax.grid(False, axis="x", which="both")
    ax.xaxis.grid(False, which="both")
    for line in ax.get_ygridlines():
        line.set_visible(False)


def _label_line_ends(ax, dims, trend) -> None:
    y_offsets = {
        "Ours": 0.85,
        "PySR": 0.95,
        "PyOperon": 1.10,
        "LLM-SR": 0.95,
        "gplearn": 0.96,
        "PhySO": 1.02,
        "DSO": 1.15,
        "PSE": 0.88,
    }
    for i, (method, values) in enumerate(trend.items()):
        ax.text(
            dims[-1] * 1.02,
            values[-1] * y_offsets.get(method, 1.0),
            method,
            color=method_color(method, i),
            fontsize=6.8,
            va="center",
            ha="left",
        )


if __name__ == "__main__":
    main()
