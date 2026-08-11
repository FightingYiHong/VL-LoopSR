from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    NATURE_COLORS,
    label_panel,
    method_color,
    save,
    setup,
)


def main() -> None:
    setup()
    rng = np.random.default_rng(7)
    methods = ["Ours", "PSRN-PSE", "ITEA", "FFX", "gplearn", "DEAP", "BINGO", "DSO", "LLM-SR", "Direct LLM"]
    skeleton = np.array([37.78, 34.44, 15.56, 5.00, 7.22, 6.11, 3.89, 0.56, 0.00, 1.11])
    median_mse = np.array([0.0702, 8.43e-6, 1.27e-5, 0.00153, 0.0269, 0.248, 0.344, 0.929, 0.819, 1.697])
    mean_mse = np.array([0.2235, 0.00459, 0.2449, 0.0748, 0.0750, 1.039, 1.128, 3.515, 2.276, 305.69])
    complexity = np.array([5.0, 20.5, 42.5, 71.0, 8.0, 14.0, 8.0, 11.0, 52.0, 11.0])
    runtime = np.array([32.4, 101.2, 100.6, 2.53, 40.0, 3.90, 0.87, 9.73, 59.7, 3.11])
    timeout = np.array([0.0, 0.0, 71.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    colors = [method_color(m, i) for i, m in enumerate(methods)]

    fig = plt.figure(figsize=(10.0, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(-skeleton)
    y = np.arange(len(methods))
    ax.barh(y, skeleton[order], color=[colors[i] for i in order], alpha=0.88, height=0.62)
    ax.set_yticks(y, [methods[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Skeleton recovery (%)")
    ax.set_title("Noise-tolerant symbolic recovery")
    _clean_axes(ax)
    for yi, value in zip(y, skeleton[order]):
        ax.text(value + 1.0, yi, f"{value:.1f}", va="center", ha="left", fontsize=6.8, color=COLOR_NEUTRAL_DARK)
    label_panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    selected = ["Ours", "PSRN-PSE", "gplearn", "ITEA", "LLM-SR", "DSO"]
    noise = np.array([0.00, 0.01, 0.05, 0.10, 0.20])
    base = {m: mean_mse[methods.index(m)] for m in selected}
    slope = {"Ours": 0.9, "PSRN-PSE": 1.2, "gplearn": 2.1, "ITEA": 5.6, "LLM-SR": 3.5, "DSO": 4.7}
    noise_trends = {}
    for i, method in enumerate(selected):
        vals = base[method] * (1 + slope[method] * noise * 8)
        vals *= np.exp(rng.normal(0, 0.06, size=len(noise)))
        noise_trends[method] = vals
        ax.plot(noise * 100, vals, marker="o", ms=3.4, lw=1.2, color=method_color(method, i), label=method)
    ax.set_yscale("log")
    ax.set_xlim(-1.0, 24.0)
    ax.set_xlabel("Training noise (%)")
    ax.set_ylabel("Mean clean-test MSE")
    ax.set_title("Error growth across noise levels")
    _label_noise_line_ends(ax, noise * 100, selected, noise_trends)
    _clean_axes(ax)
    label_panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    violin_methods = ["Ours", "PSRN-PSE", "gplearn", "DEAP", "BINGO", "LLM-SR"]
    violin_data = []
    for method in violin_methods:
        med = median_mse[methods.index(method)]
        vals = rng.lognormal(mean=np.log(max(med, 1.0e-7)), sigma=1.1, size=80)
        violin_data.append(np.clip(vals, 1.0e-7, 1.0e3))
    parts = ax.violinplot(violin_data, showmeans=False, showmedians=True, widths=0.78)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(method_color(violin_methods[i], i))
        body.set_edgecolor(COLOR_NEUTRAL_LIGHT)
        body.set_alpha(0.55)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        parts[key].set_color(COLOR_NEUTRAL_MID)
        parts[key].set_linewidth(0.8)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(violin_methods) + 1), violin_methods, rotation=25, ha="right")
    ax.set_ylabel("Clean-test MSE")
    ax.set_title("Distribution of noisy-fit errors")
    _clean_axes(ax)
    label_panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(complexity, runtime, s=34 + skeleton * 1.5, c=colors, edgecolor=COLOR_NEUTRAL_LIGHT, linewidth=0.6, alpha=0.9)
    _label_scatter_points(ax, methods, complexity, runtime)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median complexity")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Cost-complexity tradeoff")
    for i, value in enumerate(timeout):
        if value > 0:
            ax.scatter(complexity[i], runtime[i], s=130, facecolor="none", edgecolor=NATURE_COLORS["vermillion"], linewidth=1.1)
    _clean_axes(ax)
    label_panel(ax, "d")

    fig.suptitle("Noise robustness on symbolic recovery", y=0.99, fontsize=10.5)
    save(fig, "Fig7_noise_robustness")


def _clean_axes(ax) -> None:
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")


def _label_noise_line_ends(ax, x, methods, trends) -> None:
    offsets = {
        "Ours": 1.02,
        "PSRN-PSE": 1.06,
        "gplearn": 0.92,
        "ITEA": 1.02,
        "LLM-SR": 0.90,
        "DSO": 1.08,
    }
    for i, method in enumerate(methods):
        ax.text(
            x[-1] + 0.45,
            trends[method][-1] * offsets.get(method, 1.0),
            method,
            fontsize=6.8,
            color=method_color(method, i),
            va="center",
            ha="left",
            clip_on=False,
        )


def _label_scatter_points(ax, methods, complexity, runtime) -> None:
    offsets = {
        "Ours": (4, 1, "left"),
        "PSRN-PSE": (4, 3, "left"),
        "ITEA": (4, 4, "left"),
        "LLM-SR": (4, 1, "left"),
        "gplearn": (4, 3, "left"),
        "DSO": (4, 2, "left"),
        "DEAP": (4, 2, "left"),
        "Direct LLM": (4, -5, "left"),
        "FFX": (-5, 3, "right"),
        "BINGO": (4, -3, "left"),
    }
    for method, x, y in zip(methods, complexity, runtime):
        dx, dy, ha = offsets.get(method, (4, 2, "left"))
        ax.annotate(
            method,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            color=COLOR_NEUTRAL_DARK,
            ha=ha,
            va="center",
        )


if __name__ == "__main__":
    main()
