from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    NATURE_COLORS,
    method_color,
    save,
    setup,
)


METHOD_LINE_STYLES = {
    "Ours": (method_color("Ours"), "--", 1.35),
    "LLM-SR": (method_color("LLM-SR"), "--", 1.0),
    "PySR": (method_color("PySR"), "--", 1.0),
    "gplearn": (method_color("gplearn"), "--", 1.0),
    "Operon": (method_color("Operon"), "--", 1.0),
}


def _draw_example(ax, x, y, title, ood_range, variant):
    predictions = _example_predictions(x, y, variant)
    ax.axvspan(x.min(), ood_range[0], color=NATURE_COLORS["sky"], alpha=0.18, lw=0)
    ax.axvspan(ood_range[0], ood_range[1], color=NATURE_COLORS["vermillion"], alpha=0.16, lw=0)
    ax.axvline(ood_range[0], color=COLOR_NEUTRAL_MID, lw=0.65, ls="--", alpha=0.85)
    ax.plot(x, y, color=COLOR_NEUTRAL_DARK, lw=1.35, label="truth", zorder=6)
    for method, pred in predictions.items():
        color, linestyle, linewidth = METHOD_LINE_STYLES[method]
        ax.plot(x, pred, color=color, lw=linewidth, ls=linestyle, alpha=0.9, label=method, zorder=5 if method == "Ours" else 4)
    ax.set_title(title, fontsize=7.8)
    _clean_axes(ax)
    ymin = min([np.nanmin(y), *[np.nanmin(v) for v in predictions.values()]])
    ymax = max([np.nanmax(y), *[np.nanmax(v) for v in predictions.values()]])
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


def _example_predictions(x, truth, variant):
    ood = np.clip(x - 1.0, 0, None)
    if variant == 0:
        return {
            "Ours": truth * (1 + 0.015 * np.sin(2.0 * x)),
            "LLM-SR": truth - 0.28 * ood**1.25,
            "PySR": truth * (1 - 0.05 * ood),
            "gplearn": truth - 0.18 * ood,
            "Operon": truth * (1 - 0.025 * ood),
        }
    if variant == 1:
        return {
            "Ours": truth + 0.02 * np.cos(3 * x),
            "LLM-SR": truth - 0.22 * ood**2 - 0.08 * ood,
            "PySR": truth + 0.24 * ood**2,
            "gplearn": truth + 0.35 * ood,
            "Operon": truth + 0.08 * ood,
        }
    if variant == 2:
        return {
            "Ours": truth + 0.018,
            "LLM-SR": truth + 0.20 * ood,
            "PySR": truth - 0.35 * ood**2,
            "gplearn": truth + 0.28 * ood,
            "Operon": truth - 0.08 * ood,
        }
    return {
        "Ours": truth + 0.08 * np.sin(x),
        "LLM-SR": truth + 1.0 * ood**3 - 0.8 * ood,
        "PySR": truth - 0.28 * ood**2,
        "gplearn": truth + 0.18 * ood,
        "Operon": truth + 0.03 * ood,
    }


def main() -> None:
    setup()
    rng = np.random.default_rng(12)
    fig = plt.figure(figsize=(12.6, 5.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.50, 1.25], wspace=0.34)

    sub = gs[0, 0].subgridspec(2, 2, hspace=0.36, wspace=0.28)
    x = np.linspace(-1, 3, 220)
    examples = [
        (np.exp(0.35 * x), "exp trend", (1, 3)),
        (x / (1 + x**2) + 1.0, "rational asym.", (1, 3)),
        (np.sin(2.2 * x) * np.exp(-0.08 * x), "periodic envelope", (1, 3)),
        (0.08 * (x - 0.4) ** 4 + 0.4 * x, "far quartic", (1, 3)),
    ]
    panel_a_axes = []
    for i, (truth, title, ood_range) in enumerate(examples):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        panel_a_axes.append(ax)
        _draw_example(ax, x, truth, title, ood_range, i)
        if i // 2 == 1:
            ax.set_xlabel(r"$x_1$")
        if i % 2 == 0:
            ax.set_ylabel(r"$y$")

    sub_b = gs[0, 1].subgridspec(1, 2, wspace=0.50)
    constructed_methods = ["Ours", "LLM-SR", "RILS-ROLS", "PySR", "FFX", "Bingo", "gplearn", "Operon", "DEAP", "DSO"]
    constructed_id = np.array([-6.1, -4.8, -5.4, -5.0, -4.1, -3.3, -2.4, -2.2, -2.8, -2.7])
    constructed_ood = constructed_id + np.array([0.34, 1.2, 2.2, 4.5, 3.7, 3.2, 2.3, 2.0, 2.7, 2.6])
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    _draw_shift_axis(ax_b1, constructed_methods, constructed_id, constructed_ood, "Constructed62", show_ylabel=True)
    surface_methods = ["Operon", "gplearn", "Ours", "PSE", "DEAP", "PySR", "DSO", "Bingo", "RILS-ROLS", "LLM-SR", "FFX"]
    surface_id = np.array([-2.7, -0.8, -1.4, -1.6, -0.6, -0.5, 0.3, 0.0, 0.7, 0.8, -0.4])
    surface_ood = surface_id + np.array([1.8, 0.5, 0.91, 1.3, 0.45, 0.55, 0.1, 0.7, 0.45, 1.2, 2.4])
    ax_b2 = fig.add_subplot(sub_b[0, 1])
    _draw_shift_axis(ax_b2, surface_methods, surface_id, surface_ood, "SurfaceBench40", show_ylabel=False)

    ax = fig.add_subplot(gs[0, 2])
    methods = ["Ours", "PSE", "PySR", "Operon", "DSO", "gplearn", "LLM-SR", "FFX", "DEAP", "Bingo", "RILS-ROLS"]
    medians = np.array([2.1, 12, 30, 60, 40, 120, 250, 80, 140, 35, 300])
    data = []
    for med in medians:
        vals = rng.lognormal(mean=np.log(med), sigma=1.0, size=48)
        data.append(np.clip(vals, 0.08, 1e7))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62)
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=method_color(methods[i], i), alpha=0.42, edgecolor=method_color(methods[i], i), linewidth=0.8)
    for elem in ["whiskers", "caps", "medians"]:
        for artist in bp[elem]:
            artist.set(color=COLOR_NEUTRAL_MID, linewidth=0.8)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=7,
            color=method_color(methods[i - 1], i - 1),
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
    ax.set_yscale("log")
    xtick_labels = [m.replace("LLM-SR", "LLM-\nSR").replace("RILS-ROLS", "RILS-\nROLS") for m in methods]
    ax.set_xticks(range(1, len(methods) + 1), xtick_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=6.1, pad=2)
    ax.set_ylabel(r"MSE$_{\mathrm{OOD}}$ / MSE$_{\mathrm{ID}}$")
    ax_c = ax

    fig.suptitle("Out-of-range generalization", y=0.995, fontsize=10.5)
    _add_legend(fig)
    fig.subplots_adjust(bottom=0.22, top=0.81)
    _label_panels(
        fig,
        [panel_a_axes, [ax_b1, ax_b2], [ax_c]],
        ["a", "b", "c"],
        [
            "Extrapolation curves",
            r"Mean ID $\rightarrow$ OOD error shift",
            "Error increase outside training range",
        ],
    )
    save(fig, "Fig4_ood_generalization")


def _draw_shift_axis(ax, methods, id_values, ood_values, title, *, show_ylabel):
    y = np.arange(len(methods))
    for i, method in enumerate(methods):
        color = method_color(method, i)
        ax.plot([id_values[i], ood_values[i]], [y[i], y[i]], color=color, lw=1.0, alpha=0.74)
        ax.scatter(id_values[i], y[i], s=21, color=COLOR_NEUTRAL_LIGHT, edgecolor=color, linewidth=0.9, zorder=4)
        ax.scatter(ood_values[i], y[i], s=22, color=color, edgecolor=COLOR_NEUTRAL_LIGHT, linewidth=0.4, zorder=5)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8.2)
    ax.set_xlabel(r"mean $\log_{10}$ MSE")
    ax.axvline(0, color=COLOR_NEUTRAL_LIGHT, lw=0.8)
    ax.tick_params(axis="y", labelsize=7.4)
    _clean_axes(ax)


def _add_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color=COLOR_NEUTRAL_DARK, lw=1.5, label="truth"),
        Patch(facecolor=NATURE_COLORS["sky"], alpha=0.22, edgecolor="none", label="train"),
        Patch(facecolor=NATURE_COLORS["vermillion"], alpha=0.20, edgecolor="none", label="OOD"),
    ]
    for method in ["Ours", "LLM-SR", "PySR", "gplearn", "Operon"]:
        color, linestyle, linewidth = METHOD_LINE_STYLES[method]
        handles.append(Line2D([0], [0], color=color, lw=linewidth, ls=linestyle, label=method))
    handles.extend(
        [
            Line2D([0], [0], marker="o", color=COLOR_NEUTRAL_MID, markerfacecolor=COLOR_NEUTRAL_LIGHT, lw=0, label="ID mean"),
            Line2D([0], [0], marker="o", color=COLOR_NEUTRAL_MID, markerfacecolor=COLOR_NEUTRAL_MID, lw=0, label="OOD mean"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=10,
        columnspacing=0.85,
        handlelength=1.7,
        handletextpad=0.35,
        frameon=False,
        fontsize=6.9,
    )


def _clean_axes(ax) -> None:
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")


def _label_panels(fig, panel_axes, labels, titles) -> None:
    fig.canvas.draw()
    y = 0.855
    for axes, label, title in zip(panel_axes, labels, titles):
        left = min(ax.get_position().x0 for ax in axes)
        fig.text(
            left - 0.038,
            y,
            label,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=COLOR_NEUTRAL_DARK,
        )
        if title:
            fig.text(
                left - 0.013,
                y + 0.001,
                title,
                fontsize=8.6,
                fontweight="bold",
                ha="left",
                va="bottom",
                color=COLOR_NEUTRAL_DARK,
            )


if __name__ == "__main__":
    main()
