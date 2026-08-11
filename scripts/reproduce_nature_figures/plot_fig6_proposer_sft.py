from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    NATURE_COLORS,
    label_panel,
    save,
    setup,
)


PRE_SFT_COLOR = COLOR_NEUTRAL_LIGHT
PRE_SFT_EDGE = COLOR_NEUTRAL_MID
PRE_SFT_LINE = COLOR_NEUTRAL_MID
SFT_BLUE = NATURE_COLORS["blue"]
SFT_BLUE_EDGE = NATURE_COLORS["blue"]
SFT_TEAL = NATURE_COLORS["teal"]
SFT_GREEN = NATURE_COLORS["teal"]
SFT_GREEN_EDGE = NATURE_COLORS["teal"]
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "sft_quality_gain",
    ["#F1F4F7", "#E2EAF0", "#C9DAE5", "#9DBCD0", NATURE_COLORS["teal"]],
)


def main() -> None:
    setup()
    groups = ["Overall", "LLMSR", "SLD", "Feynman", "SRSD"]
    x = np.arange(len(groups))
    width = 0.31

    pre_candidates = np.array([18, 18, 7, 18, 19], dtype=float)
    sft_candidates = np.array([12, 12, 7, 12, 12], dtype=float)
    pre_pass = np.array([73.5, 79.0, 100.0, 80.0, 54.5])
    sft_pass = np.array([75.0, 79.8, 100.0, 82.2, 56.2])
    efficiency_gain = np.array([1.8, 2.4, 1.0, 2.1, 1.4])
    repair_pre = np.array([5, 2, 0, 4, 11], dtype=float)
    repair_sft = np.array([58, 77, 83, 48, 48], dtype=float)
    repair_gain = repair_sft - repair_pre

    fig = plt.figure(figsize=(8.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.46)

    ax = fig.add_subplot(gs[0, 0])
    pre_bars = ax.bar(
        x - width / 2,
        pre_candidates,
        width,
        color=PRE_SFT_COLOR,
        edgecolor=PRE_SFT_EDGE,
        linewidth=0.7,
        label="Pre-SFT",
    )
    sft_bars = ax.bar(
        x + width / 2,
        sft_candidates,
        width,
        color=SFT_BLUE,
        edgecolor=SFT_BLUE_EDGE,
        linewidth=0.7,
        label="SFT",
    )
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 24)
    ax.set_ylabel("Median candidates")
    ax.set_title("a. Search efficiency")
    _clean_axes(ax)

    ax2 = ax.twinx()
    pre_line = ax2.plot(x, pre_pass, color=PRE_SFT_LINE, marker="o", ms=4.0, lw=1.3, label="Pre-SFT PASS")
    sft_line = ax2.plot(x, sft_pass, color=SFT_TEAL, marker="o", ms=4.0, lw=1.3, label="SFT PASS")
    ax2.set_ylim(45, 103)
    ax2.set_ylabel("PASS (%)", labelpad=10)
    ax2.grid(False)
    handles = [pre_bars, sft_bars, pre_line[0], sft_line[0]]
    labels = ["Pre-SFT", "SFT", "Pre-SFT PASS", "SFT PASS"]
    ax.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.52, -0.20), frameon=False)
    label_panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    gain_colors = [SFT_BLUE, SFT_BLUE, PRE_SFT_COLOR, SFT_BLUE, SFT_BLUE]
    ax.bar(x, efficiency_gain, color=gain_colors, edgecolor=SFT_BLUE_EDGE, linewidth=0.7)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 2.9)
    ax.set_ylabel("Gain (x)", labelpad=8)
    ax.set_title("b. Efficiency gain")
    _clean_axes(ax)
    for xi, value in zip(x, efficiency_gain):
        ax.text(xi, value + 0.08, f"{value:.1f}x", ha="center", va="bottom", fontsize=8, color=SFT_BLUE_EDGE, fontweight="bold")
    label_panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(
        x - width / 2,
        repair_pre,
        width,
        color=PRE_SFT_COLOR,
        edgecolor=PRE_SFT_EDGE,
        linewidth=0.7,
        label="Pre-SFT",
    )
    ax.bar(
        x + width / 2,
        repair_sft,
        width,
        color=SFT_GREEN,
        edgecolor=SFT_GREEN_EDGE,
        linewidth=0.7,
        label="SFT",
    )
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 96)
    ax.set_ylabel("Cases improved by repair (%)")
    ax.set_title("c. Repair utility")
    _clean_axes(ax)
    for xi, gain, top in zip(x, repair_gain, repair_sft):
        ax.text(xi + width / 2, top + 3, f"+{gain:.0f}", ha="center", va="bottom", fontsize=8, color=SFT_GREEN_EDGE)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.50, -0.20), frameon=False)
    label_panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    columns = ["PASS", "MSE(PASS)", "PASS|cand.", "Repair"]
    display_text = np.array(
        [
            ["+1.5", "1.33x", "1.4x", "+53"],
            ["+0.8", "1.70x", "1.4x", "+75"],
            ["+0.0", "1.00x", "1.0x", "+83"],
            ["+2.2", "1.40x", "1.5x", "+44"],
            ["+1.7", "~0", "1.4x", "+37"],
        ],
        dtype=object,
    )
    heat = np.array(
        [
            [1.5 / 2.2, 0.33 / 0.70, 0.4 / 0.5, 53 / 83],
            [0.8 / 2.2, 0.70 / 0.70, 0.4 / 0.5, 75 / 83],
            [0.0, 0.0, 0.0, 83 / 83],
            [2.2 / 2.2, 0.40 / 0.70, 0.5 / 0.5, 44 / 83],
            [1.7 / 2.2, 0.0, 0.4 / 0.5, 37 / 83],
        ],
        dtype=float,
    )
    im = ax.imshow(heat, cmap=GAIN_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.set_title("d. Quality gains")
    _clean_axes(ax)
    for i in range(display_text.shape[0]):
        for j in range(display_text.shape[1]):
            color = COLOR_NEUTRAL_LIGHT if heat[i, j] > 0.68 else COLOR_NEUTRAL_DARK
            ax.text(j, i, display_text[i, j], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("gain")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["neutral", "gain", "large"])
    label_panel(ax, "d")

    save(fig, "Fig6_proposer_sft")


def _clean_axes(ax) -> None:
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")


if __name__ == "__main__":
    main()
