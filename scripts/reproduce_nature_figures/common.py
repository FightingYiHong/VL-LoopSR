from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.plot_style import (  # noqa: E402
    BENCHMARK_COLORS,
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MID,
    METHOD_COLORS,
    NATURE_CMAP,
    NATURE_COLORS,
    NATURE_PALETTE,
    NATURE_WARM_CMAP,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    TEXT,
    palette_for,
    set_nature_style,
)


OUT_DIR = ROOT / "figs" / "experiment_figures_cool_palette"

FIGURE_ALIASES = {
    "Fig3_standard_formula_recovery": "fig1_standard_recovery_abcd",
    "Fig4_ood_generalization": "fig2_ood_generalization_abc",
    "Fig5_high_dimensional_distractors": "highdim_conclusion_abcd_clean",
    "Fig6_proposer_sft": "sft_quality_angle_combined",
    "Fig7_noise_robustness": "fig7_noise_advantage_combined_no_gplearn_ffx",
    "Fig8_component_ablation": "fig4_component_ablation_abc",
}

FIG5_METHOD_COLORS = METHOD_COLORS

FIG5_PALETTE = [
    NATURE_COLORS["blue"],
    NATURE_COLORS["sky"],
    NATURE_COLORS["teal"],
    NATURE_COLORS["orange"],
    NATURE_COLORS["purple"],
    NATURE_COLORS["olive"],
    NATURE_COLORS["green"],
    NATURE_COLORS["vermillion"],
]


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_nature_style(plt)


def method_color(method: str, fallback_index: int = 0) -> str:
    key = str(method).strip().lower().replace("_", "-")
    return FIG5_METHOD_COLORS.get(key, METHOD_COLORS.get(key, FIG5_PALETTE[fallback_index % len(FIG5_PALETTE)]))


def save(fig, name: str) -> None:
    setup()
    pdf = OUT_DIR / f"{FIGURE_ALIASES.get(name, name)}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def label_panel(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLOR_NEUTRAL_DARK,
    )


def annotate_bars(ax, bars, fmt="{:.1f}", dy=0.02, fontsize=6.8):
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy * span,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLOR_NEUTRAL_DARK,
        )


def short_method_labels(methods):
    return [str(m).replace("PyOperon", "PyOperon").replace("PSRN-PSE", "PSRN-PSE") for m in methods]
