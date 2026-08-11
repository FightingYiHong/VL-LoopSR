from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATURE_OUTPUT_ROOT = Path(os.environ.get("LLMSR_NATURE_FIG_DIR", str(PROJECT_ROOT / "figs" / "nature_subjournal")))

POSITIVE = "#4E95A8"
NEGATIVE = "#B86C62"
NEUTRAL = "#E7ECF1"
TEXT = "#24313C"

NATURE_COLORS = {
    "blue": "#3F6E9A",
    "sky": "#7FAECD",
    "teal": "#4E95A8",
    "green": "#6F8F7A",
    "orange": "#C68A42",
    "vermillion": "#B86C62",
    "purple": "#7C72A6",
    "magenta": "#A26A92",
    "olive": "#87925A",
    "brown": "#8F7C66",
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "neutral": NEUTRAL,
    "dark": TEXT,
    "midgray": "#7E8B96",
    "lightgray": NEUTRAL,
}

NATURE_PALETTE = [
    NATURE_COLORS["blue"],
    NATURE_COLORS["orange"],
    NATURE_COLORS["green"],
    NATURE_COLORS["purple"],
    NATURE_COLORS["teal"],
    NATURE_COLORS["vermillion"],
    NATURE_COLORS["sky"],
    NATURE_COLORS["magenta"],
    NATURE_COLORS["olive"],
    NATURE_COLORS["brown"],
]

COLOR_NEUTRAL_DARK = TEXT
COLOR_NEUTRAL_MID = NATURE_COLORS["midgray"]
COLOR_NEUTRAL_LIGHT = NEUTRAL

NATURE_CMAP = LinearSegmentedColormap.from_list(
    "cool_paper_blue",
    ["#F1F4F7", "#E2EAF0", "#C9DAE5", "#9DBCD0", "#6F9DBD", NATURE_COLORS["blue"]],
)
NATURE_WARM_CMAP = LinearSegmentedColormap.from_list(
    "cool_paper_warm",
    ["#F1F4F7", "#E5E9ED", "#D5C4BD", NATURE_COLORS["orange"], NEGATIVE],
)
NATURE_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cool_paper_blue_warm",
    [NATURE_COLORS["blue"], "#C9DAE5", "#F1F4F7", "#D9C1BE", NEGATIVE],
)

METHOD_COLORS = {
    "ours": NATURE_COLORS["blue"],
    "vl-loopsr": NATURE_COLORS["blue"],
    "vlloopsr": NATURE_COLORS["blue"],
    "llm-sr": NATURE_COLORS["orange"],
    "official-llm-sr": NATURE_COLORS["orange"],
    "official_llm_sr": NATURE_COLORS["orange"],
    "pysr": NATURE_COLORS["sky"],
    "pyoperon": NATURE_COLORS["teal"],
    "operon": NATURE_COLORS["teal"],
    "pse": NATURE_COLORS["vermillion"],
    "psrn": NATURE_COLORS["vermillion"],
    "psrn-pse": NATURE_COLORS["vermillion"],
    "psrn_pse": NATURE_COLORS["vermillion"],
    "dso": NATURE_COLORS["green"],
    "dsr": NATURE_COLORS["green"],
    "gplearn": NATURE_COLORS["purple"],
    "icsr": NATURE_COLORS["midgray"],
    "physo": NATURE_COLORS["olive"],
    "deap": NATURE_COLORS["magenta"],
    "bingo": "#9EA8B0",
    "direct llm": "#68737C",
    "llm-direct": "#68737C",
    "llm_direct": "#68737C",
    "ffx": NATURE_COLORS["midgray"],
    "rils-rols": NATURE_COLORS["midgray"],
    "rils_rols": NATURE_COLORS["midgray"],
    "itea": NATURE_COLORS["olive"],
}

BENCHMARK_COLORS = {
    "sldbench": NATURE_COLORS["blue"],
    "llmsrbench": NATURE_COLORS["orange"],
    "srsd": NATURE_COLORS["green"],
    "srbench": NATURE_COLORS["purple"],
    "feynman": NATURE_COLORS["teal"],
}

INTERFERENCE_COLORS = {
    "independent_irrelevant": NATURE_COLORS["blue"],
    "correlated_proxy": NATURE_COLORS["vermillion"],
    "nonlinear_decoy": NATURE_COLORS["green"],
    "independent": NATURE_COLORS["blue"],
    "proxy": NATURE_COLORS["vermillion"],
    "nonlinear": NATURE_COLORS["green"],
}

STATE_COLORS = {
    "before": NEUTRAL,
    "after": METHOD_COLORS["ours"],
    "win": POSITIVE,
    "tie": NEUTRAL,
    "loss": NEGATIVE,
}


def _norm_key(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def stable_palette_color(value: Any, index: int = 0) -> str:
    key = _norm_key(value)
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    if key in BENCHMARK_COLORS:
        return BENCHMARK_COLORS[key]
    if key in INTERFERENCE_COLORS:
        return INTERFERENCE_COLORS[key]
    code = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return NATURE_PALETTE[(code + int(index)) % len(NATURE_PALETTE)]


def palette_for(values: Iterable[Any]) -> dict[Any, str]:
    return {value: stable_palette_color(value, i) for i, value in enumerate(values)}


def set_nature_style(plt: Any | None = None, sns: Any | None = None) -> None:
    rc = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#DADDE0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.75,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "legend.title_fontsize": 7.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    mpl.rcParams.update(rc)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=NATURE_PALETTE)
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", palette=NATURE_PALETTE, rc=rc)
    if plt is not None:
        plt.rcParams.update(mpl.rcParams)


def nature_figure_dir(section: str | None = None) -> Path:
    out_dir = NATURE_OUTPUT_ROOT
    if section:
        out_dir = out_dir / str(section).strip().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_nature_figure(fig_or_plt: Any, save_path: str | Path, section: str, **savefig_kwargs: Any) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig_or_plt.savefig(save_path, **savefig_kwargs)

    mirror_path = nature_figure_dir(section) / save_path.name
    if mirror_path.resolve() != save_path.resolve():
        fig_or_plt.savefig(mirror_path, **savefig_kwargs)
    return mirror_path
