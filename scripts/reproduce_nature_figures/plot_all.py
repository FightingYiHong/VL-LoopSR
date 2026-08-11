from __future__ import annotations

import importlib


MODULES = [
    "plot_fig3_standard_recovery",
    "plot_fig4_ood_generalization",
    "plot_fig5_high_dimensional_distractors",
    "plot_fig6_proposer_sft",
    "plot_fig7_noise_robustness",
    "plot_fig8_component_ablation",
]


def main() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()
