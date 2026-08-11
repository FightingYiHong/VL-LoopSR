# tools/plot_generator_tool.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from tools.plot_style import NATURE_CMAP, NATURE_COLORS, save_nature_figure, set_nature_style


@dataclass
class PlotResult:
    """
    保存单张图的结果。
    """
    image_path: str
    success: bool
    description: str
    error_message: Optional[str] = None


class PlotGeneratorTool:
    """
    把符号回归数据画成图片，供多模态模型阅读。

    支持：
    - 1D: y vs x1
    - 2D: y vs x1, y vs x2, x1-x2 平面按 y 着色
    - 高维: 对每个变量画 y vs xi，最多保留前 max_plots 张
    """

    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _save_current_figure(self, save_path: str):
        self.ensure_dir(os.path.dirname(save_path))
        plt.tight_layout()
        save_nature_figure(plt, save_path, "visual_observation", dpi=160)
        plt.close()

    def plot_1d_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        save_path: str,
        title: Optional[str] = None,
    ) -> PlotResult:
        """
        画单变量散点图：y vs x_col
        """
        description = f"scatter plot of {y_col} versus {x_col}"

        try:
            set_nature_style(plt)
            plt.figure(figsize=(6, 4))
            plt.scatter(df[x_col], df[y_col], s=18, color=NATURE_COLORS["blue"], alpha=0.72, edgecolors="none")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title or f"{y_col} vs {x_col}")
            self._save_current_figure(save_path)

            return PlotResult(
                image_path=save_path,
                success=True,
                description=description,
                error_message=None,
            )
        except Exception as e:
            return PlotResult(
                image_path=save_path,
                success=False,
                description=description,
                error_message=str(e),
            )

    def plot_2d_colored_scatter(
        self,
        df: pd.DataFrame,
        x1_col: str,
        x2_col: str,
        y_col: str,
        save_path: str,
        title: Optional[str] = None,
    ) -> PlotResult:
        """
        画二维平面图：x1-x2 平面，颜色表示 y
        """
        description = f"{x1_col}-{x2_col} plane colored by {y_col}"

        try:
            set_nature_style(plt)
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(df[x1_col], df[x2_col], c=df[y_col], s=18, cmap=NATURE_CMAP, alpha=0.82, edgecolors="none")
            plt.xlabel(x1_col)
            plt.ylabel(x2_col)
            plt.title(title or f"{x1_col}-{x2_col} colored by {y_col}")
            plt.colorbar(sc, label=y_col)
            self._save_current_figure(save_path)

            return PlotResult(
                image_path=save_path,
                success=True,
                description=description,
                error_message=None,
            )
        except Exception as e:
            return PlotResult(
                image_path=save_path,
                success=False,
                description=description,
                error_message=str(e),
            )

    def plot_dataset(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
        output_dir: str,
        prefix: str = "dataset_plot",
        max_plots: int = 6,
    ) -> List[PlotResult]:
        """
        按数据维度自动绘图。

        规则：
        - 1D: 只画 y vs x1
        - 2D: 画 y vs x1, y vs x2, 以及 x1-x2 平面按 y 着色
        - 高维: 画 y vs xi，最多 max_plots 张
        """
        results: List[PlotResult] = []

        if len(feature_names) == 0:
            return results

        self.ensure_dir(output_dir)

        # 1D
        if len(feature_names) == 1:
            x_col = feature_names[0]
            save_path = os.path.join(output_dir, f"{prefix}_y_vs_{x_col}.png")
            results.append(
                self.plot_1d_scatter(
                    df=df,
                    x_col=x_col,
                    y_col=target_name,
                    save_path=save_path,
                    title=f"{target_name} vs {x_col}",
                )
            )
            return results

        # 2D
        if len(feature_names) == 2:
            x1, x2 = feature_names

            save_path_1 = os.path.join(output_dir, f"{prefix}_y_vs_{x1}.png")
            save_path_2 = os.path.join(output_dir, f"{prefix}_y_vs_{x2}.png")
            save_path_3 = os.path.join(output_dir, f"{prefix}_{x1}_{x2}_colored_by_{target_name}.png")

            results.append(
                self.plot_1d_scatter(
                    df=df,
                    x_col=x1,
                    y_col=target_name,
                    save_path=save_path_1,
                    title=f"{target_name} vs {x1}",
                )
            )
            results.append(
                self.plot_1d_scatter(
                    df=df,
                    x_col=x2,
                    y_col=target_name,
                    save_path=save_path_2,
                    title=f"{target_name} vs {x2}",
                )
            )
            results.append(
                self.plot_2d_colored_scatter(
                    df=df,
                    x1_col=x1,
                    x2_col=x2,
                    y_col=target_name,
                    save_path=save_path_3,
                    title=f"{x1}-{x2} colored by {target_name}",
                )
            )
            return results

        # 高维
        for x_col in feature_names[:max_plots]:
            save_path = os.path.join(output_dir, f"{prefix}_y_vs_{x_col}.png")
            results.append(
                self.plot_1d_scatter(
                    df=df,
                    x_col=x_col,
                    y_col=target_name,
                    save_path=save_path,
                    title=f"{target_name} vs {x_col}",
                )
            )

        return results
