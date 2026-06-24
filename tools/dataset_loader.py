# tools/dataset_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    """
    数据集打包对象。

    这样做的目的：
    - 不让后续 tool 到处传很多个变量
    - 把 train / val / test 以及元信息统一封装
    """
    df: pd.DataFrame

    feature_names: List[str]
    target_name: str

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class DatasetLoader:
    """
    负责读取 csv 数据并切分 train / val / test。

    第一版先只支持 csv，后面要扩 parquet / npy 很容易。
    """

    def load_csv(
        self,
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        target_name: str = "y",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> DatasetBundle:
        """
        读取 csv 并切分数据。

        参数说明：
        - csv_path: csv 文件路径
        - feature_names: 若不传，则自动用除 target 外所有列
        - target_name: 目标列名
        - test_size: 总体测试集比例
        - val_size: 总体验证集比例
        """
        df = pd.read_csv(csv_path)

        if target_name not in df.columns:
            raise ValueError(f"目标列 {target_name} 不在数据中。现有列: {list(df.columns)}")

        if feature_names is None:
            feature_names = [c for c in df.columns if c != target_name]

        for col in feature_names:
            if col not in df.columns:
                raise ValueError(f"特征列 {col} 不在数据中。")

        # 先切 test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )

        # 再从剩余部分切 val
        # 注意这里要把 val_size 转成 train_val 内部比例
        relative_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
        )

        return DatasetBundle(
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
        )

    @staticmethod
    def get_xy(df: pd.DataFrame, feature_names: List[str], target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从 DataFrame 中提取 X 和 y。
        """
        X = df[feature_names]
        y = df[target_name]
        return X, y