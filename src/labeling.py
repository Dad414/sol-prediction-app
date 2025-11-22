"""Label generation for ternary SOL/USDT trend prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

TrendLabel = Literal["long", "neutral", "short"]


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 1
    upper_threshold: float = 0.0015  # +0.15%
    lower_threshold: float = -0.0015  # -0.15%


def generate_labels(df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Append ternary trend labels and future return to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least `close`.
    config : LabelConfig, optional
        Horizon (number of rows ahead) and threshold configuration.

    Returns
    -------
    pd.DataFrame
        Copy of input with `future_return` and `label` fields; last `horizon`
        rows are dropped due to undefined future target.
    """
    cfg = config or LabelConfig()
    data = df.copy()

    data["future_close"] = data["close"].shift(-cfg.horizon)
    data["future_return"] = data["future_close"] / data["close"] - 1

    conditions = [
        data["future_return"] >= cfg.upper_threshold,
        data["future_return"] <= cfg.lower_threshold,
    ]
    choices: tuple[TrendLabel, TrendLabel] = ("long", "short")
    data["label"] = "neutral"
    data.loc[conditions[0], "label"] = choices[0]
    data.loc[conditions[1], "label"] = choices[1]

    return data.iloc[:-cfg.horizon].reset_index(drop=True)


__all__ = ["TrendLabel", "LabelConfig", "generate_labels"]

