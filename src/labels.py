from __future__ import annotations

import pandas as pd


def add_future_return_label(
    df: pd.DataFrame,
    close_col: str = "close",
    horizon: int = 4,
    threshold: float = 0.005,
    label_col: str = "is_trend",
) -> pd.DataFrame:
    """Label rows where the absolute future return exceeds threshold."""
    out = df.copy()
    future_return_col = f"future_return_{horizon}h"
    out[future_return_col] = out[close_col].shift(-horizon) / out[close_col] - 1
    out[label_col] = (out[future_return_col].abs() > threshold).astype(int)
    return out


def add_direction_label(
    df: pd.DataFrame,
    return_col: str = "future_return_4h",
    threshold: float = 0.005,
    label_col: str = "signal",
) -> pd.DataFrame:
    """Create -1/0/1 labels from a future return column."""
    out = df.copy()
    out[label_col] = 0
    out.loc[out[return_col] > threshold, label_col] = 1
    out.loc[out[return_col] < -threshold, label_col] = -1
    return out
