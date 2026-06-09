from __future__ import annotations

import numpy as np
import pandas as pd


def build_long_short_returns(
    df: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "future_return_4h",
    cost_per_trade: float = 0.0,
) -> pd.Series:
    """Return per-trade strategy returns from -1/0/1 signals."""
    signal = df[signal_col].fillna(0).clip(-1, 1)
    gross = signal * df[return_col].fillna(0)
    turnover = signal.diff().abs().fillna(signal.abs())
    return gross - turnover * cost_per_trade


def summarize_returns(returns: pd.Series, periods_per_year: int | None = None) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "trades": 0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "mean_return": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
        "trades": float((returns != 0).sum()),
        "total_return": float(equity.iloc[-1] - 1),
        "win_rate": float((returns > 0).mean()),
        "mean_return": float(returns.mean()),
        "max_drawdown": float(drawdown.min()),
    }

    if periods_per_year:
        std = returns.std(ddof=0)
        summary["sharpe"] = float(np.sqrt(periods_per_year) * returns.mean() / std) if std else 0.0

    return summary
