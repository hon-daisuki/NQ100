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
            "entry_win_rate": 0.0,
            "mean_return": 0.0,
            "entry_mean_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    entries = returns[returns != 0]
    wins = entries[entries > 0]
    losses = entries[entries < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    summary = {
        "trades": float(len(entries)),
        "total_return": float(equity.iloc[-1] - 1),
        "win_rate": float((returns > 0).mean()),
        "entry_win_rate": float((entries > 0).mean()) if len(entries) else 0.0,
        "mean_return": float(returns.mean()),
        "entry_mean_return": float(entries.mean()) if len(entries) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf") if gross_profit else 0.0,
        "max_drawdown": float(drawdown.min()),
    }

    if periods_per_year:
        std = returns.std(ddof=0)
        summary["sharpe"] = float(np.sqrt(periods_per_year) * returns.mean() / std) if std else 0.0

    return summary


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compound a return series into an equity curve starting at 1.0."""
    return (1 + returns.fillna(0)).cumprod()


def yearly_returns(returns: pd.Series, datetime_index: pd.Series) -> pd.DataFrame:
    """Summarize compounded returns by calendar year."""
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetime_index),
            "returns": returns.fillna(0).to_numpy(),
        }
    ).dropna(subset=["datetime"])
    frame["year"] = frame["datetime"].dt.year

    rows = []
    for year, group in frame.groupby("year"):
        year_returns = group["returns"]
        year_equity = equity_curve(year_returns)
        entries = year_returns[year_returns != 0]
        rows.append(
            {
                "year": int(year),
                "return": float(year_equity.iloc[-1] - 1) if len(year_equity) else 0.0,
                "trades": int(len(entries)),
                "entry_win_rate": float((entries > 0).mean()) if len(entries) else 0.0,
            }
        )

    return pd.DataFrame(rows)
