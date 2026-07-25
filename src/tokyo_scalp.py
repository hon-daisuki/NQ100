from __future__ import annotations

import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DRIVE_ROOTS = [
    Path(r"G:\マイドライブ\CFD機械学習\NQ100"),
    Path("/content/drive/MyDrive/CFD機械学習/NQ100"),
]


@dataclass(frozen=True)
class TradeConfig:
    side: str
    take_profit_points: float
    stop_loss_points: float
    max_hold_minutes: int


@dataclass(frozen=True)
class RuleModel:
    config: TradeConfig
    filters: dict[str, str]
    train_trades: int
    train_win_rate: float
    train_avg_points: float
    train_profit_factor: float
    validation_trades: int
    validation_win_rate: float
    validation_avg_points: float
    validation_profit_factor: float
    test_trades: int
    test_win_rate: float
    test_avg_points: float
    test_profit_factor: float
    test_total_points: float
    test_max_loss_points: float


def find_drive_root() -> Path:
    for root in DEFAULT_DRIVE_ROOTS:
        if root.exists():
            return root
    raise FileNotFoundError(
        "NQ100 data folder was not found. Set --data-root to the Google Drive NQ100 folder."
    )


def _read_minute_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, encoding="cp932")
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_minute_data(
    data_root: Path | None = None,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    root = data_root or find_drive_root()
    minute_dir = root / "CSV_1min_chart"
    if not minute_dir.exists():
        raise FileNotFoundError(f"Minute data folder not found: {minute_dir}")

    year_set = set(years) if years else None
    zip_paths = sorted(minute_dir.glob("USTEC_*.zip"))
    if year_set:
        zip_paths = [
            path
            for path in zip_paths
            if int(path.stem.replace("USTEC_", "")[:4]) in year_set
        ]

    frames = []
    for path in zip_paths:
        frame = _read_minute_csv_from_zip(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No minute zip files found in {minute_dir}")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(
        columns={
            "日時": "datetime",
            "始値(BID)": "bid_open",
            "高値(BID)": "bid_high",
            "安値(BID)": "bid_low",
            "終値(BID)": "bid_close",
            "始値(ASK)": "ask_open",
            "高値(ASK)": "ask_high",
            "安値(ASK)": "ask_low",
            "終値(ASK)": "ask_close",
        }
    )
    raw["datetime"] = pd.to_datetime(raw["datetime"].astype(str), format="%Y%m%d%H%M")
    numeric_cols = [col for col in raw.columns if col != "datetime"]
    raw[numeric_cols] = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["datetime", "bid_close", "ask_close"])
    raw = raw.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    raw["open"] = (raw["bid_open"] + raw["ask_open"]) / 2
    raw["high"] = (raw["bid_high"] + raw["ask_high"]) / 2
    raw["low"] = (raw["bid_low"] + raw["ask_low"]) / 2
    raw["close"] = (raw["bid_close"] + raw["ask_close"]) / 2
    raw["spread_points"] = raw["ask_close"] - raw["bid_close"]
    return raw


def add_scalp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["minute"] = out["datetime"].dt.minute
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["year"] = out["datetime"].dt.year
    out["minute_block"] = pd.cut(
        out["minute"],
        bins=[-1, 14, 29, 44, 59],
        labels=["00-14", "15-29", "30-44", "45-59"],
    ).astype(str)
    for window in [5, 15, 30, 60]:
        out[f"return_{window}m_points"] = out["close"] - out["close"].shift(window)
        out[f"range_{window}m_points"] = (
            out["high"].rolling(window).max() - out["low"].rolling(window).min()
        )
    out["sma_30"] = out["close"].rolling(30).mean()
    out["sma_60"] = out["close"].rolling(60).mean()
    out["trend_60"] = np.where(out["close"] >= out["sma_60"], "above_sma60", "below_sma60")
    out["momentum_5"] = _bucket_signed_points(out["return_5m_points"], flat_points=2.0)
    out["momentum_15"] = _bucket_signed_points(out["return_15m_points"], flat_points=4.0)
    out["momentum_30"] = _bucket_signed_points(out["return_30m_points"], flat_points=6.0)
    out["volatility_30"] = _bucket_quantile(out["range_30m_points"], "vol")
    out["spread_bucket"] = _bucket_quantile(out["spread_points"], "spread")
    return out


def _bucket_signed_points(series: pd.Series, flat_points: float) -> pd.Series:
    return pd.Series(
        np.select(
            [series <= -flat_points, series >= flat_points],
            ["down", "up"],
            default="flat",
        ),
        index=series.index,
    )


def _bucket_quantile(series: pd.Series, prefix: str) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        bucket = pd.qcut(ranked, 3, labels=[f"{prefix}_low", f"{prefix}_mid", f"{prefix}_high"])
        return bucket.astype(str)
    except ValueError:
        return pd.Series(f"{prefix}_mid", index=series.index)


def make_entry_frame(df: pd.DataFrame, tokyo_start_hour: int = 8, tokyo_end_hour: int = 12) -> pd.DataFrame:
    frame = add_scalp_features(df)
    mask = (
        (frame["hour"] >= tokyo_start_hour)
        & (frame["hour"] <= tokyo_end_hour)
        & frame["sma_60"].notna()
    )
    return frame.loc[mask].copy().reset_index(drop=False).rename(columns={"index": "source_index"})


def simulate_trades(
    minute_df: pd.DataFrame,
    entries: pd.DataFrame,
    config: TradeConfig,
) -> pd.DataFrame:
    highs = minute_df["high"].to_numpy()
    lows = minute_df["low"].to_numpy()
    closes = minute_df["close"].to_numpy()

    rows = []
    is_long = config.side == "long"
    tp = config.take_profit_points
    sl = config.stop_loss_points
    horizon = config.max_hold_minutes

    for row in entries.itertuples(index=False):
        source_index = int(row.source_index)
        entry_price = float(row.close)
        end_index = min(source_index + horizon, len(minute_df) - 1)
        if source_index + 1 > end_index:
            continue

        outcome_points = None
        exit_reason = "timeout"
        for idx in range(source_index + 1, end_index + 1):
            if is_long:
                hit_tp = highs[idx] >= entry_price + tp
                hit_sl = lows[idx] <= entry_price - sl
            else:
                hit_tp = lows[idx] <= entry_price - tp
                hit_sl = highs[idx] >= entry_price + sl

            if hit_tp and hit_sl:
                outcome_points = -sl
                exit_reason = "tp_sl_same_minute"
                break
            if hit_tp:
                outcome_points = tp
                exit_reason = "take_profit"
                break
            if hit_sl:
                outcome_points = -sl
                exit_reason = "stop_loss"
                break

        if outcome_points is None:
            final_delta = closes[end_index] - entry_price
            outcome_points = final_delta if is_long else -final_delta

        rows.append(
            {
                "datetime": row.datetime,
                "year": int(row.year),
                "entry_price": entry_price,
                "side": config.side,
                "outcome_points": float(outcome_points),
                "is_win": outcome_points > 0,
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(rows)


def simulate_trades_fast(
    minute_df: pd.DataFrame,
    entries: pd.DataFrame,
    config: TradeConfig,
) -> pd.DataFrame:
    """Vectorized conservative trade simulation for broad strategy search.

    If both take-profit and stop-loss are touched inside the same holding window,
    the trade is counted as a stop-loss. This avoids overstating win rate during
    the search phase.
    """
    horizon = config.max_hold_minutes
    source_index = entries["source_index"].to_numpy(dtype=int)
    entry = entries["close"].to_numpy(dtype=float)

    future_high = _future_rolling_extreme(minute_df["high"], horizon, "max")[source_index]
    future_low = _future_rolling_extreme(minute_df["low"], horizon, "min")[source_index]

    close_values = minute_df["close"].to_numpy(dtype=float)
    exit_index = np.minimum(source_index + horizon, len(close_values) - 1)
    final_delta = close_values[exit_index] - entry

    tp = config.take_profit_points
    sl = config.stop_loss_points
    if config.side == "long":
        hit_tp = future_high >= entry + tp
        hit_sl = future_low <= entry - sl
        timeout_points = final_delta
    else:
        hit_tp = future_low <= entry - tp
        hit_sl = future_high >= entry + sl
        timeout_points = -final_delta

    outcome = np.where(hit_sl, -sl, np.where(hit_tp, tp, timeout_points))
    exit_reason = np.where(hit_sl, "stop_loss", np.where(hit_tp, "take_profit", "timeout"))

    return pd.DataFrame(
        {
            "datetime": entries["datetime"].to_numpy(),
            "year": entries["year"].to_numpy(dtype=int),
            "entry_price": entry,
            "side": config.side,
            "outcome_points": outcome.astype(float),
            "is_win": outcome > 0,
            "exit_reason": exit_reason,
        }
    )


def _future_rolling_extreme(series: pd.Series, horizon: int, method: str) -> np.ndarray:
    shifted = series.shift(-1).iloc[::-1]
    rolling = shifted.rolling(horizon, min_periods=1)
    if method == "max":
        out = rolling.max()
    elif method == "min":
        out = rolling.min()
    else:
        raise ValueError(f"Unknown method: {method}")
    return out.iloc[::-1].bfill().to_numpy(dtype=float)


def summarize_points(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_points": 0.0,
            "profit_factor": 0.0,
            "total_points": 0.0,
            "max_loss_points": 0.0,
        }
    points = frame["outcome_points"]
    wins = points[points > 0]
    losses = points[points < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    return {
        "trades": int(len(frame)),
        "win_rate": float((points > 0).mean()),
        "avg_points": float(points.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else math.inf if gross_profit else 0.0,
        "total_points": float(points.sum()),
        "max_loss_points": float(points.min()) if len(points) else 0.0,
    }


def find_rule_models(
    minute_df: pd.DataFrame,
    entries: pd.DataFrame,
    configs: Iterable[TradeConfig],
    feature_sets: Iterable[tuple[str, ...]],
    min_train_trades: int = 40,
    min_validation_trades: int = 10,
    min_test_trades: int = 5,
    target_win_rate: float = 0.80,
) -> list[RuleModel]:
    models: list[RuleModel] = []
    entries_by_time = entries.set_index("datetime", drop=False)

    for config in configs:
        outcomes = simulate_trades_fast(minute_df, entries, config)
        merged = entries_by_time.join(outcomes.set_index("datetime"), rsuffix="_trade")
        merged = merged.dropna(subset=["outcome_points"]).copy()

        train = merged[merged["year"] <= 2024]
        validation = merged[merged["year"] == 2025]
        test = merged[merged["year"] >= 2026]
        if train.empty or validation.empty or test.empty:
            continue

        for features in feature_sets:
            train_groups = train.groupby(list(features), dropna=False, observed=True)
            for key, train_group in train_groups:
                if len(train_group) < min_train_trades:
                    continue
                train_summary = summarize_points(train_group)
                if (
                    train_summary["win_rate"] < target_win_rate
                    or train_summary["avg_points"] <= 0
                    or train_summary["profit_factor"] < 1.0
                ):
                    continue

                if not isinstance(key, tuple):
                    key = (key,)
                filters = dict(zip(features, map(str, key)))

                validation_group = _apply_filters(validation, filters)
                if len(validation_group) < min_validation_trades:
                    continue
                validation_summary = summarize_points(validation_group)
                if (
                    validation_summary["win_rate"] < target_win_rate
                    or validation_summary["avg_points"] <= 0
                    or validation_summary["profit_factor"] < 1.0
                ):
                    continue

                test_group = _apply_filters(test, filters)
                if len(test_group) < min_test_trades:
                    continue
                test_summary = summarize_points(test_group)
                if (
                    test_summary["win_rate"] < target_win_rate
                    or test_summary["avg_points"] <= 0
                    or test_summary["profit_factor"] < 1.0
                ):
                    continue

                models.append(
                    RuleModel(
                        config=config,
                        filters=filters,
                        train_trades=train_summary["trades"],
                        train_win_rate=train_summary["win_rate"],
                        train_avg_points=train_summary["avg_points"],
                        train_profit_factor=train_summary["profit_factor"],
                        validation_trades=validation_summary["trades"],
                        validation_win_rate=validation_summary["win_rate"],
                        validation_avg_points=validation_summary["avg_points"],
                        validation_profit_factor=validation_summary["profit_factor"],
                        test_trades=test_summary["trades"],
                        test_win_rate=test_summary["win_rate"],
                        test_avg_points=test_summary["avg_points"],
                        test_profit_factor=test_summary["profit_factor"],
                        test_total_points=test_summary["total_points"],
                        test_max_loss_points=test_summary["max_loss_points"],
                    )
                )

    return sorted(
        models,
        key=lambda model: (
            model.test_win_rate,
            model.test_profit_factor,
            model.test_trades,
            model.validation_win_rate,
        ),
        reverse=True,
    )


def _apply_filters(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for col, value in filters.items():
        mask &= frame[col].astype(str) == str(value)
    return frame.loc[mask]


def models_to_frame(models: list[RuleModel]) -> pd.DataFrame:
    rows = []
    for model in models:
        row = asdict(model)
        config = row.pop("config")
        row.update({f"config_{key}": value for key, value in config.items()})
        row["filters_json"] = json.dumps(row.pop("filters"), ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)
