from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokyo_scalp import (
    TradeConfig,
    find_drive_root,
    find_rule_models,
    load_minute_data,
    make_entry_frame,
    models_to_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Tokyo-session short-hold USTEC strategies with an 80% win-rate target."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    parser.add_argument("--target-win-rate", type=float, default=0.80)
    parser.add_argument("--min-train-trades", type=int, default=40)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--min-test-trades", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or find_drive_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading minute data from: {data_root}")
    minute_df = load_minute_data(data_root=data_root, years=args.years)
    entries = make_entry_frame(minute_df)
    print(f"Minute rows: {len(minute_df):,}")
    print(f"Tokyo candidate entries: {len(entries):,}")
    print(f"Date range: {minute_df['datetime'].min()} -> {minute_df['datetime'].max()}")

    configs = [
        TradeConfig(side=side, take_profit_points=tp, stop_loss_points=sl, max_hold_minutes=hold)
        for side in ["long", "short"]
        for tp in [2, 3, 5, 8, 10, 15]
        for sl in [8, 10, 15, 20, 30, 50]
        for hold in [15, 30, 60, 120, 180]
    ]
    feature_sets = [
        ("hour",),
        ("hour", "minute_block"),
        ("hour", "momentum_5"),
        ("hour", "momentum_15"),
        ("hour", "momentum_30"),
        ("hour", "trend_60"),
        ("hour", "volatility_30"),
        ("hour", "spread_bucket"),
        ("hour", "minute_block", "momentum_5"),
        ("hour", "minute_block", "momentum_15"),
        ("hour", "momentum_5", "trend_60"),
        ("hour", "momentum_15", "trend_60"),
        ("hour", "momentum_5", "volatility_30"),
        ("hour", "momentum_15", "volatility_30"),
        ("hour", "trend_60", "volatility_30"),
        ("hour", "minute_block", "trend_60"),
    ]

    print(f"Searching {len(configs):,} trade configs x {len(feature_sets)} feature sets...")
    models = find_rule_models(
        minute_df=minute_df,
        entries=entries,
        configs=configs,
        feature_sets=feature_sets,
        min_train_trades=args.min_train_trades,
        min_validation_trades=args.min_validation_trades,
        min_test_trades=args.min_test_trades,
        target_win_rate=args.target_win_rate,
    )

    models_df = models_to_frame(models)
    csv_path = output_dir / "tokyo_scalp_models.csv"
    json_path = output_dir / "tokyo_scalp_results.json"
    models_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    best = asdict(models[0]) if models else None
    payload = {
        "title": "NQ100 Tokyo Scalp Strategy Search",
        "target_win_rate": args.target_win_rate,
        "data_root": str(data_root),
        "years": args.years,
        "date_start": str(minute_df["datetime"].min()),
        "date_end": str(minute_df["datetime"].max()),
        "minute_rows": int(len(minute_df)),
        "tokyo_candidate_entries": int(len(entries)),
        "candidate_models": int(len(models)),
        "best_model": best,
        "top_models": [asdict(model) for model in models[:25]],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if models:
        print("Found 80%+ out-of-sample strategy candidates.")
        print(models_df.head(10).to_string(index=False))
    else:
        print("No strategy met the target across train, validation, and test splits.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
