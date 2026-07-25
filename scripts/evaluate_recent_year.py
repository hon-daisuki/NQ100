from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokyo_scalp import (
    TradeConfig,
    apply_filters,
    find_drive_root,
    load_minute_data,
    make_entry_frame,
    simulate_trades,
    simulate_trades_fast,
    summarize_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Tokyo scalp candidates over the latest available one-year window."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--models-csv", type=Path, default=Path("docs/tokyo_scalp_models.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--top-precise", type=int, default=30)
    parser.add_argument("--target-win-rate", type=float, default=0.80)
    parser.add_argument("--max-stop-loss", type=float, default=None)
    return parser.parse_args()


def row_to_config(row: pd.Series) -> TradeConfig:
    return TradeConfig(
        side=str(row["config_side"]),
        take_profit_points=float(row["config_take_profit_points"]),
        stop_loss_points=float(row["config_stop_loss_points"]),
        max_hold_minutes=int(row["config_max_hold_minutes"]),
    )


def parse_filters(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    return {str(key): str(val) for key, val in parsed.items()}


def evaluate_candidate(
    minute_df: pd.DataFrame,
    entries: pd.DataFrame,
    row: pd.Series,
    precise: bool,
) -> dict:
    config = row_to_config(row)
    filters = parse_filters(row["filters_json"])
    filtered_entries = apply_filters(entries, filters)
    if filtered_entries.empty:
        summary = summarize_points(pd.DataFrame())
    else:
        simulator = simulate_trades if precise else simulate_trades_fast
        trades = simulator(minute_df, filtered_entries, config)
        summary = summarize_points(trades)
        exit_counts = trades["exit_reason"].value_counts().to_dict()

    result = {
        "side": config.side,
        "take_profit_points": config.take_profit_points,
        "stop_loss_points": config.stop_loss_points,
        "max_hold_minutes": config.max_hold_minutes,
        "filters": filters,
        "trades": summary["trades"],
        "win_rate": summary["win_rate"],
        "avg_points": summary["avg_points"],
        "profit_factor": summary["profit_factor"],
        "total_points": summary["total_points"],
        "max_loss_points": summary["max_loss_points"],
        "source_test_win_rate": float(row.get("test_win_rate", 0.0)),
        "source_test_total_points": float(row.get("test_total_points", 0.0)),
    }
    if filtered_entries.empty:
        result["exit_counts"] = {}
    else:
        result["exit_counts"] = exit_counts
    return result


def main() -> None:
    args = parse_args()
    data_root = args.data_root or find_drive_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = pd.read_csv(args.models_csv)
    if args.max_stop_loss is not None:
        models = models[models["config_stop_loss_points"] <= args.max_stop_loss].copy()

    print(f"Loading recent data from: {data_root}")
    minute_df = load_minute_data(data_root=data_root, years=[2025, 2026])
    end_date = minute_df["datetime"].max()
    start_date = end_date - pd.DateOffset(years=1)
    minute_recent = minute_df[minute_df["datetime"] >= start_date].copy().reset_index(drop=True)
    entries_recent = make_entry_frame(minute_recent)

    print(f"Recent window: {start_date} -> {end_date}")
    print(f"Minute rows: {len(minute_recent):,}")
    print(f"Tokyo entries: {len(entries_recent):,}")
    print(f"Candidates: {len(models):,}")

    fast_rows = [
        evaluate_candidate(minute_recent, entries_recent, row, precise=False)
        for _, row in models.iterrows()
    ]
    fast_df = pd.DataFrame(fast_rows)
    valid_fast = fast_df[
        (fast_df["trades"] > 0)
        & (fast_df["win_rate"] >= args.target_win_rate)
        & (fast_df["total_points"] > 0)
    ].copy()
    valid_fast = valid_fast.sort_values(
        ["total_points", "profit_factor", "win_rate", "trades"],
        ascending=[False, False, False, False],
    )

    precise_source = valid_fast.head(args.top_precise)
    precise_rows = []
    for _, fast_row in precise_source.iterrows():
        mask = (
            (models["config_side"].astype(str) == str(fast_row["side"]))
            & (models["config_take_profit_points"].astype(float) == float(fast_row["take_profit_points"]))
            & (models["config_stop_loss_points"].astype(float) == float(fast_row["stop_loss_points"]))
            & (models["config_max_hold_minutes"].astype(int) == int(fast_row["max_hold_minutes"]))
            & (models["filters_json"].map(parse_filters).map(json.dumps) == json.dumps(fast_row["filters"]))
        )
        source_row = models.loc[mask].iloc[0]
        precise_rows.append(evaluate_candidate(minute_recent, entries_recent, source_row, precise=True))

    precise_df = pd.DataFrame(precise_rows)
    if not precise_df.empty:
        precise_df = precise_df[
            (precise_df["trades"] > 0)
            & (precise_df["win_rate"] >= args.target_win_rate)
            & (precise_df["total_points"] > 0)
        ].sort_values(
            ["total_points", "profit_factor", "win_rate", "trades"],
            ascending=[False, False, False, False],
        )

    best = precise_df.iloc[0].to_dict() if not precise_df.empty else None
    payload = {
        "title": "NQ100 Recent One-Year Tokyo Scalp Evaluation",
        "target_win_rate": args.target_win_rate,
        "max_stop_loss": args.max_stop_loss,
        "data_root": str(data_root),
        "date_start": str(start_date),
        "date_end": str(end_date),
        "minute_rows": int(len(minute_recent)),
        "tokyo_candidate_entries": int(len(entries_recent)),
        "candidate_models": int(len(models)),
        "fast_positive_candidates": int(len(valid_fast)),
        "precise_positive_candidates": int(len(precise_df)),
        "best_recent_model": best,
        "top_recent_models": precise_df.head(25).to_dict(orient="records") if not precise_df.empty else [],
    }

    suffix = "" if args.max_stop_loss is None else f"_sl{int(args.max_stop_loss)}"
    json_path = output_dir / f"tokyo_scalp_recent_year{suffix}.json"
    csv_path = output_dir / f"tokyo_scalp_recent_year{suffix}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    precise_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if best:
        print("Best recent one-year model:")
        print(pd.DataFrame([best]).to_string(index=False))
    else:
        print("No candidate remained positive with 80%+ win rate in precise recent-year evaluation.")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
