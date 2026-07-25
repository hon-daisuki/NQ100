from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokyo_scalp import find_drive_root, load_minute_data, make_entry_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thresholds used by the live signal app.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--output", type=Path, default=Path("docs/tokyo_scalp_live_profile.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or find_drive_root()
    minute_df = load_minute_data(data_root=data_root, years=args.years)
    entries = make_entry_frame(minute_df)

    range_30 = entries["range_30m_points"].dropna()
    spread = entries["spread_points"].dropna()
    profile = {
        "title": "NQ100 Tokyo Scalp Live Profile",
        "data_root": str(data_root),
        "years": args.years,
        "date_start": str(minute_df["datetime"].min()),
        "date_end": str(minute_df["datetime"].max()),
        "tokyo_entries": int(len(entries)),
        "range_30m_points": {
            "low_threshold": float(range_30.quantile(1 / 3)),
            "mid_threshold": float(range_30.quantile(2 / 3)),
            "median": float(range_30.median()),
        },
        "spread_points": {
            "low_threshold": float(spread.quantile(1 / 3)),
            "mid_threshold": float(spread.quantile(2 / 3)),
            "median": float(spread.median()),
        },
        "reduced_stop_loss_model": {
            "side": "long",
            "take_profit_points": 2.0,
            "stop_loss_points": 30.0,
            "max_hold_minutes": 15,
            "filters": {
                "hour": "10",
                "momentum_15": "down",
                "volatility_30": "vol_low",
            },
            "momentum_15_down_points": -4.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
