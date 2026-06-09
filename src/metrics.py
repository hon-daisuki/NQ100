from __future__ import annotations

import pandas as pd


def describe_time_range(df: pd.DataFrame, datetime_col: str = "日時") -> dict[str, object]:
    dt = pd.to_datetime(df[datetime_col])
    return {
        "rows": len(df),
        "start": dt.min(),
        "end": dt.max(),
        "columns": list(df.columns),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return (
        pd.DataFrame({"missing": missing, "missing_rate": missing / len(df)})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )
