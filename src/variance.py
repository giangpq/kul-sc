"""
variance.py — Compute per-(season_bucket, WeekBeforeArrival) demand variance
from calibration set residuals, using the retrained model.

Output: logs/demand_variance.csv
  columns: season_bucket, WeekBeforeArrival, demand_variance
"""

import sys
import joblib
import polars as pl
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("logs")


def apply_encodings(df: pl.DataFrame, cat_encoders: dict) -> pl.DataFrame:
    for col, mapping in cat_encoders.items():
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32)
                .alias(col + "_enc")
            )
    return df


def add_season_bucket(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("WeekStartDate").cast(pl.Date, strict=False).dt.month().alias("month")
    )
    return df.with_columns(
        pl.when(pl.col("month").is_in([7, 8])).then(pl.lit("Peak"))
          .when(pl.col("month").is_in([3, 4, 6, 12])).then(pl.lit("High"))
          .when(pl.col("month").is_in([1, 5, 9, 10])).then(pl.lit("Mid"))
          .otherwise(pl.lit("Low"))
          .alias("season_bucket")
    )


def main():
    model        = joblib.load(MODEL_DIR / "model.pkl")
    fit_stats    = joblib.load(MODEL_DIR / "fit_stats.pkl")
    cat_encoders = joblib.load(MODEL_DIR / "cat_encoders.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

    val = pl.read_csv(DATA_DIR / "calibration.csv")
    val, _ = engineer_features(val, fit_stats=fit_stats)
    val = apply_encodings(val, cat_encoders)
    val = add_season_bucket(val)

    X = val[feature_cols].to_numpy()
    predicted = model.predict(X)
    actual    = val["HistoricalBookedNights"].to_numpy()
    residuals = actual - predicted

    val = val.with_columns(
        pl.Series("squared_residual", residuals ** 2)
    )

    variance_lookup = (
        val.group_by(["season_bucket", "WeekBeforeArrival"])
        .agg(pl.col("squared_residual").mean().alias("demand_variance"))
        .sort(["season_bucket", "WeekBeforeArrival"])
    )

    variance_lookup.write_csv(MODEL_DIR / "demand_variance.csv")
    print(f"Saved demand_variance.csv — {len(variance_lookup)} rows")
    print(variance_lookup.head(10))


if __name__ == "__main__":
    main()
