import polars as pl
from pathlib import Path
import random
import numpy as np


RANDOM_SEED = 42
VAL_FRAC_2024 = 0.20
PEAK_BOOST = 1.20  # slight over-representation for Jul-Aug bucket

input_csv = Path("data/raw/simulation_output.csv")
out_dir = Path("data/processed")


# --- Locked price grid config — do not modify ---
PRICE_GRID_STEPS = 21        # number of candidate prices per accommodation
PRICE_GRID_RANGE = 0.40      # search ±40% around each accommodation's median price


def build_price_grids(df: pl.DataFrame, price_col: str = "DiscountedPrice") -> dict:
    """
    Build a per-accommodation price grid centred on each accommodation's
    own median historical price. Returns a dict of {id -> list[float]}.
    This is locked — the agent must not modify it.
    """
    medians = (
        df.group_by("ReservableOptionMarketGroupId")
        .agg(pl.col(price_col).median().alias("median_price"))
    )
    grids = {}
    for row in medians.iter_rows(named=True):
        centre = row["median_price"]
        grids[row["ReservableOptionMarketGroupId"]] = list(np.linspace(
            centre * (1 - PRICE_GRID_RANGE),
            centre * (1 + PRICE_GRID_RANGE),
            PRICE_GRID_STEPS,
        ))
    return grids


def simulate_revenue(
    model,
    df: pl.DataFrame,
    price_grids: dict[str, list[float]],
    feature_cols: list[str],
    price_col: str = "DiscountedPrice",
    capacity_col: str = "Capacity",
    id_col: str = "ReservableOptionMarketGroupId",
) -> pl.DataFrame:
    """
    Simulate revenue over a per-accommodation price grid.

    price_grids: dict mapping each accommodation ID to its list of candidate prices.
                 Build this with build_price_grids() — do not construct it manually.

    NOTE: Pass *remaining* capacity (i.e. Capacity - cumulative_booked_nights_so_far)
    as `capacity_col`. The agent must not modify this function.

    Returns a DataFrame with columns:
      [id_col, WeekBeforeArrival, candidate_price,
       predicted_nights, simulated_revenue, revenue_per_available_night]
    """
    results = []

    for acc_id, price_grid in price_grids.items():
        acc_df = df.filter(pl.col(id_col) == acc_id)
        if acc_df.is_empty():
            continue
        capacity = acc_df[capacity_col].to_numpy()

        for price in price_grid:
            sim_df = acc_df.with_columns(pl.lit(price).alias(price_col))
            X = sim_df[feature_cols].to_numpy()
            predicted_nights = model.predict(X).clip(0, capacity)

            results.append(
                sim_df.select([id_col, "WeekBeforeArrival"])
                .with_columns([
                    pl.lit(price).alias("candidate_price"),
                    pl.Series("predicted_nights", predicted_nights),
                    pl.Series("simulated_revenue", price * predicted_nights),
                    pl.Series("revenue_per_available_night",
                              (price * predicted_nights) / capacity),
                ])
            )

    return pl.concat(results)


if __name__ == "__main__":
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load lazily
    lf_all = pl.scan_csv(input_csv, infer_schema_length=10_000)

    # Unique stay weeks in 2024 with season buckets
    weeks_2024 = (
        lf_all.select(pl.col("WeekStartDate").cast(pl.Date, strict=False).alias("week_start"))
              .unique()
              .with_columns([
                  pl.col("week_start").dt.year().alias("year"),
                  pl.col("week_start").dt.month().alias("month"),
              ])
              .filter(pl.col("year") == 2024)
              .with_columns([
                  pl.when(pl.col("month").is_in([7, 8])).then(pl.lit("Peak"))
                    .when(pl.col("month").is_in([3, 4, 6, 12])).then(pl.lit("High"))
                    .when(pl.col("month").is_in([1, 5, 9, 10])).then(pl.lit("Mid"))
                    .otherwise(pl.lit("Low"))
                    .alias("season_bucket")
              ])
              .sort("week_start")
              .collect(engine="streaming")
    )

    n_weeks_2024 = weeks_2024.height
    n_val_total = max(1, int(round(n_weeks_2024 * VAL_FRAC_2024)))

    # Proportional quotas by bucket + slight peak boost
    bucket_counts = weeks_2024.group_by("season_bucket").agg(pl.len().alias("n")).to_dicts()
    count_map = {row["season_bucket"]: int(row["n"]) for row in bucket_counts}

    weights = {}
    for b, n in count_map.items():
        w = float(n)
        if b == "Peak":
            w *= PEAK_BOOST
        weights[b] = w

    weight_sum = sum(weights.values()) if weights else 1.0
    raw_quota = {b: n_val_total * (w / weight_sum) for b, w in weights.items()}
    quota = {b: int(raw_quota[b]) for b in raw_quota}

    # distribute remainder by largest fractional part
    remaining = n_val_total - sum(quota.values())
    frac_order = sorted(raw_quota.keys(), key=lambda b: (raw_quota[b] - quota[b]), reverse=True)
    for i in range(remaining):
        quota[frac_order[i % len(frac_order)]] += 1

    # cap quotas by available weeks and rebalance if needed
    for b in list(quota.keys()):
        quota[b] = min(quota[b], count_map.get(b, 0))
    while sum(quota.values()) < n_val_total:
        # add to bucket with most spare capacity
        candidates = [b for b in quota if quota[b] < count_map.get(b, 0)]
        if not candidates:
            break
        b = max(candidates, key=lambda x: count_map[x] - quota[x])
        quota[b] += 1

    # Sample validation weeks deterministically
    rng = random.Random(RANDOM_SEED)
    val_weeks = []
    for b in sorted(quota.keys()):
        n_take = quota[b]
        weeks_b = (
            weeks_2024.filter(pl.col("season_bucket") == b)
                      .get_column("week_start")
                      .to_list()
        )
        rng.shuffle(weeks_b)
        val_weeks.extend(weeks_b[:n_take])

    val_weeks_df = pl.DataFrame({"week_start": val_weeks}) if val_weeks else pl.DataFrame({"week_start": []})

    # Build row-level split labels
    lf_labeled = (
        lf_all.with_columns([
            pl.col("WeekStartDate").cast(pl.Date, strict=False).alias("week_start"),
            pl.col("WeekStartDate").cast(pl.Date, strict=False).dt.year().alias("year"),
        ])
        .join(val_weeks_df.lazy().with_columns(pl.lit(1).alias("is_val_week")), on="week_start", how="left")
        .with_columns([
            pl.when(pl.col("year") == 2025).then(pl.lit("test"))
              .when((pl.col("year") == 2024) & (pl.col("is_val_week") == 1)).then(pl.lit("val"))
              .when(pl.col("year") == 2024).then(pl.lit("train"))
              .otherwise(pl.lit("drop"))
              .alias("split")
        ])
    )

    # Write split CSVs
    lf_labeled.filter(pl.col("split") == "train").drop(["week_start", "year", "is_val_week", "split"]).sink_csv(out_dir / "train.csv")
    lf_labeled.filter(pl.col("split") == "val").drop(["week_start", "year", "is_val_week", "split"]).sink_csv(out_dir / "val.csv")
    lf_labeled.filter(pl.col("split") == "test").drop(["week_start", "year", "is_val_week", "split"]).sink_csv(out_dir / "test.csv")