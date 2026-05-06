import polars as pl

ID_COL = "ReservableOptionMarketGroupId"

CATEGORICAL_COLS = [
    "MarketGroupCode", "BrandGroupCode", "AccoKindCode", "AccoTypeRangeCode",
    "SpecialPeriodCode", "SeasonalCluster", "CampsiteCluster", "CampsiteCountry",
    "CampsiteType", "AccommodationType", "AccommodationRange",
    "Kitchen", "DeckingType", "DeckingExtras",
]

NUMERIC_COLS = [
    "WeekBeforeArrival", "DiscountedPrice",
    "cumulative_booked_nights", "remaining_capacity", "occupancy_rate",
    "momentum_3w", "horizon_fraction",
    "Bedrooms", "Bathrooms", "Sleeps", "AvgTemperature", "latitude", "longitude",
    "price_per_sleep", "momentum_per_capacity", "occupancy_horizon_interaction",
    "capacity_gap_rate", "wba_sin", "wba_cos",
    "MarketGroupCode_te", "AccoTypeRangeCode_te", "CampsiteCluster_te",
]


def engineer_features(df: pl.DataFrame, fit_stats: dict | None = None) -> tuple[pl.DataFrame, dict]:
    """
    Engineer features for the booking prediction model.

    Args:
        df:         Input DataFrame (train fold or val/test).
        fit_stats:  Pass None when fitting on train — stats computed and returned.
                    Pass the returned stats dict when transforming val/test.

    Returns:
        (transformed_df, fit_stats)
    """
    fit_stats = fit_stats or {}

    # Sort: for each combo, WBA goes 52 → 0 (descending)
    df = df.sort([ID_COL, "WeekBeforeArrival"], descending=[False, True])

    # --- Booking curve features (safe: only use past rows within same combo) ---
    df = df.with_columns(
        pl.col("HistoricalBookedNights").shift(1).cum_sum().over(ID_COL).alias("cumulative_booked_nights")
    )
    df = df.with_columns(
        pl.col("cumulative_booked_nights").fill_null(0).alias("cumulative_booked_nights")
    )
    df = df.with_columns(
        (pl.col("Capacity") - pl.col("cumulative_booked_nights")).clip(0, None).alias("remaining_capacity")
    )
    df = df.with_columns(
        (pl.col("cumulative_booked_nights") / pl.col("Capacity")).alias("occupancy_rate")
    )
    df = df.with_columns(
        pl.col("HistoricalBookedNights").shift(1).rolling_sum(window_size=3).over(ID_COL).alias("momentum_3w")
    )
    df = df.with_columns(
        pl.col("momentum_3w").fill_null(0).alias("momentum_3w")
    )
    df = df.with_columns(
        (pl.col("WeekBeforeArrival") / 52.0).alias("horizon_fraction")
    )

    # --- Derived features (safe: all inputs are row-level or already computed above) ---
    df = df.with_columns([
        (pl.col("DiscountedPrice") / pl.col("Sleeps").clip(1, None)).alias("price_per_sleep"),
        (pl.col("momentum_3w") / (pl.col("remaining_capacity") + 1.0)).alias("momentum_per_capacity"),
        (pl.col("occupancy_rate") * (1.0 - pl.col("horizon_fraction"))).alias("occupancy_horizon_interaction"),
        (1.0 - pl.col("occupancy_rate")).alias("capacity_gap_rate"),
        (pl.col("WeekBeforeArrival") * 2 * 3.14159265 / 52.0).sin().alias("wba_sin"),
        (pl.col("WeekBeforeArrival") * 2 * 3.14159265 / 52.0).cos().alias("wba_cos"),
    ])

    # --- Target encoding (safe ONLY if fit_stats fitted on train fold, not full train) ---
    te_cols = ["MarketGroupCode", "AccoTypeRangeCode", "CampsiteCluster"]
    global_key = "global_target_mean"
    if global_key not in fit_stats:
        fit_stats[global_key] = float(df["HistoricalBookedNights"].mean())
    global_mean = float(fit_stats[global_key])

    for col in te_cols:
        stat_key = f"te_{col}"
        if stat_key not in fit_stats:
            mapping = {
                row[col]: float(row["te"])
                for row in df.group_by(col)
                    .agg(pl.col("HistoricalBookedNights").mean().alias("te"))
                    .iter_rows(named=True)
            }
            fit_stats[stat_key] = mapping
        mapping = fit_stats[stat_key]
        df = df.with_columns(
            pl.col(col)
            .map_elements(lambda x, m=mapping, g=global_mean: m.get(x, g), return_dtype=pl.Float64)
            .alias(f"{col}_te")
        )

    return df, fit_stats
