import polars as pl


def engineer_features(df: pl.DataFrame, fit_stats: dict | None = None) -> tuple[pl.DataFrame, dict]:
    """
    Engineer features for the booking prediction model.

    Args:
        df:         Input DataFrame (train, val, or test).
        fit_stats:  Dict of statistics fitted on training data (e.g. mean encodings).
                    Pass None when fitting on train — stats will be computed and returned.
                    Pass the returned stats dict when transforming val/test.

    Returns:
        (transformed_df, fit_stats)
    """
    fit_stats = fit_stats or {}

    # ------------------------------------------------------------------ #
    # BASELINE FEATURES — do not remove these                             #
    # ------------------------------------------------------------------ #

    df = df.sort([id_col, "WeekBeforeArrival"], descending=[False, True])

    # Cumulative booked nights so far (running total as WeekBeforeArrival decreases)
    df = df.with_columns(
        pl.col("HistoricalBookedNights")
          .cum_sum()
          .over(id_col)
          .alias("cumulative_booked_nights")
    )

    # Remaining capacity
    df = df.with_columns(
        (pl.col("Capacity") - pl.col("cumulative_booked_nights"))
          .clip(0, None)
          .alias("remaining_capacity")
    )

    # Occupancy rate so far
    df = df.with_columns(
        (pl.col("cumulative_booked_nights") / pl.col("Capacity"))
          .alias("occupancy_rate")
    )

    # Booking momentum: booked nights over last 3 weeks
    df = df.with_columns(
        pl.col("HistoricalBookedNights")
          .rolling_sum(window_size=3)
          .over(id_col)
          .alias("momentum_3w")
    )

    # Weeks remaining as a fraction of total horizon
    df = df.with_columns(
        (pl.col("WeekBeforeArrival") / 52.0).alias("horizon_fraction")
    )

    # ------------------------------------------------------------------ #
    # EXPERIMENTAL FEATURES — agent may freely add/modify below this line #
    # ------------------------------------------------------------------ #

    # (agent adds features here)

    return df, fit_stats


# Column names
id_col = "ReservableOptionMarketGroupId"

CATEGORICAL_COLS = [
    "MarketGroupCode",
    "BrandGroupCode",
    "AccoKindCode",
    "AccoTypeRangeCode",
    "SpecialPeriodCode",
    "SeasonalCluster",
    "CampsiteCluster",
    "CampsiteCountry",
    "CampsiteType",
    "AccommodationType",
    "AccommodationRange",
    "Kitchen",
    "DeckingType",
    "DeckingExtras",
]

NUMERIC_COLS = [
    "WeekBeforeArrival",
    "DiscountedPrice",
    "cumulative_booked_nights",
    "remaining_capacity",
    "occupancy_rate",
    "momentum_3w",
    "price_ratio_yoy",
    "horizon_fraction",
    "Bedrooms",
    "Bathrooms",
    "Sleeps",
    "AvgTemperature",
    "latitude",
    "longitude",
]
