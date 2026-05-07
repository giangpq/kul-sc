import polars as pl
from pathlib import Path
import random
import numpy as np

RANDOM_SEED = 42

input_csv = Path("data/raw/simulation_output.csv")
out_dir = Path("data/processed")

def split_campsites(test_df: pl.DataFrame, seed: int = RANDOM_SEED) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split test set into calibration and holdout by campsite,
    stratified by CampsiteCountry. Writes calibration.csv and holdout.csv.
    Returns (calibration_df, holdout_df).
    """
    rng = random.Random(seed)

    campsites = (
        test_df.select(["CampsiteCode", "CampsiteCountry"])
        .unique()
        .sort(["CampsiteCountry", "CampsiteCode"])
    )

    calibration_sites, holdout_sites = [], []
    for country in campsites["CampsiteCountry"].unique().sort().to_list():
        sites = (
            campsites.filter(pl.col("CampsiteCountry") == country)
            ["CampsiteCode"].to_list()
        )
        rng.shuffle(sites)
        mid = len(sites) // 2
        calibration_sites.extend(sites[:mid])
        holdout_sites.extend(sites[mid:])

    calibration_df = test_df.filter(pl.col("CampsiteCode").is_in(calibration_sites))
    holdout_df     = test_df.filter(pl.col("CampsiteCode").is_in(holdout_sites))

    calibration_df.write_csv(out_dir / "calibration.csv")
    holdout_df.write_csv(out_dir / "holdout.csv")

    print(f"Calibration: {calibration_df['CampsiteCode'].n_unique()} campsites, {len(calibration_df):,} rows")
    print(f"Holdout:     {holdout_df['CampsiteCode'].n_unique()} campsites, {len(holdout_df):,} rows")

    return calibration_df, holdout_df


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    lf_all = pl.scan_csv(input_csv, infer_schema_length=10_000)

    lf_labeled = lf_all.with_columns([
        pl.col("WeekStartDate").cast(pl.Date, strict=False).dt.year().alias("year")
    ]).with_columns([
        pl.when(pl.col("year") == 2024).then(pl.lit("train"))
          .when(pl.col("year") == 2025).then(pl.lit("test"))
          .otherwise(pl.lit("drop"))
          .alias("split")
    ])

    lf_labeled.filter(pl.col("split") == "train").drop(["year", "split"]).sink_csv(out_dir / "train.csv")

    test_df = lf_labeled.filter(pl.col("split") == "test").drop(["year", "split"]).collect()
    test_df.write_csv(out_dir / "test.csv")

    split_campsites(test_df)


if __name__ == "__main__":
    main()
