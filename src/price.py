"""
Pricing strategy script.
Agent may freely modify: price grid range, step size, optimisation logic,
future-week weighting.
The simulate_revenue() function imported from src/prepare.py is LOCKED — do not modify.
"""
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.tracking
import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, engineer_features
from prepare import build_price_grids, simulate_revenue

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
ID_COL = "ReservableOptionMarketGroupId"
MLFLOW_EXPERIMENT = "campsite-pricing"
MLFLOW_PRICING_EXPERIMENT = "campsite-pricing"


def load_best_model_artifacts(experiment_name: str) -> tuple:
    """Load artefacts from the best Phase 1 run (lowest val_rmse)."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"],
        max_results=1,
    )
    best_run = runs[0]
    artifact_uri = best_run.info.artifact_uri

    model        = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/model.pkl"))
    fit_stats    = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/fit_stats.pkl"))
    cat_encoders = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/cat_encoders.pkl"))
    feature_cols = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/feature_cols.pkl"))

    print(f"Loaded best model: run_id={best_run.info.run_id}  val_rmse={best_run.data.metrics['val_rmse']:.6f}")
    return model, fit_stats, cat_encoders, feature_cols, best_run.info.run_id


def apply_encodings(df: pl.DataFrame, cat_encoders: dict) -> pl.DataFrame:
    for col, mapping in cat_encoders.items():
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32)
                .alias(col + "_enc")
            )
    return df


def main():
    mlflow.set_experiment(MLFLOW_PRICING_EXPERIMENT)

    # Load best Phase 1 model
    model, fit_stats, cat_encoders, feature_cols, source_run_id = load_best_model_artifacts(MLFLOW_EXPERIMENT)

    # Load and prepare val data
    val = pl.read_csv(DATA_DIR / "val.csv")
    val, _ = engineer_features(val, fit_stats=fit_stats)
    val = apply_encodings(val, cat_encoders)

    with mlflow.start_run():
        mlflow.log_param("source_model_run_id", source_run_id)

        # --- Baseline revenue: model predictions at current prices ---
        X = val[feature_cols].to_numpy()
        predicted = model.predict(X).clip(0, val["remaining_capacity"].to_numpy())
        baseline_revenue_per_row = val["DiscountedPrice"].to_numpy() * predicted

        baseline_total = (
            val
            .with_columns(pl.Series("baseline_revenue", baseline_revenue_per_row))
            .group_by(ID_COL)
            .agg(pl.col("baseline_revenue").sum())
        )

        # --- Optimised revenue: simulate over price grid ---
        price_grids = build_price_grids(val)

        sim = simulate_revenue(
            model=model,
            df=val,
            price_grids=price_grids,
            feature_cols=feature_cols,
            capacity_col="remaining_capacity",
        )

        # For each (accommodation, week), pick the price with highest revenue
        best = (
            sim.sort("simulated_revenue", descending=True)
            .group_by([ID_COL, "WeekBeforeArrival"])
            .first()
        )

        optimised_total = (
            best
            .group_by(ID_COL)
            .agg(pl.col("simulated_revenue").sum().alias("optimised_revenue"))
        )

        # --- Revenue lift ---
        combined = baseline_total.join(optimised_total, on=ID_COL)
        combined = combined.with_columns(
            ((pl.col("optimised_revenue") - pl.col("baseline_revenue"))
             / pl.col("baseline_revenue") * 100)
            .alias("revenue_lift_pct")
        )

        lift_mean = combined["revenue_lift_pct"].mean()
        lift_std  = combined["revenue_lift_pct"].std()

        mlflow.log_metric("revenue_lift_mean_pct", lift_mean)
        mlflow.log_metric("revenue_lift_std_pct", lift_std)

        print("---")
        print(f"revenue_lift_mean_pct: {lift_mean:.6f}")
        print(f"revenue_lift_std_pct:  {lift_std:.6f}")


if __name__ == "__main__":
    main()
