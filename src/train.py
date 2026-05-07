"""
train.py — Final retraining on full train.csv using best experiment config.
Run once after Phase 1 is complete and approved. Do not modify during experiments.
"""
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "HistoricalBookedNights"
MLFLOW_EXPERIMENT = "campsite-demand"


def load_best_experiment_params() -> tuple[str, dict]:
    """Return (model_family, params) from the MLflow run with lowest cv_rmse."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in MLflow experiment.")

    best_run = runs[0]
    print(f"Best run id:  {best_run.info.run_id}")
    print(f"Best cv_rmse: {best_run.data.metrics.get('cv_rmse'):.6f}")

    run_params = best_run.data.params
    model_family = run_params.pop("model_family")
    # Convert numeric strings back to appropriate types
    parsed = {}
    for k, v in run_params.items():
        if k == "n_splits":
            continue  # not a model param
        try:
            parsed[k] = int(v)
        except ValueError:
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
    return model_family, parsed


def encode_categoricals(df: pl.DataFrame, cat_cols: list[str], encoders: dict) -> pl.DataFrame:
    for col in cat_cols:
        if col not in df.columns:
            continue
        mapping = encoders.get(col, {})
        df = df.with_columns(
            pl.col(col).map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32).alias(col + "_enc")
        )
    return df


def build_encoders(df: pl.DataFrame, cat_cols: list[str]) -> dict:
    encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        unique_vals = sorted(df[col].drop_nulls().unique().to_list())
        encoders[col] = {v: i for i, v in enumerate(unique_vals)}
    return encoders


def main():
    model_family, params = load_best_experiment_params()
    print(f"model_family: {model_family}")
    print(f"params:       {params}")
    print("---")
    print("Retraining on full train.csv ...")

    train_full = pl.read_csv(DATA_DIR / "train.csv")

    # Engineer features — fit on full training data
    train_full, fit_stats = engineer_features(train_full, fit_stats=None)

    # Encode categoricals — fit on full training data
    cat_encoders = build_encoders(train_full, CATEGORICAL_COLS)
    train_full = encode_categoricals(train_full, CATEGORICAL_COLS, cat_encoders)

    enc_cols = [c + "_enc" for c in CATEGORICAL_COLS if c in train_full.columns]
    feature_cols = [c for c in NUMERIC_COLS if c in train_full.columns] + enc_cols

    X_train = train_full[feature_cols].to_numpy()
    y_train = train_full[TARGET].to_numpy()

    if model_family == "lgbm":
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
    elif model_family == "xgb":
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_family={model_family}")

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(fit_stats, MODEL_DIR / "fit_stats.pkl")
    joblib.dump(cat_encoders, MODEL_DIR / "cat_encoders.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    print("Saved artifacts to logs/:")
    print("  model.pkl, fit_stats.pkl, cat_encoders.pkl, feature_cols.pkl")
    print("Ready for price.py.")


if __name__ == "__main__":
    main()
