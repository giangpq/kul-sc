"""
Model training script.
Agent may freely modify: model type, hyperparameters, feature selection, training loop.
"""
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "HistoricalBookedNights"
MLFLOW_EXPERIMENT = "campsite-demand"


def encode_categoricals(
    train: pl.DataFrame,
    val: pl.DataFrame,
    cat_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Label-encode categoricals using train-only mapping. No leakage."""
    encoders = {}
    for col in cat_cols:
        if col not in train.columns:
            continue
        unique_vals = sorted(train[col].drop_nulls().unique().to_list())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        encoders[col] = mapping
        enc_col = col + "_enc"
        for df_name, df in [("train", train), ("val", val)]:
            encoded = df.with_columns(
                pl.col(col)
                .map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32)
                .alias(enc_col)
            )
            if df_name == "train":
                train = encoded
            else:
                val = encoded
    return train, val, encoders


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load data
    train = pl.read_csv(DATA_DIR / "train.csv")
    val = pl.read_csv(DATA_DIR / "val.csv")

    # Engineer features — fit on train, apply to val
    train, fit_stats = engineer_features(train, fit_stats=None)
    val, _ = engineer_features(val, fit_stats=fit_stats)

    # Encode categoricals
    train, val, cat_encoders = encode_categoricals(train, val, CATEGORICAL_COLS)

    # Build feature matrix
    enc_cols = [c + "_enc" for c in CATEGORICAL_COLS if c in train.columns]
    feature_cols = [c for c in NUMERIC_COLS if c in train.columns] + enc_cols

    X_train = train[feature_cols].to_numpy()
    y_train = train[TARGET].to_numpy()
    X_val = val[feature_cols].to_numpy()
    y_val = val[TARGET].to_numpy()

    # --- Agent may tune all parameters below ---
    params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=12,
        device="gpu"
    )

    with mlflow.start_run():
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        # Evaluate
        val_pred = model.predict(X_val).clip(0, None)
        val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
        val_mae = mean_absolute_error(y_val, val_pred)

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)

        # Save artefacts
        joblib.dump(model,        MODEL_DIR / "model.pkl")
        joblib.dump(fit_stats,    MODEL_DIR / "fit_stats.pkl")
        joblib.dump(cat_encoders, MODEL_DIR / "cat_encoders.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

        mlflow.log_artifact(str(MODEL_DIR / "model.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "fit_stats.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "cat_encoders.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "feature_cols.pkl"))

    print("---")
    print(f"val_rmse:              {val_rmse:.6f}")
    print(f"val_mae:               {val_mae:.6f}")


if __name__ == "__main__":
    main()
