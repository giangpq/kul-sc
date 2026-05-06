"""
experiment.py — CV-based model experimentation with TimeSeriesSplit.
Modify freely during Phase 1. Do not use for final retraining (use train.py).
"""
import warnings
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "HistoricalBookedNights"
MLFLOW_EXPERIMENT = "campsite-demand"
N_SPLITS = 5


def encode_categoricals(train, val, cat_cols):
    encoders = {}
    for col in cat_cols:
        if col not in train.columns:
            continue
        unique_vals = sorted(train[col].drop_nulls().unique().to_list())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        encoders[col] = mapping
        enc_col = col + "_enc"
        train = train.with_columns(
            pl.col(col).map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32).alias(enc_col)
        )
        val = val.with_columns(
            pl.col(col).map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32).alias(enc_col)
        )
    return train, val, encoders


def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.abs(y_true).sum())
    wape = float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else 0.0
    return rmse, mae, wape


def fit_model(X_train, y_train, X_val, y_val, model_family, params):
    if model_family == "lgbm":
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        )
    elif model_family == "xgb":
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        raise ValueError(f"Unknown MODEL_FAMILY={model_family}")
    return model


def main():
    import os
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_family = os.getenv("MODEL_FAMILY", "lgbm").lower()

    if model_family == "lgbm":
        params = dict(
            n_estimators=1800, learning_rate=0.03, num_leaves=127,
            min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="tweedie", tweedie_variance_power=1.3,
            random_state=42, n_jobs=12,
            device=os.getenv("LGBM_DEVICE", "gpu"),
        )
    elif model_family == "xgb":
        params = dict(
            n_estimators=900, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:tweedie", tweedie_variance_power=1.3,
            random_state=42, tree_method="hist",
            device=os.getenv("XGB_DEVICE", "cuda"),
            n_jobs=12,
        )

    print(f"model_family: {model_family}")
    print(f"params:       {params}")
    print(f"n_splits:     {N_SPLITS}")
    print("---")

    train_full = pl.read_csv(DATA_DIR / "train.csv")

    stay_weeks = (
        train_full
        .select(pl.col("WeekStartDate").cast(pl.Date, strict=False).alias("week_start"))
        .unique().sort("week_start")
        .get_column("week_start").to_list()
    )
    stay_weeks = np.array(stay_weeks)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_rmses, fold_maes, fold_wapes = [], [], []
    last_model = last_fit_stats = last_cat_encoders = last_feature_cols = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(stay_weeks)):
        train_weeks = set(stay_weeks[train_idx].tolist())
        val_weeks = set(stay_weeks[val_idx].tolist())

        train_fold = train_full.filter(
            pl.col("WeekStartDate").cast(pl.Date, strict=False).is_in(list(train_weeks))
        )
        val_fold = train_full.filter(
            pl.col("WeekStartDate").cast(pl.Date, strict=False).is_in(list(val_weeks))
        )

        train_fold, fit_stats = engineer_features(train_fold, fit_stats=None)
        val_fold, _ = engineer_features(val_fold, fit_stats=fit_stats)

        train_fold, val_fold, cat_encoders = encode_categoricals(train_fold, val_fold, CATEGORICAL_COLS)

        enc_cols = [c + "_enc" for c in CATEGORICAL_COLS if c in train_fold.columns]
        feature_cols = [c for c in NUMERIC_COLS if c in train_fold.columns] + enc_cols

        X_train = train_fold[feature_cols].to_numpy()
        y_train = train_fold[TARGET].to_numpy()
        X_val = val_fold[feature_cols].to_numpy()
        y_val = val_fold[TARGET].to_numpy()

        try:
            model = fit_model(X_train, y_train, X_val, y_val, model_family, params)
        except Exception:
            params["device"] = "cpu"
            model = fit_model(X_train, y_train, X_val, y_val, model_family, params)

        pred = model.predict(X_val).clip(0, None)
        rmse, mae, wape = compute_metrics(y_val, pred)
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_wapes.append(wape)
        print(f"Fold {fold+1}: rmse={rmse:.4f}  mae={mae:.4f}  wape={wape:.4f}")

        last_model, last_fit_stats, last_cat_encoders, last_feature_cols = (
            model, fit_stats, cat_encoders, feature_cols
        )

    cv_rmse = float(np.mean(fold_rmses))
    cv_mae = float(np.mean(fold_maes))
    cv_wape = float(np.mean(fold_wapes))

    with mlflow.start_run():
        mlflow.log_params({"model_family": model_family, "n_splits": N_SPLITS, **params})
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_wape", cv_wape)

        joblib.dump(last_model, MODEL_DIR / "model.pkl")
        joblib.dump(last_fit_stats, MODEL_DIR / "fit_stats.pkl")
        joblib.dump(last_cat_encoders, MODEL_DIR / "cat_encoders.pkl")
        joblib.dump(last_feature_cols, MODEL_DIR / "feature_cols.pkl")

        mlflow.log_artifact(str(MODEL_DIR / "model.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "fit_stats.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "cat_encoders.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "feature_cols.pkl"))

    print("---")
    print(f"cv_rmse:               {cv_rmse:.6f}")
    print(f"cv_mae:                {cv_mae:.6f}")
    print(f"cv_wape:               {cv_wape:.6f}")


if __name__ == "__main__":
    main()
