"""
Model training script.
Agent may freely modify: model type, hyperparameters, feature selection, training loop.
"""
import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "HistoricalBookedNights"
MLFLOW_EXPERIMENT = "campsite-demand"


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


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

        train = train.with_columns(
            pl.col(col)
            .map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32)
            .alias(enc_col)
        )
        val = val.with_columns(
            pl.col(col)
            .map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int32)
            .alias(enc_col)
        )
    return train, val, encoders


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    val_rmse = mean_squared_error(y_true, y_pred) ** 0.5
    val_mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.abs(y_true).sum())
    val_wape = float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else 0.0
    return val_rmse, val_mae, val_wape


def tune_lgbm_optuna(X_train, y_train, X_val, y_val, base_params, trials: int):
    optuna_device = os.getenv("OPTUNA_LGBM_DEVICE", "cpu")

    def objective(trial):
        params = dict(base_params)
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 191),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
                "device": optuna_device,
            }
        )
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred = model.predict(X_val).clip(0, None)
            rmse, _, _ = compute_metrics(y_val, pred)
            return rmse
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False, catch=(Exception,))

    if len(study.trials) == 0 or study.best_trial.value is None or not np.isfinite(study.best_trial.value):
        return dict(base_params), float("inf")

    best_params = dict(base_params)
    best_params.update(study.best_params)
    return best_params, study.best_value


def train_lgbm(X_train, y_train, X_val, y_val, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )
    return model


def train_xgb(X_train, y_train, X_val, y_val, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_family = os.getenv("MODEL_FAMILY", "lgbm").lower()
    feature_set = os.getenv("FEATURE_SET", "baseline")

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

    run_params = {"model_family": model_family, "feature_set": feature_set}

    if model_family == "lgbm":
        params = dict(
            n_estimators=env_int("N_ESTIMATORS", 1200),
            learning_rate=env_float("LEARNING_RATE", 0.05),
            num_leaves=env_int("NUM_LEAVES", 63),
            min_child_samples=env_int("MIN_CHILD_SAMPLES", 20),
            subsample=env_float("SUBSAMPLE", 0.8),
            colsample_bytree=env_float("COLSAMPLE_BYTREE", 0.8),
            reg_alpha=env_float("REG_ALPHA", 0.0),
            reg_lambda=env_float("REG_LAMBDA", 0.0),
            objective="tweedie",
            tweedie_variance_power=env_float("TWEEDIE_VARIANCE_POWER", 1.3),
            random_state=42,
            n_jobs=12,
            device=os.getenv("LGBM_DEVICE", "gpu"),
        )

        optuna_trials = env_int("OPTUNA_TRIALS", 0)
        if optuna_trials > 0:
            tuned_params, optuna_best_rmse = tune_lgbm_optuna(
                X_train, y_train, X_val, y_val, params, trials=optuna_trials
            )
            params = tuned_params
            run_params["optuna_trials"] = optuna_trials
            run_params["optuna_best_rmse"] = optuna_best_rmse
            run_params["optuna_lgbm_device"] = os.getenv("OPTUNA_LGBM_DEVICE", "cpu")

        run_params.update(params)
        try:
            model = train_lgbm(X_train, y_train, X_val, y_val, params)
        except Exception:
            params["device"] = "cpu"
            model = train_lgbm(X_train, y_train, X_val, y_val, params)
            run_params["device_fallback"] = "cpu"

    elif model_family == "xgb":
        params = dict(
            n_estimators=env_int("N_ESTIMATORS", 900),
            learning_rate=env_float("LEARNING_RATE", 0.05),
            max_depth=env_int("MAX_DEPTH", 8),
            min_child_weight=env_float("MIN_CHILD_WEIGHT", 1.0),
            subsample=env_float("SUBSAMPLE", 0.8),
            colsample_bytree=env_float("COLSAMPLE_BYTREE", 0.8),
            reg_alpha=env_float("REG_ALPHA", 0.0),
            reg_lambda=env_float("REG_LAMBDA", 1.0),
            objective="reg:tweedie",
            tweedie_variance_power=env_float("TWEEDIE_VARIANCE_POWER", 1.3),
            random_state=42,
            tree_method="hist",
            device=os.getenv("XGB_DEVICE", "cuda"),
            n_jobs=12,
        )
        run_params.update(params)

        try:
            model = train_xgb(X_train, y_train, X_val, y_val, params)
        except Exception:
            params["device"] = "cpu"
            model = train_xgb(X_train, y_train, X_val, y_val, params)
            run_params["device_fallback"] = "cpu"
    else:
        raise ValueError(f"Unknown MODEL_FAMILY={model_family}")

    val_pred = model.predict(X_val).clip(0, None)
    val_rmse, val_mae, val_wape = compute_metrics(y_val, val_pred)

    with mlflow.start_run():
        mlflow.log_params(run_params)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_wape", val_wape)

        # Save artefacts
        joblib.dump(model, MODEL_DIR / "model.pkl")
        joblib.dump(fit_stats, MODEL_DIR / "fit_stats.pkl")
        joblib.dump(cat_encoders, MODEL_DIR / "cat_encoders.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

        mlflow.log_artifact(str(MODEL_DIR / "model.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "fit_stats.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "cat_encoders.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "feature_cols.pkl"))

    print("---")
    print(f"val_rmse:              {val_rmse:.6f}")
    print(f"val_mae:               {val_mae:.6f}")
    print(f"val_wape:              {val_wape:.6f}")


if __name__ == "__main__":
    main()
