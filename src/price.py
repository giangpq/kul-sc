"""
price.py — Multiplier model + CMA-ES pricing optimisation.

Phase 2 workflow:
  1. Tune shared policy parameters on calibration.csv (CMA-ES)
  2. Evaluate final policy on holdout.csv
  3. Log results to MLflow experiment 'campsite-pricing'

Parameters optimised (5 total):
  A1    : advance multiplier at arrival (WBA=0), < 1 (last-minute discount)
  A2    : advance multiplier at max horizon (WBA=52)
  A3    : advance multiplier peak, > 1
  t1    : WBA at which peak occurs
  yD1   : availability multiplier when empty (< 1)
  yD2 = 2 - yD1 derived from mean=1 constraint
"""

import sys
import warnings
from pathlib import Path

import cma
import joblib
import mlflow
import mlflow.tracking
import numpy as np
import polars as pl
from tqdm import tqdm


warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent))
from features import CATEGORICAL_COLS, engineer_features

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("logs")
ID_COL    = "ReservableOptionMarketGroupId"

MLFLOW_DEMAND_EXPERIMENT  = "campsite-demand"
MLFLOW_PRICING_EXPERIMENT = "campsite-pricing"

N_UNITS_OPT = 1000     # Sample units for optimization
N_MC_RUNS_OPT = 10    # Fewer runs for faster optimization
MAX_WBA = 52    # booking horizon in weeks
N_UNITS_EVAL = 10_000 # Sample units for final evaluation
N_MC_RUNS_EVAL = 10    # Full runs for final evaluation
MAXITER = 30

# 1.1% average lift in  revenue, 5.5% lift std 

# ---------------------------------------------------------------------------
# Multiplier model
# ---------------------------------------------------------------------------

def advance_multiplier(wba: float, A1: float, A2: float, A3: float, t1: float) -> float:
    """
    Piecewise linear advance multiplier.
      WBA in [t1, 52] : linearly interpolate between A2 (at 52) and A3 (at t1)
      WBA in [0,  t1] : linearly interpolate between A1 (at 0)  and A3 (at t1)
    """
    if wba >= t1:
        # from A2 at WBA=52 up to A3 at WBA=t1
        frac = (wba - t1) / max(MAX_WBA - t1, 1e-6)
        return A3 + frac * (A2 - A3)
    else:
        # from A3 at WBA=t1 down to A1 at WBA=0
        frac = wba / max(t1, 1e-6)
        return A1 + frac * (A3 - A1)


def availability_multiplier(occupancy_rate: float, yD1: float) -> float:
    """
    Linear availability multiplier.
      occupancy_rate = booked / capacity  (0=empty, 1=full)
      yD1: multiplier when empty (< 1, discount)
      yD2 = 2 - yD1: multiplier when full (> 1, premium) — from mean=1 constraint
    """
    yD2 = 2.0 - yD1
    return yD1 + (yD2 - yD1) * occupancy_rate


def compute_price(
    reference_price: float,
    wba: float,
    booked: float,
    capacity: float,
    params: np.ndarray,
) -> float:
    A1, A2, A3, t1, yD1 = params
    occ = min(booked / max(capacity, 1.0), 1.0)
    adv = advance_multiplier(wba, A1, A2, A3, t1)
    avl = availability_multiplier(occ, yD1)
    return reference_price * adv * avl


# ---------------------------------------------------------------------------
# Monte Carlo simulation for one unit
# ---------------------------------------------------------------------------

def simulate_unit_mc(
    model,
    unit_df: pl.DataFrame,
    params: np.ndarray | None,
    feature_cols: list[str],
    reference_price: float | None,
    variance_lookup: dict[tuple[str, int], float],
    fallback_variance: float = 1.0,
    n_runs: int = N_MC_RUNS_EVAL,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(42)

    unit_df     = unit_df.sort("WeekBeforeArrival", descending=True)
    wbas        = unit_df["WeekBeforeArrival"].to_numpy()
    seasons     = unit_df["season_bucket"].to_list()
    hist_prices = unit_df["DiscountedPrice"].to_numpy()
    capacity    = float(unit_df["Capacity"][0])
    feature_arr = unit_df[feature_cols].to_numpy().copy()  # (n_steps, n_features)

    idx = {col: feature_cols.index(col) for col in [
        "DiscountedPrice", "cumulative_booked_nights", "remaining_capacity",
        "occupancy_rate", "momentum_3w", "momentum_per_capacity",
        "occupancy_horizon_interaction", "capacity_gap_rate", "price_per_sleep",
    ] if col in feature_cols}

    sleeps = float(unit_df["Sleeps"][0]) if "Sleeps" in unit_df.columns else 1.0

    # --- vectorised state across n_runs ---
    booked    = np.zeros(n_runs)
    revenue   = np.zeros(n_runs)
    recent_3w = np.zeros((3, n_runs))  # rows = lag 1,2,3

    for step_idx, (wba, season) in enumerate(zip(wbas, seasons)):
        remaining = np.maximum(capacity - booked, 0.0)
        active    = remaining > 0  # mask: runs still open

        if not active.any():
            break

        occupancy    = booked / max(capacity, 1.0)
        horizon_frac = wba / 52.0
        momentum     = recent_3w.sum(axis=0)
        mom_per_cap  = momentum / (remaining + 1.0)
        occ_horiz    = occupancy * (1.0 - horizon_frac)
        cap_gap      = 1.0 - occupancy

        if params is None:
            prices = np.full(n_runs, float(hist_prices[step_idx]))
        else:
            if reference_price is None:
                raise ValueError("reference_price must be provided when params is not None")
            # price depends on booked (occupancy), which varies per run
            prices = np.array([
                compute_price(reference_price, wba, float(b), capacity, params)
                for b in booked
            ])

        # build (n_runs, n_features) batch
        batch = np.tile(feature_arr[step_idx], (n_runs, 1))
        if "DiscountedPrice"               in idx: batch[:, idx["DiscountedPrice"]]               = prices
        if "cumulative_booked_nights"      in idx: batch[:, idx["cumulative_booked_nights"]]      = booked
        if "remaining_capacity"            in idx: batch[:, idx["remaining_capacity"]]            = remaining
        if "occupancy_rate"                in idx: batch[:, idx["occupancy_rate"]]                = occupancy
        if "momentum_3w"                   in idx: batch[:, idx["momentum_3w"]]                   = momentum
        if "momentum_per_capacity"         in idx: batch[:, idx["momentum_per_capacity"]]         = mom_per_cap
        if "occupancy_horizon_interaction" in idx: batch[:, idx["occupancy_horizon_interaction"]] = occ_horiz
        if "capacity_gap_rate"             in idx: batch[:, idx["capacity_gap_rate"]]             = cap_gap
        if "price_per_sleep"               in idx: batch[:, idx["price_per_sleep"]]               = prices / max(sleeps, 1.0)

        preds = np.maximum(model.predict(batch), 0.0)  # single batch predict

        var   = variance_lookup.get((season, int(wba)), fallback_variance)
        noisy = np.maximum(rng.normal(preds, np.sqrt(var)), 0.0)
        booked_this_week = np.where(active, np.minimum(noisy, remaining), 0.0)

        revenue   += prices * booked_this_week
        booked    += booked_this_week
        recent_3w  = np.vstack([booked_this_week, recent_3w[:2]])

    return float(revenue.mean())


# ---------------------------------------------------------------------------
# CMA-ES objective
# ---------------------------------------------------------------------------

def objective(
    raw_params: list[float],
    model,
    unit_dfs: list[pl.DataFrame],
    feature_cols: list[str],
    reference_prices: dict[str, float],
    variance_lookup,
    fallback_variance,
    rng: np.random.Generator,
) -> float:
    """
    Negative mean revenue across all units in df.
    CMA-ES minimises this, so maximising revenue.
    """
    params = np.array(raw_params)

    # hard penalty if params violate logical constraints
    A1, A2, A3, t1, yD1 = params
    if not (0.3 < A1 < 1.0 and 0.5 < A2 < 1.5 and A3 > 1.0 and 2 < t1 < 50 and 0.3 < yD1 < 1.0):
        return 1e9

    revenues = []
    for unit_df in tqdm(unit_dfs, desc="units", leave=False):
        unit_id = unit_df[ID_COL][0]
        ref_price = reference_prices.get(unit_id, float(unit_df["DiscountedPrice"].median()))
        rev = simulate_unit_mc(model, unit_df, params, feature_cols, ref_price, variance_lookup, fallback_variance, n_runs=N_MC_RUNS_OPT, rng=rng)
        revenues.append(rev)

    return -float(np.mean(revenues))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_model_artifacts() -> tuple:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_DEMAND_EXPERIMENT)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_rmse ASC"],
        max_results=1,
    )
    best_run = runs[0]
    artifact_uri = best_run.info.artifact_uri

    model        = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/model.pkl"))
    fit_stats    = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/fit_stats.pkl"))
    cat_encoders = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/cat_encoders.pkl"))
    feature_cols = joblib.load(mlflow.artifacts.download_artifacts(artifact_uri + "/feature_cols.pkl"))

    print(f"Loaded best demand model: run_id={best_run.info.run_id}  cv_rmse={best_run.data.metrics['cv_rmse']:.6f}")
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
    ).drop("month")


def compute_reference_prices(df: pl.DataFrame) -> dict[str, float]:
    """Median historical price per unit as reference price."""
    return {
        row[ID_COL]: row["median_price"]
        for row in df.group_by(ID_COL)
            .agg(pl.col("DiscountedPrice").median().alias("median_price"))
            .iter_rows(named=True)
    }


def evaluate_policy(
    model,
    unit_dfs: list[pl.DataFrame],
    params: np.ndarray,
    feature_cols: list[str],
    reference_prices: dict[str, float],
    variance_lookup: dict[tuple[str, int], float],
    fallback_variance: float,
    n_runs: int = N_MC_RUNS_EVAL
) -> tuple[float, float]:
    """
    Compute revenue lift (%) of optimised policy vs. baseline (historical prices).
    Uses common random numbers per unit (same seed for baseline and policy)
    to reduce Monte Carlo comparison noise.
    Returns (mean_lift_pct, std_lift_pct).
    """
    lifts = []

    for i, unit_df in enumerate(tqdm(unit_dfs, desc="evaluating holdout")):
        unit_id   = unit_df[ID_COL][0]
        ref_price = reference_prices.get(unit_id, float(unit_df["DiscountedPrice"].median()))

        unit_seed = 10_000 + i

        # baseline: same simulator/state roll-forward, with historical row prices
        baseline_rev = simulate_unit_mc(
            model, unit_df, None, feature_cols, None,
            variance_lookup, fallback_variance,
            n_runs=n_runs,
            rng=np.random.default_rng(unit_seed),
        )

        opt_rev = simulate_unit_mc(
            model, unit_df, params, feature_cols, ref_price,
            variance_lookup, fallback_variance,
            n_runs=n_runs,
            rng=np.random.default_rng(unit_seed),
        )

        if baseline_rev > 0:
            lifts.append((opt_rev - baseline_rev) / baseline_rev * 100)

    return float(np.mean(lifts)), float(np.std(lifts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mlflow.set_experiment(MLFLOW_PRICING_EXPERIMENT)
    rng = np.random.default_rng(RANDOM_SEED := 42)

    # --- Load model ---
    model, fit_stats, cat_encoders, feature_cols, source_run_id = load_best_model_artifacts()

    # --- Load variance estimates ---

    variance_df = pl.read_csv(MODEL_DIR / "demand_variance.csv")
    variance_lookup = {
        (row["season_bucket"], row["WeekBeforeArrival"]): row["demand_variance"]
        for row in variance_df.iter_rows(named=True)
    }
    fallback_variance = float(variance_df["demand_variance"].mean())

    # --- Load and prepare calibration set ---
    print("Preparing calibration set ...")
    cal = pl.read_csv(DATA_DIR / "calibration.csv")
    cal, _ = engineer_features(cal, fit_stats=fit_stats)
    cal = apply_encodings(cal, cat_encoders)
    cal = add_season_bucket(cal)
    cal_ref_prices = compute_reference_prices(cal)

    # Pre-partition for speed and sample units for optimization
    cal_units = cal.partition_by(ID_COL, maintain_order=True)
    if len(cal_units) > N_UNITS_OPT:
        indices = rng.choice(len(cal_units), size=N_UNITS_OPT, replace=False)
        cal_units_opt = [cal_units[i] for i in indices]
    else:
        cal_units_opt = cal_units

    # --- CMA-ES optimisation on calibration ---
    # Initial params: [A1,  A2,   A3,   t1,   yD1 ]
    x0 = [0.85, 1.02, 1.08, 14.0, 0.85]
    sigma0 = 0.1 # smaller step size since bounds are tight

    print(f"Running CMA-ES on calibration set (sample of {len(cal_units_opt)} units) ...")
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        # Initial params: [A1,  A2,   A3,   t1,   yD1]
        "bounds": [[0.7, 0.85, 1.00,  2.0, 0.5],
                   [1.0, 1.30, 1.50, 50.0, 1.0]],
        "maxiter": MAXITER,
        "tolx": 1e-4,
        "seed": 42,
        "verbose": 1,
    })
    
    with tqdm(total=MAXITER, desc="CMA-ES iters") as pbar:
        while not es.stop():
            solutions = es.ask()
            fitvals = [
                objective(
                    sol, model, cal_units_opt, feature_cols, cal_ref_prices,
                    variance_lookup, fallback_variance, rng
                )
                for sol in solutions
            ]
            es.tell(solutions, fitvals)
            es.disp()

    best_params = np.array(es.result.xbest)
    print(f"\nBest params: A1={best_params[0]:.4f}  A2={best_params[1]:.4f}  "
          f"A3={best_params[2]:.4f}  t1={best_params[3]:.1f}  yD1={best_params[4]:.4f}")

    # --- Evaluate on holdout ---
    print("\nEvaluating on holdout set ...")
    holdout = pl.read_csv(DATA_DIR / "holdout.csv")
    holdout, _ = engineer_features(holdout, fit_stats=fit_stats)
    holdout = apply_encodings(holdout, cat_encoders)
    holdout = add_season_bucket(holdout)
    holdout_ref_prices = compute_reference_prices(holdout)
    holdout_units = holdout.partition_by(ID_COL, maintain_order=True)
    
    if len(holdout_units) > N_UNITS_EVAL:
        indices = rng.choice(len(holdout_units), size=N_UNITS_EVAL, replace=False)
        holdout_units_eval = [holdout_units[i] for i in indices]
    else:
        holdout_units_eval = holdout_units

    lift_mean, lift_std = evaluate_policy(
        model, holdout_units_eval, best_params, feature_cols,
        holdout_ref_prices, variance_lookup, fallback_variance, n_runs=N_MC_RUNS_EVAL
    )

    # --- Log to MLflow ---
    with mlflow.start_run():
        mlflow.log_param("source_model_run_id", source_run_id)
        mlflow.log_param("A1",  best_params[0])
        mlflow.log_param("A2",  best_params[1])
        mlflow.log_param("A3",  best_params[2])
        mlflow.log_param("t1",  best_params[3])
        mlflow.log_param("yD1", best_params[4])
        mlflow.log_param("n_mc_runs", N_MC_RUNS_EVAL)
        mlflow.log_param("n_units_opt", len(cal_units_opt))
        mlflow.log_metric("revenue_lift_mean_pct", lift_mean)
        mlflow.log_metric("revenue_lift_std_pct",  lift_std)

    print("---")
    print(f"revenue_lift_mean_pct: {lift_mean:.6f}")
    print(f"revenue_lift_std_pct:  {lift_std:.6f}")


if __name__ == "__main__":
    main()
