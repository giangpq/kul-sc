import time
import numpy as np
import polars as pl
import joblib
from pathlib import Path
import sys
sys.path.insert(0, "src")

from features import engineer_features
from price import (
    simulate_unit_mc, compute_reference_prices,
    apply_encodings, add_season_bucket,
    load_best_model_artifacts,
    ID_COL, DATA_DIR, MODEL_DIR,
    N_UNITS_OPT,
)


# --- Load artifacts ---
model, fit_stats, cat_encoders, feature_cols, _ = load_best_model_artifacts()

variance_df = pl.read_csv(MODEL_DIR / "demand_variance.csv")
variance_lookup = {
    (row["season_bucket"], row["WeekBeforeArrival"]): row["demand_variance"]
    for row in variance_df.iter_rows(named=True)
}
fallback_variance = float(variance_df["demand_variance"].mean())

# --- Load calibration ---
cal = pl.read_csv(DATA_DIR / "calibration.csv")
cal, _ = engineer_features(cal, fit_stats=fit_stats)
cal = apply_encodings(cal, cat_encoders)
cal = add_season_bucket(cal)
cal_ref_prices = compute_reference_prices(cal)
cal_units = cal.partition_by(ID_COL, maintain_order=True)

# --- Speed test on 30 units ---
test_units = cal_units[:30]
rng = np.random.default_rng(42)
x0 = np.array([0.80, 0.95, 1.25, 14.0, 0.85])

start = time.time()
revenues = []
for unit_df in test_units:
    unit_id = unit_df[ID_COL][0]
    ref_price = cal_ref_prices.get(unit_id, float(unit_df["DiscountedPrice"].median()))
    rev = simulate_unit_mc(
        model, unit_df, x0, feature_cols, ref_price,
        variance_lookup, fallback_variance, n_runs=10, rng=rng,
    )
    revenues.append(rev)

elapsed = time.time() - start
n_full = min(len(cal_units), N_UNITS_OPT)
est_per_iter = elapsed / 30 * n_full * 10  # ~10 CMA-ES candidates per iter
est_total = est_per_iter * 30              # maxiter=30

print(f"30 units in {elapsed:.1f}s")
print(f"Estimated time per CMA-ES iteration: {est_per_iter:.0f}s")
print(f"Estimated total CMA-ES (30 iters):   {est_total/60:.1f} min")
print(f"Mean revenue: {np.mean(revenues):.2f}")
print(f"Std revenue: {np.std(revenues):.2f}")
