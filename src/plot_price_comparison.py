import altair as alt
import polars as pl
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.insert(0, "src")

from features import engineer_features
from price import (
    compute_price, apply_encodings, add_season_bucket,
    load_best_model_artifacts, ID_COL, DATA_DIR, MODEL_DIR,
)

model, fit_stats, cat_encoders, feature_cols, _ = load_best_model_artifacts()
best_params = np.array([0.5758, 1.4076, 2.9750, 15.4, 0.9950])

holdout = pl.read_csv(DATA_DIR / "holdout.csv")
holdout, _ = engineer_features(holdout, fit_stats=fit_stats)
holdout = apply_encodings(holdout, cat_encoders)
holdout = add_season_bucket(holdout)

# Pick 3 units of different seasons
sample_ids = (
    holdout.group_by(ID_COL)
    .agg(pl.col("season_bucket").first())
    .unique("season_bucket")
    .sample(3, seed=42)[ID_COL].to_list()
)

rows = []
for uid in sample_ids:
    udf = holdout.filter(pl.col(ID_COL) == uid).sort("WeekBeforeArrival", descending=True)
    cap = float(udf["Capacity"][0])
    ref = float(udf["DiscountedPrice"].median())
    for row in udf.iter_rows(named=True):
        wba, hist_price = row["WeekBeforeArrival"], row["DiscountedPrice"]
        # compute recommended price at median occupancy
        rec = compute_price(ref, wba, cap/2, cap, best_params)
        rows.append({"unit": uid[:20], "wba": int(wba), "type": "historical", "price": hist_price})
        rows.append({"unit": uid[:20], "wba": int(wba), "type": "recommended", "price": rec})

df = pl.DataFrame(rows)

chart = alt.Chart(df).mark_line().encode(
    x="wba:Q",
    y="price:Q",
    color="type:N",
    row="unit:N",
).properties(width=500, height=120)

chart.save("figures/price_comparison.html")
