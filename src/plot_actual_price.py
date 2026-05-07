import polars as pl
import altair as alt
import numpy as np

cal = pl.read_csv("data/processed/calibration.csv")

# pick 5 random campsites
campsites = np.random.RandomState(42).choice(cal["CampsiteCode"].unique(), 5, replace=False)

# pick one stay week per season per campsite
seasons = {"Peak": [7,8], "High": [3,4,6,12], "Mid": [1,5,9,10], "Low": [2,11]}

rows = []
for cs in campsites:
    cs_df = cal.filter(pl.col("CampsiteCode") == cs)
    for season, months in seasons.items():
        subset = cs_df.filter(pl.col("WeekStartDate").str.slice(5,2).cast(int).is_in(months))
        if subset.is_empty(): continue
        week = subset.sample(1, seed=42)["WeekStartDate"][0]
        week_df = cs_df.filter(pl.col("WeekStartDate") == week)
        for r in week_df.iter_rows(named=True):
            rows.append({"campsite": cs[:10], "season": season, "price": r["DiscountedPrice"]})

df = pl.DataFrame(rows)

chart = alt.Chart(df).mark_boxplot().encode(
    x="season:N", y="price:Q", color="season:N", column="campsite:N"
).properties(width=150, height=200)

chart.save("figures/price_distribution.html")
