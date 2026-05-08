import joblib
import numpy as np
import polars as pl
import altair as alt
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from features import engineer_features
from price import apply_encodings, add_season_bucket

OUT_HTML = Path('figures/booking_curves_actual_vs_predicted.html')
OUT_PNG  = Path('figures/booking_curves_actual_vs_predicted.png')
SEASON_ORDER = ["Low", "Mid", "High", "Peak"]

model = joblib.load('logs/model.pkl')
fit_stats = joblib.load('logs/fit_stats.pkl')
cat_encoders = joblib.load('logs/cat_encoders.pkl')
feature_cols = joblib.load('logs/feature_cols.pkl')

test = pl.read_csv('data/processed/test.csv')
test, _ = engineer_features(test, fit_stats=fit_stats)
test = apply_encodings(test, cat_encoders)
test = add_season_bucket(test)

# Campsites that have all 4 seasons represented
site_seasons = (
    test.group_by('CampsiteCode')
    .agg(pl.col('season_bucket').n_unique().alias('n_seasons'), pl.len().alias('n_rows'))
    .filter(pl.col('n_seasons') == 4)
    .sort('n_rows', descending=True)
)
selected_campsites = site_seasons['CampsiteCode'].head(5).to_list()
if len(selected_campsites) < 5:
    raise RuntimeError('Could not select 5 campsites with all 4 seasons.')

records = []
for campsite in selected_campsites:
    cdf = test.filter(pl.col('CampsiteCode') == campsite)

    for season in SEASON_ORDER:
        sdf = cdf.filter(pl.col('season_bucket') == season)
        if sdf.is_empty():
            raise RuntimeError(f'Missing season {season} for campsite {campsite}')

        weeks = sdf.select('WeekStartDate').unique().sort('WeekStartDate').get_column('WeekStartDate').to_list()
        chosen_week = weeks[len(weeks)//2]

        wk_df = sdf.filter(pl.col('WeekStartDate') == chosen_week)
        unit_score = (
            wk_df.group_by('ReservableOptionMarketGroupId')
            .agg(pl.col('HistoricalBookedNights').sum().alias('tot_nights'))
            .sort('tot_nights', descending=True)
        )
        unit_id = unit_score['ReservableOptionMarketGroupId'][0]
        curve_df = wk_df.filter(pl.col('ReservableOptionMarketGroupId') == unit_id).sort('WeekBeforeArrival', descending=True)

        actual = curve_df['HistoricalBookedNights'].to_numpy()
        pred = model.predict(curve_df[feature_cols].to_numpy()).clip(0, None)

        for wba, a, p in zip(curve_df['WeekBeforeArrival'], actual, pred):
            records.append({'CampsiteCode': campsite, 'season_bucket': season,
                            'WeekBeforeArrival': int(wba), 'curve': 'Actual', 'booked_nights': float(a)})
            records.append({'CampsiteCode': campsite, 'season_bucket': season,
                            'WeekBeforeArrival': int(wba), 'curve': 'Predicted', 'booked_nights': float(p)})

plot_df = pl.DataFrame(records)
pdf = plot_df.to_pandas()
alt.data_transformers.disable_max_rows()

color_scale = alt.Scale(domain=['Actual', 'Predicted'], range=['#1f77b4', '#2ca02c'])

chart = (
    alt.Chart(pdf)
    .mark_line(strokeWidth=1.8)
    .encode(
        x=alt.X('WeekBeforeArrival:Q', scale=alt.Scale(reverse=True), title='WeekBeforeArrival'),
        y=alt.Y('booked_nights:Q', title='Booked nights'),
        color=alt.Color('curve:N', scale=color_scale, title='Curve')
    )
    .properties(width=200, height=120)
    .facet(
        row=alt.Row('CampsiteCode:N', sort=selected_campsites, header=alt.Header(title='Campsite')),
        column=alt.Column('season_bucket:N', sort=SEASON_ORDER, header=alt.Header(title='Season'))
    )
    .resolve_scale(y='independent')
)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
chart.save(str(OUT_HTML))

try:
    chart.save(str(OUT_PNG))
    print('saved_png', OUT_PNG)
except Exception as e:
    print('png_save_failed', type(e).__name__, str(e))

print('saved_html', OUT_HTML)
