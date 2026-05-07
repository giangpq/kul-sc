import polars as pl

train = pl.read_csv("data/processed/train.csv")
test  = pl.read_csv("data/processed/test.csv")

for name, df in [("train", train), ("test", test)]:
    ratios = (
        df.group_by("ReservableOptionMarketGroupId")
        .agg([
            pl.col("DiscountedPrice").median().alias("median_price"),
            pl.col("DiscountedPrice").max().alias("max_price"),
            pl.col("DiscountedPrice").min().alias("min_price"),
            pl.col("DiscountedPrice").filter(pl.col("WeekBeforeArrival") == 52).median().alias("price_wba52"),
        ])
        .with_columns([
            (pl.col("max_price") / pl.col("median_price")).alias("max_ratio"),
            (pl.col("min_price") / pl.col("median_price")).alias("min_ratio"),
            (pl.col("price_wba52") / pl.col("median_price")).alias("wba52_ratio"),
        ])
    )
    
    print(f"A3 upper bound (99th pctile max/median): {ratios['max_ratio'].quantile(0.99):.2f}")
    print(f"A1 lower bound (1st pctile  min/median): {ratios['min_ratio'].quantile(0.01):.2f}")
    print(f"A2 approx (median of WBA=52/median):     {ratios['wba52_ratio'].median():.2f}")