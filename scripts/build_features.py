import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collection.market_data import download_price_data, compute_forward_returns
from src.processing.feature_builder import load_scored_posts, aggregate_daily_sentiment, build_features

TICKER     = "GME"
START_DATE = "2020-01-01"
END_DATE   = "2022-12-31"

posts_df = load_scored_posts(Path(f"data/processed/{TICKER.lower()}_mentions.json"))
print(f"Loaded {len(posts_df)} posts")

daily = aggregate_daily_sentiment(posts_df)
print(f"Daily sentiment: {len(daily)} days")

prices = download_price_data(TICKER, START_DATE, END_DATE, Path("data/market"))
prices = compute_forward_returns(prices, horizon=5)
print(f"Price data: {len(prices)} days")

features = build_features(daily, prices)
print(f"Feature DataFrame: {len(features)} rows, {len(features.columns)} columns")
print(features.head())

out = Path(f"data/processed/{TICKER.lower()}_features.csv")
features.to_csv(out)
print(f"Saved to {out}")
