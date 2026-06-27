import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_scored_posts(posts_path: Path) -> pd.DataFrame:
    df = pd.read_json(posts_path)

    sentiment_df = pd.json_normalize(df['sentiment']).add_prefix('sentiment_')

    df = df.drop(columns=['sentiment']).join(sentiment_df)

    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
    df['date'] = df['created_utc'].dt.date

    return df

def aggregate_daily_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    daily_sentiment_df = posts_df.groupby('date').agg(
        sentiment_mean=('sentiment_score', 'mean'),
        mention_volume=('sentiment_score', 'count'),
        sentiment_positive_mean=('sentiment_positive', 'mean'),
        sentiment_negative_mean=('sentiment_negative', 'mean')
    )

    return daily_sentiment_df

def build_features(
    daily_sentiment: pd.DataFrame,
    price_df: pd.DataFrame,
    rolling_windows: list[int] = [3, 7],
) -> pd.DataFrame:
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    price_df.index = pd.to_datetime(price_df.index)

    daily_sentiment = daily_sentiment.sort_index()
    price_df = price_df.sort_index()

    merged_df = price_df.join(daily_sentiment, how="left")

    for window in rolling_windows:
        merged_df[f"sentiment_mean_{window}d"] = (
            merged_df["sentiment_mean"].rolling(window).mean()
        )

    for lag in [1, 2, 3]:
        merged_df[f"sentiment_mean_lag_{lag}"] = merged_df["sentiment_mean"].shift(lag)

    merged_df = merged_df.dropna()

    return merged_df
