import json
import pandas as pd
from pathlib import Path
import logging
from sentiment_analyzer import score_posts

logger = logging.getLogger(__name__)

def load_scored_posts(posts_path: Path) -> pd.DataFrame:
    """
    Load the scored posts JSON (output of score_posts from M3).
    Return a DataFrame with columns:
        id, created_utc, subreddit, title, score,
        sentiment_positive, sentiment_negative, sentiment_neutral
        sentiment_score
    
        Flatten the nested 'sentiment' dict into separate column
        Parse created_utc as a datetime and extract just the date
        into a 'date' column.
    """
    df = pd.read_json(posts_path)

    sentiment_df = pd.json_normalize(df['sentiment']).add_prefix('sentiment_')

    df = df.drop(columns=['sentiment']).join(sentiment_df)

    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
    df['date'] = df['created_utc'].dt.date

    return df

def aggregate_daily_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group posts by date. For each date compute:
    - sentiment_mean: mean of sentiment_score across all posts that day
    - mention_volume: count of posts that day
    - sentiment_positive_mean: mean of positive probability
    - sentiment_negative_mean: mean of negative probability

    Return a DataFrame indexed by date, one row per day.
    """
    daily_sentiment_df = df.groupby('date').agg(
        sentiment_mean=('sentiment_score', 'mean'),
        mention_volume=('posts', 'count'),
        sentiment_positive_mean=('sentiment_positive', 'mean'),
        sentiment_negative_mean=('sentiment_negative', 'mean')
    )

    return daily_sentiment_df

def build_features(
    daily_sentiment: pd.DataFrame,
    price_df: pd.DataFrame,
    rolling_windows: list[int] = [3, 7],
) -> pd.DataFrame:
    """
    Merge daily sentiment with price data. Then add:
    - Rolling sentiment averages for each window in rolling_windows
      e.g. sentiment_mean_3d, sentiment_mean_7d
    - Lag features: sentiment_mean_lag_1, sentiment_mean_lag_2,
    sentiment-mean_lag_3
    (sentiment from 1, 2, 3 days ago)

    Drop rows with NaN (from rolling/lag windows and forward return
    tail.)
    Return the final feature DataFrame.
    """
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    price_df.index = pd.to_datetime(price_df.index)

    daily_sentiment = daily_sentiment.sort_index()
    price_df = price_df.sort_index()

    merged_df = price_df.join(daily_sentiment, how="left")

    df_clean = merged_df.dropna()

    return df_clean
