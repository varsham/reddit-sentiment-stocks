import json
import pandas as pd
from pathlib import Path
import logging

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
    pass

def aggregate_daily_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group posts by date. For each date compute:
    - sentiment_mean: mean of sentiment_score across all posts that day
    - mention_volume: count of posts that day
    - sentiment_positive_mean: mean of positive probability
    - sentiment_negative_mean: mean of negative probability

    Return a DataFrame indexed by date, one row per day.
    """
    pass

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
    pass

