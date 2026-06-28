import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing.sentiment_analyzer import load_finbert, score_posts

CSV_PATH = Path("data/raw/reddit_wsb.csv")
OUTPUT_DIR = Path("data/processed")
TICKER = "GME"

def csv_to_posts(csv_path: Path, ticker: str) -> list[dict]:
    df = pd.read_csv(csv_path)

    # normalize column names
    df = df.rename(columns={
        "Title": "title",
        "body": "selftext",
        "comms_num": "num_comments"
    })

    # filter rows mentioning the ticker
    mask = (
        df["title"].str.contains(ticker, case=False, na=False) |
        df["selftext"].str.contains(ticker, case=False, na=False)
    )

    df = df[mask].copy()

    # normalize timestamp (if not present, just use 'created')
    if "timestamp" in df.columns:
        df["created_utc"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        df["created_utc"] = df["created"].apply(
            lambda ts: datetime.utcfromtimestamp(float(ts)).isoformat()
        )
    
    df["subreddit"] = "wallstreetbets"
    df["selftext"] = df["selftext"].fillna("")

    keep = ["id", "title", "selftext", "score", "num_comments", "created_utc", "subreddit", "url"]

    return df[keep].to_dict(orient="records")

if __name__ == "__main__":
    print(f"Loading {CSV_PATH}...")
    posts = csv_to_posts(CSV_PATH, TICKER)
    print(f"Found {len(posts)} posts mentioning {TICKER}")
    print("Loading FinBERT...")
    tokenizer, model = load_finbert()

    cache_path = OUTPUT_DIR / f"{TICKER.lower()}_sentiment_cache.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scoring posts (this will take a while)...")
    score = score_posts(posts, tokenizer, model, cache_path)

    out_path = OUTPUT_DIR / f"{TICKER.lower()}_mentions.json"
    with open(out_path, "w") as f:
        json.dump(score, f, indent=2)
    
    print(f"Done. {len(score)} posts saved to {out_path}")
