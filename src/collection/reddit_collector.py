# NOTE: Not currently used. All data in this project was sourced from the
# Kaggle WSB dataset via scripts/ingest_kaggle.py. This module is a
# live-collection alternative for pulling fresh posts directly via the
# Reddit API (requires REDDIT_* credentials in .env) — kept as a future
# option if the Kaggle snapshot needs to be extended with recent data.
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import praw
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def get_reddit_client():
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

    return reddit

def collect_ticker_posts(
        client: praw.Reddit,
        ticker: str,
        subreddits: list[str],
        posts_per_search: int,
        time_filter: str,
        output_dir: Path,
) -> int:
    """
    Search each subreddit for posts mentioning `ticker`. Save results 
    to output_dir/raw/{ticker}_{subreddit}_{date}.json.
    Return total number of posts collected.
    """
    all_subreddits = "+".join(subreddits)
    toSearch = client.subreddit(all_subreddits)

    file_path = output_dir / f"{ticker.lower()}_mentions.json"
    if file_path.exists():
        logger.info(f"Cache hit: {file_path}, skipping fetch")
        with open(file_path) as f:
            return len(json.load(f))

    executeSearch = toSearch.search(
        query=ticker,
        sort="new",
        time_filter=time_filter,
        limit=posts_per_search
    )
    # returns submission objects

    posts_list = []

    for submission in executeSearch:
        result_dict = post_to_dict(submission)
        posts_list.append(result_dict)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(posts_list, f, indent=4, ensure_ascii=False)

    return len(posts_list)

def post_to_dict(submission) -> dict:
    prawDict = {
        "id": submission.id,
        "title": submission.title,
        "selftext": submission.selftext,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
        "subreddit": str(submission.subreddit),
        "url": submission.url
    }

    return prawDict
