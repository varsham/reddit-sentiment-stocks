import json
import logging
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

LABEL_ORDER = ["positive", "negative", "neutral"]

def load_finbert(model_name: str = "ProsusAI/finbert"):
    """
    Load FinBERT tokenizer and model.
    Returns (tokenizer, model).
    Downloads on first call (~400MB), cached by HuggingFace after that.
    """

    # TODO: use AutoTokenizer.from_pretrained(model_name)
    # and AutoModelForSequenceClassification.from_pretrained(model_name)
    # then call model.eval() before returning
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()

    return (tokenizer, model)

def score_text(text: str, tokenizer, model) -> dict:
    """
    Run FinBERT on a single text string.
    Returns dict: {"positive": float, "negative": float, "neutral": float, "score": float}
    where score = P(positive) - P(negative)

    truncate input to 512 tokens (FinBERT's max)
    """
    # TODO: implement
    # Hint: tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # then model(**inputs) gives you logits
    # apply torch.nn.functional.softmax(logits, dim=1) to get probabilities
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        scores = probabilities[0].tolist()

        sentiment_dict = {
            "positive": scores[0],
            "negative": scores[1],
            "neutral": scores[2],
            "score": scores[0] - scores[1]
        }

        return sentiment_dict

def build_post_text(title: str, body: str) -> str:
    total_str = title

    if body and body != '[removed]':
        total_str += " " + body

    return total_str[:2000]

def score_posts(
        posts: list[dict],
        tokenizer,
        model,
        cache_path: Path,
) -> list[dict]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    # 2. For each post, check if post["id"] is in cache
    for post in posts:
        post_id = post["id"]
        if post_id not in cache:
            text = build_post_text(post["title"], post["selftext"])
            cache[post_id] = score_text(text, tokenizer, model)
        post["sentiment"] = cache[post_id]
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)

    return posts
