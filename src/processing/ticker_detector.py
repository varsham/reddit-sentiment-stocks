import re

COMMON_WORDS = {
      "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
      "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON",
      "OR", "SO", "TO", "UP", "US", "WE", "ALL", "AND", "ARE",
      "BUT", "CAN", "FOR", "GET", "GOT", "HAD", "HAS", "HIM",
      "HIS", "HOW", "ITS", "NEW", "NOT", "NOW", "ONE", "OUR",
      "OUT", "OWN", "SAY", "SHE", "THE", "TOO", "TWO", "WAS",
      "WAY", "WHO", "WHY", "WILL", "WITH", "YOU",
      # finance-specific false positives
      "CEO", "IPO", "ETF", "GDP", "IMO", "EPS", "ATH", "DD",
      "YOY", "QOQ", "PE", "ER",
}

def post_mentions_ticker(ticker: str, title: str, body: str) -> bool:
    """
    Return True if the post (title or body) contains a reference to the 
    given ticker symbol.

    Matches:
    - $TICKER
    - TICKER

    ticker: uppercase string like "TSLA"
    title: post title string
    body: post selftext string
    """
    ticker = ticker.upper()

    pattern = rf'\${re.escape(ticker)}\b|\b{re.escape(ticker)}\b'
    text = title + " " + body
    return bool(re.search(pattern, text, re.IGNORECASE))

def extract_tickers_from_text(text: str) -> set[str]:
    """
    Extract ALL ticker-like tokens from text.
    Returns a set of uppercase strings that look like tickers and are not
    in COMMON_WORDS.

    This is a bonus/exploratory function - not used in the main pipeline for V1,
    but useful for notebooks.
    """
    tickers = set()
    pattern = r'\$?([A-Z]{1,5})\b'

    for match in re.finditer(pattern, text):
        token = match.group(1)
        if token not in COMMON_WORDS:
            tickers.add(token)
    return tickers