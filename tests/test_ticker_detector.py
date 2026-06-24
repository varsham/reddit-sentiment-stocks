from src.processing.ticker_detector import post_mentions_ticker, extract_tickers_from_text

def test_dollar_sign_match():
      assert post_mentions_ticker("TSLA", "$TSLA is going to the moon", "") is True


def test_standalone_uppercase_match():
    assert post_mentions_ticker("TSLA", "I bought TSLA today", "") is True


def test_no_match():
    assert post_mentions_ticker("TSLA", "I love electric cars", "") is False


def test_common_word_not_matched():
    # "IT" should not match ticker "IT" (Gartner) in normal English text
    # This one is genuinely hard — just document the limitation
    pass  # we'll discuss this edge case

def test_body_match():
    assert post_mentions_ticker("AAPL", "Thoughts?", "AAPL earnings coming up") is True


def test_case_insensitive_input():
    # ticker input "tsla" should still match "$TSLA" in text
    assert post_mentions_ticker("tsla", "$TSLA is up 5%", "") is True