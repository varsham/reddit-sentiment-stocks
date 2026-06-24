import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def download_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for ticker between start_date and end_date.
    Save to output_dir/{ticker_lower}_{start}_{end}.csv.
    Return the DataFrame.

    If file already exists, load and return it (cache).

    start_date, end_date: strings like "2023-01-01"
    """
    # TODO: implement
    # yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    # returns a DataFrame with columns: Open, High, Low, Close Volume
    # auto_adjust=True adjusts for splits and dividends automatically
    pass

def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add a column f'return_{horizon}d' to df.
    return_{horizon}d[t] = (Close[t+horizon] / Close[t]) - 1

    Rows where the forward return can't be computed (last `horizon` rows)
    will have NaN - that's correct, don't drop them here.

    Returns the modified DataFrame.
    """
    # TODO: implement
    # Hint: df["Close"].shift(-horizon) gives you the future price
    pass