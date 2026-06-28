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

    file_path = output_dir / f"{ticker.lower()}_{start_date}_{end_date}.csv"

    if (file_path.is_file()):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path)

    return df

def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    future_close = df["Close"].shift(-horizon)
    df[f'return_{horizon}d'] = (future_close/ df["Close"]) - 1

    return df