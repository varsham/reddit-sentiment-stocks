import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.collection.market_data import download_price_data, compute_forward_returns
from src.processing.feature_builder import load_scored_posts, aggregate_daily_sentiment
from src.analysis.correlation import pearson_and_spearman, lagged_correlations, run_granger_causality, summarize_results
from src.analysis.ml_models import run_all_models

DATA_DIR = Path("data")

st.set_page_config(page_title="Reddit Sentiment Stocks", layout="wide")
st.title("Reddit Sentiment vs. Stock Returns")

with st.sidebar:
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    start_date = st.date_input("Start date", value=pd.Timestamp("2023-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2024-01-01"))
    horizon = st.slider("Prediction horizon (trading days)", 1, 20, 5)
    run_btn = st.button("Run analysis", type="primary")

if not run_btn:
    st.info("Enter a ticker and click **Run analysis** to begin.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Price and Sentiment", "Correlations", "ML Models"])

with tab1:

    chart_data = pd.DataFrame(

    )

    pass

with tab2:

    pass

