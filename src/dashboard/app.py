import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from src.analysis.correlation import pearson_and_spearman, lagged_correlations, run_granger_causality
from src.analysis.ml_models import run_all_models

DATA_DIR = Path("data/processed")

st.set_page_config(page_title="Reddit Sentiment Research", layout="wide")
st.title("Reddit Sentiment vs. Stock Returns")
st.caption("Does retail investor sentiment on WallStreetBets predict future stock returns?")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    available = sorted([p.stem.replace("_features", "").upper()
                        for p in DATA_DIR.glob("*_features.csv")])
    if not available:
        st.error("No feature data found. Run scripts/build_features.py first.")
        st.stop()

    ticker = st.selectbox("Ticker", available)
    horizon = st.slider("Prediction horizon (trading days)", 1, 20, 5)
    run_btn = st.button("Run analysis", type="primary")

if not run_btn:
    st.info("Select a ticker and click **Run analysis** to begin.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker.lower()}_features.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)

with st.spinner("Loading data..."):
    df = load_features(ticker)

if df.empty:
    st.error(f"No data found for {ticker}.")
    st.stop()

return_col = f"return_{horizon}d"
if return_col not in df.columns:
    df[return_col] = (df["Close"].shift(-horizon) / df["Close"]) - 1

df_clean = df.dropna(subset=[return_col, "sentiment_mean"])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Price & Sentiment", "Correlations", "ML Models"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("Avg Sentiment Score", f"{df['sentiment_mean'].mean():.3f}")
    col3.metric("Total Mentions", f"{int(df['mention_volume'].sum()):,}")

    st.subheader("Stock Price")
    st.line_chart(df[["Close"]])

    st.subheader("Daily Sentiment Score")
    st.caption("Positive = bullish, Negative = bearish  (FinBERT: P(positive) − P(negative))")
    st.line_chart(df[["sentiment_mean"]])

    st.subheader("Daily Mention Volume")
    st.bar_chart(df[["mention_volume"]])

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    with st.spinner("Running correlation analysis..."):
        ps      = pearson_and_spearman(df_clean, "sentiment_mean", return_col)
        lagged  = lagged_correlations(df_clean, "sentiment_mean", return_col)
        granger = run_granger_causality(df_clean, "sentiment_mean", return_col)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pearson r",  f"{ps['pearson_r']:+.4f}")
    col2.metric("Pearson p",  f"{ps['pearson_p']:.4f}")
    col3.metric("Spearman r", f"{ps['spearman_r']:+.4f}")
    col4.metric("Spearman p", f"{ps['spearman_p']:.4f}")

    st.subheader("Lagged Correlations")
    st.caption("Does sentiment from N days ago predict returns today?")
    st.dataframe(lagged, use_container_width=True)
    st.bar_chart(lagged.set_index("lag")["pearson_r"])

    st.subheader("Granger Causality")
    g1, g2, g3 = st.columns(3)
    g1.metric("Best lag", granger["lag"])
    g2.metric("F-statistic", f"{granger['f_stat']:.4f}")
    g3.metric("p-value", f"{granger['p_value']:.4f}")

    if granger["significant"]:
        st.success("Significant at p<0.05: past sentiment improves prediction of future returns beyond past returns alone.")
    else:
        st.warning("Not significant: sentiment does not appear to Granger-cause returns in this sample.")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    with st.spinner("Training models..."):
        results = run_all_models(df_clean)

    results_df = pd.DataFrame(results)[["model", "r2", "rmse", "directional_accuracy"]]
    results_df.columns = ["Model", "R²", "RMSE", "Directional Accuracy"]

    st.subheader("Model Performance")
    st.dataframe(results_df, use_container_width=True)
    st.bar_chart(results_df.set_index("Model")["Directional Accuracy"])

    st.subheader("What does directional accuracy mean?")
    st.write(
        f"How often the model correctly predicted whether the stock went **up or down** "
        f"over the next {horizon} trading days. "
        "A random guess gives 50%. Anything above ~55% sustained would be meaningful in practice. "
        "This does not account for transaction costs or position sizing."
    )
