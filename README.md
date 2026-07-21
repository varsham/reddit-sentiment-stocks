# Reddit Sentiment vs. Stock Returns

Does retail investor sentiment on r/wallstreetbets predict future stock returns? This project
scores historical WallStreetBets posts with FinBERT, aligns daily sentiment with price data, and
tests the relationship with correlation/causality analysis and ML models — all explorable through
an interactive Streamlit dashboard.

## Pipeline

1. **Ingest** — `scripts/ingest_kaggle.py` loads the [Kaggle WSB dataset](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)
   from `data/raw/reddit_wsb.csv`, filters posts mentioning a target ticker, and scores each with
   FinBERT (`src/processing/sentiment_analyzer.py`).
2. **Build features** — `scripts/build_features.py` aggregates daily sentiment
   (`src/processing/feature_builder.py`), pulls price history via yfinance
   (`src/collection/market_data.py`), and joins them into rolling/lagged sentiment features plus
   forward returns.
3. **Analyze** — `src/analysis/correlation.py` runs Pearson/Spearman correlation, lagged
   correlation, and Granger causality; `src/analysis/ml_models.py` benchmarks Linear Regression vs.
   Random Forest on directional accuracy for predicting 5-day forward returns.
4. **Dashboard** — `src/dashboard/app.py` (Streamlit) ties it all together per-ticker.

Note: `TICKER` in `ingest_kaggle.py`/`build_features.py` is a hardcoded constant edited by hand
per run, not a CLI arg.

### Live collection (unused / future option)

`src/collection/reddit_collector.py` pulls posts directly from the Reddit API via PRAW instead of
the static Kaggle snapshot. It's not part of the current pipeline — no data in this repo was
generated with it. It's kept as an option for extending the dataset with recent posts. Using it
requires Reddit API credentials in `.env` (see `.env.example`).

## Running it

```bash
.venv/bin/streamlit run src/dashboard/app.py
```

Data for TSLA and GME is already processed under `data/processed/`. To add a new ticker, edit the
`TICKER` constant in `ingest_kaggle.py` and `build_features.py`, run both, then relaunch the
dashboard.
