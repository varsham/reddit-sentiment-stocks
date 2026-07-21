import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
import logging

logger = logging.getLogger(__name__)

def pearson_and_spearman(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    return_col: str = "return_5d",
) -> dict:

    df = df.dropna()

    if len(df) < 2:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }

    pearsonR, pearsonP = stats.pearsonr(df[sentiment_col], df[return_col])
    spearmanR, spearmanP = stats.spearmanr(df[sentiment_col], df[return_col])

    return_dict = {
        "pearson_r": round(float(pearsonR), 4),
        "pearson_p": round(float(pearsonP), 4),
        "spearman_r": round(float(spearmanR), 4),
        "spearman_p": round(float(spearmanP), 4)
    }

    return return_dict

def lagged_correlations(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    return_col: str = "return_5d",
    max_lag: int = 10,
) -> pd.DataFrame:
    row_list = []

    isSignif = False

    # Shifting by l leaves len(df) - l usable rows; pearsonr needs at least 2.
    effective_max_lag = min(max_lag, max(len(df) - 2, 0))

    for l in range(1, effective_max_lag + 1):
        df_lagged = df.copy()
        df_lagged[sentiment_col] = df[sentiment_col].shift(l)
        spearman_pearson_dict = pearson_and_spearman(df_lagged, sentiment_col, return_col)
        currPearsonR = spearman_pearson_dict["pearson_r"]
        currPearsonP = spearman_pearson_dict["pearson_p"]

        if (currPearsonP < 0.05):
            isSignif = True

        row_list.append({"lag": l, "pearson_r": currPearsonR, "pearson_p": currPearsonP, "significant": isSignif})
        isSignif = False
    
    returnDf = pd.DataFrame(row_list, columns=["lag", "pearson_r", "pearson_p", "significant"])

    return returnDf
        

def run_granger_causality(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_mean",
    return_col: str = "return_5d",
    max_lag: int = 5,
) -> dict:
    df = df[[return_col, sentiment_col]].dropna()

    # grangercausalitytests (with its default addconst=True) requires
    # nobs > 3 * maxlag + 1, i.e. maxlag <= (nobs - 1) // 3 - 1.
    safe_max_lag = (len(df) - 1) // 3 - 1
    effective_max_lag = min(max_lag, safe_max_lag)

    if effective_max_lag < 1:
        return {
            "lag": None,
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "significant": False,
        }

    results = grangercausalitytests(df, effective_max_lag)

    min_lag, best_p, best_f = None, float("inf"), None

    for lag in range(1, effective_max_lag + 1):
        p = results[lag][0]['ssr_ftest'][1]
        f = results[lag][0]['ssr_ftest'][0]
        if p < best_p:
            best_p, best_f, min_lag = p, f, lag

    return_dict = {
        "lag": min_lag,
        "f_stat": round(float(best_f), 4),
        "p_value": round(float(best_p), 4),
        "significant": best_p < 0.05
    }

    return return_dict  

def summarize_results(
    pearson_spearman: dict,
    lagged: pd.DataFrame,
     granger: dict,  
  ) -> str:
    sig = lambda p: "YES" if p < 0.05 else "no"
    lines = [ "=== Sentiment-Return Analysis ===", f"Pearson  r={pearson_spearman['pearson_r']:+.4f} p={pearson_spearman['pearson_p']:.4f} significant={sig(pearson_spearman['pearson_p'])}",
    f"Spearman r={pearson_spearman['spearman_r']:+.4f} p={pearson_spearman['spearman_p']:.4f} significant={sig(pearson_spearman['spearman_p'])}",
    f"Granger  best_lag={granger['lag']}  F={granger['f_stat']:.4f} p={granger['p_value']:.4f}  significant={sig(granger['p_value'])}",
    f"Lagged: {lagged['significant'].sum()} of {len(lagged)} lags significant at p<0.05",
]
  
    return "\n".join(lines)