import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "sentiment_mean",
    "mention_volume",
    "sentiment_mean_3d",
    "sentiment_mean_7d",
    "sentiment_mean_lag_1",
    "sentiment_mean_lag_2",
    "sentiment_mean_lag_3",
]
TARGET_COL = "return_5d"

def time_series_split(df: pd.DataFrame, train_ratio:float = 0.8):
    """
    Split df into train and test sets preserving time order
    Returns (train_df, test_df).
    Do NOT shuffle
    """

    train_df, test_df = train_test_split(
        df, test_size=1 - train_ratio, random_state=42, shuffle=False
    )

    return (train_df, test_df)


def evaluate(y_true, y_pred, model_name: str) -> dict:
    """
    Compute evaluation metrics for a model's predictions.
    Returns dict with keys: model, rmse, r2, directional_accuracy

    directional_accuracy: fraction of predictions where the sign of the predicted
    return matches the sign of the actual return.
    This matters more than the RMSE for trading - being wrong about the direction
    is worse than being wrong about magnitude.
    """
    # np.sqrt(mean_square_error(y_true, y_pred)) for RMSE
    # np.sign(y_pred) == np.sign(y_true) for directional accuracy

    return_dict = {
        "model": model_name,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    }

    return return_dict

def run_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Fit LinearRegression on train_df, evaluate on test_df.
    Scale features with StandardScaler (fit on train only, transform both).
    Return evaluate() dict plus 'coefficients': dict mapping feature name to coefficient
    """
    model = LinearRegression()

    x_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    x_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    evaluateDict = evaluate(y_test, y_pred, "LinearRegression")
    evaluateDict.update({"coefficients": dict(zip(FEATURE_COLS, model.coef_))})

    return evaluateDict

def run_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 4,
) -> dict:
    """
    Fit RandomForestRegressor on train_df, evaluate on test_df.
    No scaling needed for tree models.
    Return evaluate() dict plus 'feature_importances': dict mapping
    feature name to importance score
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    x_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    x_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    evaluateDict = evaluate(y_test, y_pred, "RandomForestRegressor")
    evaluateDict.update({"feature_importances": dict(zip(FEATURE_COLS, model.feature_importances_))})

    return evaluateDict

def run_all_models(df: pd.DataFrame) -> list[dict]:
    """
    Run the full ML pipeline:
    1. Split into train/test
    2. Run linear regression
    3. Run random forest
    Return list of result dicts from each model, sorted by r2 descending
    """

    train_df, test_df = time_series_split(df)

    results = [
        run_linear_regression(train_df, test_df),
        run_random_forest(train_df, test_df),
    ]

    return sorted(results, key=lambda x: x["r2"], reverse=True)

