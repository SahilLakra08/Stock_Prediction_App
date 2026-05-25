
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# These are the features we engineered in features.py
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA7', 'MA21', 'MA50', 'Daily_Return',
    'Volatility', 'RSI', 'BB_Upper', 'BB_Lower'
]

def prepare_data(df: pd.DataFrame):
    """
    Split data into training and testing sets.
    """
    X = df[FEATURES]
    y = df['Target']

    # No shuffle — order matters in time series data!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # CRITICAL: preserves temporal order
    )
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1        # use all CPU cores for faster training
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model and return metrics + predictions.
    """
    preds = model.predict(X_test)
    return {
        "MAE":  round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "R2":   round(r2_score(y_test, preds), 4),
        "predictions": preds,
        "actuals": y_test.values
    }


def predict_next_day(model, df: pd.DataFrame) -> float:
    """
    Predict tomorrow's closing price using the latest data row.
    """
    latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
    return round(float(model.predict(latest)[0]), 2)


def get_buy_sell_signal(current_price: float, predicted_price: float) -> str:
    """
    Simple buy/sell/hold signal based on prediction.
    """
    diff_pct = ((predicted_price - current_price) / current_price) * 100

    if diff_pct > 1.5:
        return "🟢 BUY"
    elif diff_pct < -1.5:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"
    