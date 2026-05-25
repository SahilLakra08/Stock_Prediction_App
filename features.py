

import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as ML features.
    """
    # Moving Averages
    df['MA7']  = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    # Volatility (rolling std)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff() 
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std)

    # Target: Next day's closing price (what we want to predict)
    df['Target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)
    return df
