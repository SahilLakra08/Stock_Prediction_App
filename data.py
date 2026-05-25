# Data Acquisition Module — Stock Prediction App
# Responsibilities:
#   1. Fetch historical OHLCV data from Yahoo Finance
#   2. Fetch company metadata (name, sector, price, etc.)
# Dependencies: yfinance, pandas

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    ticker: e.g. 'RELIANCE.NS', 'TCS.NS', 'AAPL'
    period: '1y', '2y', '5y', 'max'
    """
   
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic company info.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "currency": info.get("currency", "N/A"),
        "current_price": info.get("currentPrice", "N/A")
    }
