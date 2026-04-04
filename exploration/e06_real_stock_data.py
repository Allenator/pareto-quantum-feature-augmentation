"""
Section 10: Real Stock Data — S&P 500 Excess Returns

Adapts the quantum feature augmentation framework to real financial data
with walk-forward backtesting.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Top S&P 500 stocks by market cap (starting set)
TOP_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSM"]
MARKET_INDEX = "^GSPC"  # S&P 500

TRAIN_WINDOW_YEARS = 2
FORWARD_RETURN_DAYS = 5


def download_data(tickers, start="2020-01-01", end="2025-01-01"):
    """Download OHLCV data for stocks and market index."""
    all_tickers = tickers + [MARKET_INDEX]
    data = yf.download(all_tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
    return data


def compute_excess_returns(stock_prices, market_prices, days=FORWARD_RETURN_DAYS):
    """Compute forward N-day excess returns (stock - market)."""
    stock_ret = stock_prices["Close"].pct_change(days).shift(-days)
    market_ret = market_prices["Close"].pct_change(days).shift(-days)
    return stock_ret - market_ret


def compute_features(stock_prices, market_prices):
    """
    Compute stock-minus-market features from price and volume data.

    Returns DataFrame with features aligned to stock_prices index.
    """
    features = pd.DataFrame(index=stock_prices.index)

    s_close = stock_prices["Close"]
    m_close = market_prices["Close"]
    s_vol = stock_prices["Volume"]
    m_vol = market_prices["Volume"]
    s_high = stock_prices["High"]
    s_low = stock_prices["Low"]

    # Price-related features (stock - market)
    features["ret_5d"] = s_close.pct_change(5) - m_close.pct_change(5)
    features["ret_20d"] = s_close.pct_change(20) - m_close.pct_change(20)
    features["ret_120d"] = s_close.pct_change(120) - m_close.pct_change(120)

    # Max/min price ratios
    features["max_high_20d"] = (s_high.rolling(20).max() / s_close - 1) - (
        market_prices["High"].rolling(20).max() / m_close - 1
    )
    features["min_low_20d"] = (s_low.rolling(20).min() / s_close - 1) - (
        market_prices["Low"].rolling(20).min() / m_close - 1
    )

    # RSI (10-day)
    def rsi(prices, period=10):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    features["rsi_10d"] = rsi(s_close) - rsi(m_close)

    # Price trend: MA10/MA50 - 1
    features["price_trend"] = (s_close.rolling(10).mean() / s_close.rolling(50).mean() - 1) - (
        m_close.rolling(10).mean() / m_close.rolling(50).mean() - 1
    )

    # Volume-related features
    def vol_zscore(vol, window):
        return (vol.rolling(window).mean() - vol.rolling(252).mean()) / vol.rolling(252).std()

    features["vol_z_5d"] = vol_zscore(s_vol, 5) - vol_zscore(m_vol, 5)
    features["vol_z_20d"] = vol_zscore(s_vol, 20) - vol_zscore(m_vol, 20)

    # Volume trend
    features["vol_trend"] = (s_vol.rolling(10).mean() / s_vol.rolling(50).mean() - 1) - (
        m_vol.rolling(10).mean() / m_vol.rolling(50).mean() - 1
    )

    # Cross features
    features["vol_trend_minus_price_trend"] = features["vol_trend"] - features["price_trend"]

    return features


def walk_forward_split(dates, train_window_years=TRAIN_WINDOW_YEARS):
    """
    Generate walk-forward train/test indices.

    Yields (train_mask, test_date) tuples rolling forward one day at a time.
    """
    train_window = timedelta(days=train_window_years * 365)
    min_date = dates.min() + train_window

    for test_date in dates[dates >= min_date]:
        train_start = test_date - train_window
        train_mask = (dates >= train_start) & (dates < test_date)
        yield train_mask, test_date


if __name__ == "__main__":
    print("Downloading S&P 500 data...")
    data = download_data(TOP_STOCKS[:3], start="2020-01-01", end="2025-01-01")

    ticker = TOP_STOCKS[0]
    print(f"\nBuilding features for {ticker}...")
    stock_data = data[ticker]
    market_data = data[MARKET_INDEX]

    features = compute_features(stock_data, market_data)
    target = compute_excess_returns(stock_data, market_data)

    # Drop NaN rows
    valid = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid]
    target = target.loc[valid]

    print(f"Feature matrix: {features.shape}")
    print(f"Target: {target.shape}")
    print(f"Date range: {valid.min()} to {valid.max()}")
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"\nTarget stats: mean={target.mean():.6f}, std={target.std():.6f}")
