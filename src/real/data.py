"""Data pipeline: Yahoo Finance download, caching, and feature engineering."""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.real.config import RealDataConfig

# Feature column names (order matters — must match compute_features output)
FEATURE_COLS = [
    "ret_5d", "ret_20d", "ret_120d",
    "high_20d", "low_20d",
    "rsi_10d", "price_trend",
    "vol_z_5d", "vol_z_20d", "vol_trend",
    "vwret_5d", "ret_x_vol", "rsi_x_vol", "vol_minus_price_trend",
]


def _cache_key(config: RealDataConfig) -> str:
    """Build a short hash from config for cache filenames."""
    key = f"{sorted(config.tickers)}_{config.market_ticker}_{config.start_date}_{config.end_date}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _download_single_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for one ticker from Yahoo Finance chart API.

    Returns DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date.
    Uses adjusted close prices.
    """
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": adj,  # adjusted close
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s").normalize(),
    )
    df.index.name = "Date"
    return df.dropna()


def download_prices(config: RealDataConfig) -> pd.DataFrame:
    """Download OHLCV data for all tickers + market. Cache as parquet.

    Returns a DataFrame with MultiIndex columns (ticker, field) where field is
    one of: Open, High, Low, Close, Volume.
    """
    cache_dir = Path(config.data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prices_{_cache_key(config)}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    all_tickers = list(config.tickers) + [config.market_ticker]
    frames = {}

    for ticker in all_tickers:
        for attempt in range(3):
            try:
                single = _download_single_ticker(ticker, config.start_date, config.end_date)
                if not single.empty:
                    frames[ticker] = single
                    print(f"  {ticker}: {len(single)} days")
                    break
            except Exception as e:
                print(f"  Attempt {attempt+1}/3 for {ticker}: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  WARNING: Failed to download {ticker} after 3 attempts")

    if not frames:
        raise RuntimeError("Failed to download any ticker data. Check network.")

    # Build MultiIndex DataFrame: (ticker, field)
    df = pd.concat(frames, axis=1)

    df.to_parquet(cache_path)
    return df


def _compute_rsi(close: pd.Series, window: int = 10) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_vol_zscore(volume: pd.Series, avg_window: int, zscore_window: int = 252) -> pd.Series:
    """Compute z-score of rolling average volume."""
    avg_vol = volume.rolling(window=avg_window, min_periods=avg_window).mean()
    roll_mean = avg_vol.rolling(window=zscore_window, min_periods=60).mean()
    roll_std = avg_vol.rolling(window=zscore_window, min_periods=60).std()
    return (avg_vol - roll_mean) / roll_std.replace(0, np.nan)


def _compute_single_features(close: pd.Series, high: pd.Series,
                              low: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Compute raw features for a single asset (before stock-market subtraction).

    Returns DataFrame indexed by date with columns matching FEATURE_COLS.
    """
    feats = pd.DataFrame(index=close.index)

    # Price returns
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_20d"] = close.pct_change(20)
    feats["ret_120d"] = close.pct_change(120)

    # High/low relative to current price
    feats["high_20d"] = high.rolling(20, min_periods=20).max() / close - 1
    feats["low_20d"] = low.rolling(20, min_periods=20).min() / close - 1

    # RSI
    feats["rsi_10d"] = _compute_rsi(close, 10)

    # Price trend: MA10 / MA50 - 1
    ma10 = close.rolling(10, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    feats["price_trend"] = ma10 / ma50 - 1

    # Volume z-scores
    feats["vol_z_5d"] = _compute_vol_zscore(volume, 5)
    feats["vol_z_20d"] = _compute_vol_zscore(volume, 20)

    # Volume trend
    vol_ma10 = volume.rolling(10, min_periods=10).mean()
    vol_ma50 = volume.rolling(50, min_periods=50).mean()
    feats["vol_trend"] = vol_ma10 / vol_ma50 - 1

    # Volume-weighted return minus actual return
    daily_ret = close.pct_change()
    weights = volume / volume.rolling(5, min_periods=5).sum()
    vwret = (weights * daily_ret).rolling(5, min_periods=5).sum()
    feats["vwret_5d"] = vwret - feats["ret_5d"]

    # Cross features
    feats["ret_x_vol"] = feats["ret_5d"] * feats["vol_z_5d"]
    feats["rsi_x_vol"] = feats["rsi_10d"] * feats["vol_z_5d"]
    feats["vol_minus_price_trend"] = feats["vol_trend"] - feats["price_trend"]

    return feats


def compute_features(prices: pd.DataFrame, ticker: str,
                     market: str, horizon: int) -> pd.DataFrame:
    """Compute stock-minus-market features and target for one ticker.

    Returns DataFrame with columns: date, ticker, FEATURE_COLS..., target.
    """
    # Extract OHLCV for stock and market
    stock_close = prices[(ticker, "Close")].dropna()
    stock_high = prices[(ticker, "High")].reindex(stock_close.index)
    stock_low = prices[(ticker, "Low")].reindex(stock_close.index)
    stock_vol = prices[(ticker, "Volume")].reindex(stock_close.index)

    mkt_close = prices[(market, "Close")].reindex(stock_close.index)
    mkt_high = prices[(market, "High")].reindex(stock_close.index)
    mkt_low = prices[(market, "Low")].reindex(stock_close.index)
    mkt_vol = prices[(market, "Volume")].reindex(stock_close.index)

    # Compute raw features for stock and market
    stock_feats = _compute_single_features(stock_close, stock_high, stock_low, stock_vol)
    mkt_feats = _compute_single_features(mkt_close, mkt_high, mkt_low, mkt_vol)

    # Stock minus market
    diff = stock_feats - mkt_feats

    # Target: 5-day forward excess return
    stock_fwd = stock_close.pct_change(horizon).shift(-horizon)
    mkt_fwd = mkt_close.pct_change(horizon).shift(-horizon)
    diff["target"] = stock_fwd - mkt_fwd

    diff["date"] = diff.index
    diff["ticker"] = ticker

    return diff


def build_dataset(config: RealDataConfig) -> pd.DataFrame:
    """Build the full panel dataset: all tickers stacked, features + target.

    Returns DataFrame with columns: date, ticker, FEATURE_COLS..., target.
    NaN rows (from lookback warmup and forward target shift) are dropped.
    """
    cache_dir = Path(config.data_dir)
    cache_path = cache_dir / f"dataset_{_cache_key(config)}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    prices = download_prices(config)

    panels = []
    for ticker in config.tickers:
        panel = compute_features(prices, ticker, config.market_ticker, config.prediction_horizon)
        panels.append(panel)

    df = pd.concat(panels, ignore_index=True)

    # Drop rows with any NaN (from lookback warmup or forward target shift)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Cache
    df.to_parquet(cache_path, index=False)

    return df
