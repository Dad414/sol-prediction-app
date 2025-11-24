"""Feature engineering utilities for SOL/USDT 30 minute candles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import StochRSIIndicator
from ta.trend import MACD


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    avg_gain = avg_gain.fillna(0)
    avg_loss = avg_loss.fillna(0)

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(window=period, min_periods=period).mean()


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower})


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with engineered features aligned to input index."""

    data = df.copy()

    # Basic Returns
    data["close_return_1"] = data["close"].pct_change()
    data["close_return_3"] = data["close"].pct_change(3)
    data["close_return_6"] = data["close"].pct_change(6)
    data["log_return_1"] = np.log(data["close"] / data["close"].shift(1))

    # Lag Features
    for lag in (1, 2, 3, 6, 12):
        data[f"close_return_1_lag_{lag}"] = data["close_return_1"].shift(lag)

    # Volatility (Rolling Std Dev)
    for window in (20, 50):
        data[f"volatility_{window}"] = data["close_return_1"].rolling(window=window).std()

    # Rate of Change (ROC)
    for period in (6, 12, 24):
        data[f"roc_{period}"] = data["close"].pct_change(period)

    # EMA
    for span in (5, 10, 20, 40):
        data[f"ema_{span}"] = _ema(data["close"], span)
        data[f"ema_distance_{span}"] = data["close"] / data[f"ema_{span}"] - 1

    # RSI
    data["rsi_14"] = _rsi(data["close"], period=14)
    data["rsi_6"] = _rsi(data["close"], period=6)
    data["rsi_24"] = _rsi(data["close"], period=24)

    # ATR
    data["atr_14"] = _atr(data, period=14)
    data["atr_pct"] = data["atr_14"] / data["close"]

    # Bollinger Bands
    bb = _bollinger_bands(data["close"], window=20)
    data = pd.concat([data, bb], axis=1)
    data["bb_position"] = (data["close"] - data["bb_mid"]) / (data["bb_upper"] - data["bb_lower"])

    # MACD
    macd = MACD(close=data["close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()

    # Stochastic RSI
    stoch_rsi = StochRSIIndicator(close=data["close"], window=14, smooth1=3, smooth2=3)
    data["stoch_k"] = stoch_rsi.stochrsi_k()
    data["stoch_d"] = stoch_rsi.stochrsi_d()

    # Volume Features
    data["volume_change_1"] = data["volume"].pct_change()
    data["volume_ma_10"] = data["volume"].rolling(window=10, min_periods=10).mean()
    data["volume_ma_ratio"] = data["volume"] / data["volume_ma_10"]

    # Time Features
    data["day_of_week"] = data["open_time"].dt.dayofweek
    data["hour"] = data["open_time"].dt.hour
    data["minute"] = data["open_time"].dt.minute

    data = data.dropna().reset_index(drop=True)
    return data


__all__ = ["compute_features"]


