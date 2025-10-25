import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    r = close.diff()
    up = r.clip(lower=0.0)
    dn = -r.clip(upper=0.0)
    rs = up.ewm(alpha=1/window, adjust=False).mean() / dn.ewm(alpha=1/window, adjust=False).mean()
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Basis-Returns & Volatilität
    px = df["Close"]
    out = df.copy()
    out["ret1"] = px.pct_change()
    out["vol_20"] = out["ret1"].ewm(span=20).std()

    # Momentum-Features
    out["mom_10"] = px.pct_change(10)
    out["mom_20"] = px.pct_change(20)

    # Gleitende Durchschnitte & Abweichung
    out["ema_20"] = _ema(px, 20)
    out["ema_50"] = _ema(px, 50)
    out["ema_spread"] = (out["ema_20"] / out["ema_50"] - 1.0)

    out["ma20"] = px.rolling(20).mean()
    out["price_z_20"] = (px / out["ma20"] - 1.0)

    # RSI / MACD / ATR
    out["rsi_14"] = _rsi(px, 14)
    macd, macds, mach = _macd(px, 12, 26, 9)
    out["macd"] = macd
    out["macd_signal"] = macds
    out["macd_hist"] = mach
    out["atr_14"] = _atr(out, 14)

    # Robuste Standardisierung ausgewählter Features
    for col in ["mom_10","mom_20","price_z_20","ema_spread"]:
        s = out[col]
        mu = s.rolling(252, min_periods=100).mean()
        sd = s.rolling(252, min_periods=100).std(ddof=0).replace(0, np.nan)
        out[col] = ((s - mu) / sd).fillna(0.0)

    return out
