"""Local CSV data + OHLCV resampling.

Reads committed real gold history (futures and a long XAUUSD spot series) and
resamples it to whatever timeframe the strategy runs on. No network needed, so
the backtest/search runs identically locally and in CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .adapter import DataError, normalize_ohlcv

# Repo-root-relative location of the committed data (this file is at
# src/gold_bot/data/local.py -> repo root is parents[3]).
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Our timeframe vocabulary -> pandas resample rule.
_RULE = {"5min": "5min", "15min": "15min", "30min": "30min", "60min": "60min", "1h": "60min", "1d": "1D"}

# Known committed datasets: symbol -> (filename, price_scale, native_timeframe).
# price_scale multiplies raw OHLC to get USD (the XAUUSD set is stored x100).
_FILES = {
    "mgc": ("mgc_5m.csv", 1.0, "5min"),            # real MGC=F futures, ~2 months, 2026
    "xau_ext": ("gold_intraday_ext.csv", 0.01, "15min"),  # XAUUSD spot, ~9.8 years, 2012-2022
}


def load_csv(path: str | Path, price_scale: float = 1.0) -> pd.DataFrame:
    """Load a gold CSV into canonical OHLCV (UTC), applying a price scale."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise DataError(f"{path}: missing 'Date' column")
    df = df.rename(columns={"Date": "timestamp", "tick_volume": "volume"}).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    if price_scale != 1.0:
        for col in ("open", "high", "low", "close", "Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = df[col] * price_scale
    return normalize_ohlcv(df)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample canonical OHLCV to ``timeframe`` (downsample only, e.g. 5m->15m)."""
    if timeframe not in _RULE:
        raise DataError(f"unsupported timeframe {timeframe!r}")
    rule = _RULE[timeframe]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return normalize_ohlcv(out)


def load_local(symbol: str = "xau_ext", timeframe: str = "15min") -> pd.DataFrame:
    """Load a committed dataset for ``symbol`` and resample to ``timeframe``."""
    if symbol not in _FILES:
        raise DataError(f"unknown local symbol {symbol!r}; have {sorted(_FILES)}")
    fname, scale, native = _FILES[symbol]
    path = DATA_DIR / fname
    if not path.exists():
        raise DataError(f"local data not found: {path}")
    base = load_csv(path, price_scale=scale)
    return base if timeframe == native else resample_ohlcv(base, timeframe)


def infer_periods_per_year(df: pd.DataFrame, trading_days: int = 252) -> int:
    """Annualisation factor inferred from the median bars per trading day."""
    if len(df) < 2:
        return trading_days
    bars_per_day = int(df.groupby(df.index.date).size().median())
    return max(trading_days, bars_per_day * trading_days)
