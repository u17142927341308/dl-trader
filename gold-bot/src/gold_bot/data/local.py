"""Local CSV data + OHLCV resampling.

Reads the committed real MGC/GC 5-minute futures history (the canonical
research data — no proxy, no scaling, real contract prices) and resamples it to
whatever timeframe the strategy runs on.
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


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load one of the uploaded 5-minute CSVs into canonical OHLCV (UTC)."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise DataError(f"{path}: missing 'Date' column")
    df = df.rename(columns={"Date": "timestamp"}).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return normalize_ohlcv(df)  # keeps only OHLCV, sorts, dedupes, drops NaN


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample canonical OHLCV to ``timeframe`` (e.g. '5min' -> '15min')."""
    if timeframe not in _RULE:
        raise DataError(f"unsupported timeframe {timeframe!r}")
    rule = _RULE[timeframe]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return normalize_ohlcv(out)


def load_local(symbol: str = "mgc", timeframe: str = "15min") -> pd.DataFrame:
    """Load committed 5m data for ``symbol`` and resample to ``timeframe``."""
    path = DATA_DIR / f"{symbol}_5m.csv"
    if not path.exists():
        raise DataError(f"local data not found: {path}")
    base = load_csv(path)
    return base if timeframe == "5min" else resample_ohlcv(base, timeframe)


def infer_periods_per_year(df: pd.DataFrame, trading_days: int = 252) -> int:
    """Annualisation factor inferred from the median bars per trading day."""
    if len(df) < 2:
        return trading_days
    bars_per_day = int(df.groupby(df.index.date).size().median())
    return max(trading_days, bars_per_day * trading_days)
