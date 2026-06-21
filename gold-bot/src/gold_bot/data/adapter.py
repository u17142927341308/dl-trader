"""Abstract data-adapter interface.

Strategy / backtest / signal code only ever talks to a :class:`DataAdapter`.
That means the free yfinance source used today can be swapped for a paid
intraday source (Databento, CME, Tradovate history) tomorrow WITHOUT touching a
single line of strategy code. Just implement this interface and inject it.

Contract for every adapter:
    * return a pandas DataFrame indexed by a tz-aware UTC DatetimeIndex,
      sorted ascending, with no duplicate timestamps;
    * columns are exactly ["open", "high", "low", "close", "volume"] (lowercase);
    * rows with any NaN in OHLC are dropped;
    * the index name is "timestamp".

The shared validator :func:`normalize_ohlcv` enforces this so individual
adapters cannot drift from the contract.
"""

from __future__ import annotations

import abc

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Raised when an adapter cannot return valid data."""


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical contract.

    Validates and normalises so downstream code can rely on the invariants.
    Raises :class:`DataError` if the frame cannot be made conformant.
    """
    if df is None or len(df) == 0:
        raise DataError("empty dataframe")

    out = df.copy()

    # Case-insensitive column mapping.
    rename: dict[str, str] = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in OHLCV_COLUMNS:
            rename[col] = key
    out = out.rename(columns=rename)

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataError(f"missing columns: {missing}; got {list(out.columns)}")
    out = out[OHLCV_COLUMNS]

    # Index -> tz-aware UTC DatetimeIndex.
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"

    # Sort, de-duplicate (keep last), drop NaN OHLC rows.
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])

    if len(out) == 0:
        raise DataError("no valid rows after normalisation")
    return out


class DataAdapter(abc.ABC):
    """Abstract source of OHLCV bars for one instrument/timeframe."""

    @abc.abstractmethod
    def fetch(self, symbol: str, timeframe: str, start: str | None = None) -> pd.DataFrame:
        """Return canonical OHLCV bars.

        Args:
            symbol: instrument symbol understood by the adapter (e.g. "GC=F").
            timeframe: "1d", "1h", etc.
            start: ISO date string lower bound, or None for the adapter default.

        Returns:
            DataFrame conforming to :func:`normalize_ohlcv`.
        """
        raise NotImplementedError
