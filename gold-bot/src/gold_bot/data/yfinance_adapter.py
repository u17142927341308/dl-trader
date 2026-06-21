"""yfinance-backed data adapter (free, default).

Uses ``GC=F`` (continuous COMEX gold front-month proxy) for daily history and
an hourly fallback. yfinance data is good enough for *research and structure*,
but it is a continuous-contract proxy, not exchange settlement data — when you
go live, swap in a paid intraday adapter behind the same :class:`DataAdapter`
interface. This adapter wraps fetches in the :class:`ParquetCache`.

Network access is required only on a cache miss; tests use a fake adapter and
never hit the network.
"""

from __future__ import annotations

import time

import pandas as pd

from .adapter import DataAdapter, DataError, normalize_ohlcv
from .cache import ParquetCache

# yfinance interval strings keyed by our timeframe vocabulary.
_INTERVALS = {"1d": "1d", "1h": "1h", "1wk": "1wk"}


class YFinanceAdapter(DataAdapter):
    def __init__(self, cache: ParquetCache | None = None, max_retries: int = 3) -> None:
        self.cache = cache or ParquetCache()
        self.max_retries = max_retries

    def fetch(self, symbol: str, timeframe: str, start: str | None = None) -> pd.DataFrame:
        if timeframe not in _INTERVALS:
            raise DataError(f"unsupported timeframe {timeframe!r}")

        cached = self.cache.load(symbol, timeframe)
        if cached is not None:
            return cached

        df = self._download(symbol, timeframe, start)
        normalized = normalize_ohlcv(df)
        self.cache.save(symbol, timeframe, normalized)
        return normalized

    def _download(self, symbol: str, timeframe: str, start: str | None) -> pd.DataFrame:
        # Imported lazily so the package imports without yfinance installed
        # (e.g. in unit tests that use a fake adapter).
        import yfinance as yf

        interval = _INTERVALS[timeframe]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if df is not None and len(df) > 0:
                    # yfinance returns a MultiIndex column frame for single
                    # tickers in newer versions; flatten to the OHLCV level.
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df
            except Exception as exc:  # noqa: BLE001 - retry any transient error
                last_err = exc
            time.sleep(1.0 * (attempt + 1))

        raise DataError(f"yfinance download failed for {symbol} {timeframe}: {last_err}")
