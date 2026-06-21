"""Technical indicators — strictly causal (NO look-ahead).

THE most important property in this whole project: an indicator value at bar
``t`` may depend ONLY on data at bars ``0..t``, never on a future bar. If that
invariant breaks, every backtest becomes a fantasy. We therefore:

    * implement each indicator in pure pandas using only causal operations
      (``rolling``, ``ewm``, ``shift`` with a positive lag, cumulative ops);
    * NEVER use centred windows, ``shift(-k)``, future-fill, or full-series
      normalisation that leaks the future into the past;
    * test the "prefix-stability" property in tests/test_indicators.py: the
      indicator computed on data[0:t] must equal the indicator computed on the
      full series, restricted to [0:t]. If appending future bars changes a past
      value, the indicator leaks.

pandas-ta is listed as an optional dependency, but these hand-rolled versions
are what the strategies use, precisely because we can prove they don't leak.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over the trailing ``period`` bars (inclusive)."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. ``adjust=False`` => purely recursive/causal."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).std(ddof=0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range = max(H-L, |H-prev_close|, |L-prev_close|).

    Uses ``close.shift(1)`` (a PAST close), so it is causal.
    """
    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range via Wilder's smoothing (an EMA; causal)."""
    tr = true_range(high, low, close)
    # Wilder's smoothing == EMA with alpha = 1/period.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). Range 0..100, causal."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 the ratio is inf -> RSI 100; pandas handles inf->100.
    return out


class BollingerBands(pd.DataFrame):
    """Marker type for clarity; instances are plain DataFrames with mid/upper/lower."""


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger bands: middle SMA +/- ``num_std`` rolling std. Causal."""
    mid = sma(series, period)
    sd = rolling_std(series, period)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * sd, "lower": mid - num_std * sd}
    )


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, histogram. All from causal EMAs."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20, exclude_current: bool = True
) -> pd.DataFrame:
    """Donchian channel: rolling highest-high / lowest-low.

    For BREAKOUT logic you want the channel of the bars BEFORE the current one,
    otherwise the current bar's own high trivially equals the channel high and
    no breakout can ever register. ``exclude_current=True`` shifts the channel
    by one bar (still causal: it uses only past bars).
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    if exclude_current:
        upper = upper.shift(1)
        lower = lower.shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower})
