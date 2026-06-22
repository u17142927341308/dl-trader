"""Opening-Range Breakout (ORB) — session-filtered intraday strategy.

The classic day-trading setup, built to be friendly to a tight prop-firm
drawdown: it is FLAT most of the time, takes at most one directional trade per
session, and always goes flat by the session close (no overnight risk).

Logic (all times in UTC, on the trading timeframe's bars):
    * The "opening range" is the high/low of the first ``range_minutes`` after
      the session open (``open_minute`` minutes past midnight UTC).
    * After the range completes, the first close ABOVE the range high goes long;
      the first close BELOW the range low goes short.
    * The position is held until the session end (``end_minute``); the
      event engine's ATR stop provides the risk exit. Flat outside the session.

This is causal: the range high/low for a day are computed only from bars inside
that day's opening window, and signals fire only AFTER the window has closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class OpeningRange(Strategy):
    name = "opening_range"
    param_spec = {
        "open_minute": (int, 810),   # 13:30 UTC (COMEX gold open)
        "range_minutes": (int, 30),
        "end_minute": (int, 1200),   # 20:00 UTC
        "atr_period": (int, 14),
        "atr_stop_mult": (float, 1.5),
    }

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        idx = df.index
        tod = idx.hour * 60 + idx.minute  # minutes past midnight UTC
        day = pd.Index(idx.date)

        open_min = self.params["open_minute"]
        range_end = open_min + self.params["range_minutes"]
        end_min = self.params["end_minute"]

        in_range = (tod >= open_min) & (tod < range_end)
        in_session = (tod >= open_min) & (tod < end_min)
        after_range = tod >= range_end

        high = df["high"]
        low = df["low"]
        close = df["close"]
        # Per-day opening-range high/low (NaN outside the range window are ignored).
        rh = high.where(in_range).groupby(day).transform("max")
        rl = low.where(in_range).groupby(day).transform("min")

        active = pd.Series(after_range & in_session, index=idx)
        enter_long = (active & (close > rh)).fillna(False)
        enter_short = (active & (close < rl)).fillna(False)
        flat = pd.Series(~np.asarray(in_session), index=idx)  # force flat off-session

        return self._stateful_fill(enter_long, flat, enter_short, flat)
