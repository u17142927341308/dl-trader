"""MACD trend with a regime filter.

Long when the MACD line is above its signal line AND price is above a slow
regime moving average (trend-up confirmation). Short on the mirror condition.
The regime filter keeps the strategy from fighting the dominant trend.
"""

from __future__ import annotations

import pandas as pd

from ..features import indicators as ind
from .base import Strategy
from .registry import register


@register
class MacdTrend(Strategy):
    name = "macd_trend"
    param_spec = {
        "fast": (int, 12),
        "slow": (int, 26),
        "signal": (int, 9),
        "regime_ma": (int, 200),
        "atr_period": (int, 14),
        "atr_stop_mult": (float, 2.0),
    }

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        macd = ind.macd(close, self.params["fast"], self.params["slow"], self.params["signal"])
        regime = ind.sma(close, self.params["regime_ma"])

        up = (macd["macd"] > macd["signal"]) & (close > regime)
        down = (macd["macd"] < macd["signal"]) & (close < regime)

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(up, 1.0).mask(down, -1.0)
        # Zero-out the warmup window where the regime MA is undefined.
        return signal.where(regime.notna(), 0.0)
