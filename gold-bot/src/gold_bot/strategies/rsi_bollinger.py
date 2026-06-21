"""RSI + Bollinger mean reversion.

Enter long when price closes below the lower Bollinger band AND RSI is
oversold; cover/flat when price reverts back to the middle band. Mirror logic
for shorts at the upper band when RSI is overbought. Positions are held until
the mean-reversion target (middle band) is reached — implemented as a causal
forward fill.
"""

from __future__ import annotations

import pandas as pd

from ..features import indicators as ind
from .base import Strategy
from .registry import register


@register
class RsiBollinger(Strategy):
    name = "rsi_bollinger"
    param_spec = {
        "rsi_period": (int, 14),
        "rsi_low": (float, 30.0),
        "rsi_high": (float, 70.0),
        "bb_period": (int, 20),
        "bb_std": (float, 2.0),
        "atr_period": (int, 14),
        "atr_stop_mult": (float, 2.0),
    }

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi = ind.rsi(close, self.params["rsi_period"])
        bb = ind.bollinger_bands(close, self.params["bb_period"], self.params["bb_std"])

        enter_long = (close < bb["lower"]) & (rsi < self.params["rsi_low"])
        exit_long = close >= bb["mid"]
        enter_short = (close > bb["upper"]) & (rsi > self.params["rsi_high"])
        exit_short = close <= bb["mid"]

        # Mask warmup region where bands/rsi are undefined.
        valid = bb["mid"].notna() & rsi.notna()
        enter_long &= valid
        enter_short &= valid

        return self._stateful_fill(
            enter_long.fillna(False),
            exit_long.fillna(False),
            enter_short.fillna(False),
            exit_short.fillna(False),
        )
