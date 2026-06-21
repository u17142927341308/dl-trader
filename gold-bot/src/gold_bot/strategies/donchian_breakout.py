"""Donchian channel breakout — classic trend/breakout (turtle-style).

Go long when price closes above the highest high of the last ``entry_lookback``
bars; go short below the lowest low. Exit when price closes through the opposite
``exit_lookback`` channel. The channels exclude the current bar so a breakout
can actually register.
"""

from __future__ import annotations

import pandas as pd

from ..features import indicators as ind
from .base import Strategy
from .registry import register


@register
class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    param_spec = {
        "entry_lookback": (int, 40),
        "exit_lookback": (int, 20),
        "atr_period": (int, 14),
        "atr_stop_mult": (float, 2.0),
    }

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        entry = ind.donchian(high, low, self.params["entry_lookback"], exclude_current=True)
        exit_ = ind.donchian(high, low, self.params["exit_lookback"], exclude_current=True)

        enter_long = close > entry["upper"]
        enter_short = close < entry["lower"]
        # Exit a long when price breaks the lower exit channel, and vice versa.
        exit_long = close < exit_["lower"]
        exit_short = close > exit_["upper"]

        valid = entry["upper"].notna() & exit_["lower"].notna()
        enter_long &= valid
        enter_short &= valid

        return self._stateful_fill(
            enter_long.fillna(False),
            exit_long.fillna(False),
            enter_short.fillna(False),
            exit_short.fillna(False),
        )
