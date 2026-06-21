"""EMA crossover — trend following.

Long when the fast EMA is above the slow EMA, short when below. A canonical,
hard-to-overfit trend model: few parameters, monotone behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ind
from .base import Strategy
from .registry import register


@register
class EmaCross(Strategy):
    name = "ema_cross"
    param_spec = {
        "fast": (int, 20),
        "slow": (int, 100),
        "atr_period": (int, 14),
        "atr_stop_mult": (float, 2.0),
    }

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        if self.params["fast"] >= self.params["slow"]:
            # Degenerate config: no trend signal possible.
            return pd.Series(0.0, index=df.index)
        fast = ind.ema(df["close"], self.params["fast"])
        slow = ind.ema(df["close"], self.params["slow"])
        diff = fast - slow
        signal = pd.Series(np.sign(diff), index=df.index)
        return signal.where(slow.notna(), 0.0)
