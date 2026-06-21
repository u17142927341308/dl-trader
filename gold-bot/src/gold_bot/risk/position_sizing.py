"""Position sizing — ATR/volatility-targeted, contract-rounded, plan-capped.

Pure functions so sizing is testable in isolation and reused identically by the
backtest engine and the live signal generator.
"""

from __future__ import annotations

import math


def atr_risk_per_contract(atr: float, stop_mult: float, point_value: float) -> float:
    """Dollar risk of one contract if stopped out at ``stop_mult`` x ATR."""
    return max(0.0, atr) * stop_mult * point_value


def contracts_for_budget(
    risk_budget: float, risk_per_contract: float, max_contracts: int
) -> int:
    """Largest whole-contract count whose stop risk fits ``risk_budget``."""
    if risk_per_contract <= 0 or risk_budget <= 0:
        return 0
    n = int(math.floor(risk_budget / risk_per_contract))
    return max(0, min(max_contracts, n))
