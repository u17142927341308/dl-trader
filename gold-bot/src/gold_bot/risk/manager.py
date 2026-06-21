"""DynamicRiskManager — prop-firm-aware sizing & circuit breaker.

The SAME instance is used by the event-driven backtest and the live signal
generator, so what you test is what you trade. It:

* sizes each trade so the stop-loss risk fits a budget (a fraction of the daily
  loss limit), volatility-targeted via ATR and rounded to whole contracts;
* scales size DOWN as the headroom to the trailing-drawdown floor shrinks, and
  goes flat entirely once headroom is critical (circuit breaker);
* scales size down as the day's loss approaches the daily limit.

The hard halts (daily-loss stop, trailing-DD death) are enforced by the engine's
prop-rule layer; this manager handles the *graduated* de-risking before those
hard limits are hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.settings import AccountRules, InstrumentSpec, Settings, get_settings

from .position_sizing import atr_risk_per_contract, contracts_for_budget


@dataclass
class DynamicRiskManager:
    rules: AccountRules
    instrument: InstrumentSpec
    # Risk budget per trade as a fraction of the daily loss limit.
    risk_fraction_of_daily_limit: float = 0.5
    # Headroom thresholds (dollars to the trailing-DD floor). Default from rules.
    headroom_reduce: float | None = None  # below this, scale size down linearly
    headroom_flat: float | None = None  # below this, go flat (circuit breaker)

    def __post_init__(self) -> None:
        if self.headroom_reduce is None:
            self.headroom_reduce = self.rules.trailing_drawdown
        if self.headroom_flat is None:
            self.headroom_flat = 0.25 * self.rules.trailing_drawdown

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> DynamicRiskManager:
        s = settings or get_settings()
        return cls(rules=s.account_rules, instrument=s.instrument)

    @property
    def base_budget(self) -> float:
        return self.risk_fraction_of_daily_limit * self.rules.daily_loss_limit

    def _headroom_factor(self, headroom: float) -> float:
        assert self.headroom_reduce is not None and self.headroom_flat is not None
        if headroom <= self.headroom_flat:
            return 0.0
        if headroom >= self.headroom_reduce:
            return 1.0
        span = self.headroom_reduce - self.headroom_flat
        return max(0.0, min(1.0, (headroom - self.headroom_flat) / span)) if span > 0 else 1.0

    def _daily_factor(self, day_loss: float) -> float:
        if day_loss <= 0:
            return 1.0
        remaining = self.rules.daily_loss_limit - day_loss
        return max(0.0, remaining / self.rules.daily_loss_limit)

    def size(
        self,
        atr: float,
        stop_mult: float,
        equity: float,
        headroom: float,
        day_loss: float = 0.0,
    ) -> int:
        """Contracts to trade given volatility and current risk state (0 = flat)."""
        if atr <= 0 or math.isnan(atr):
            return 0
        h_factor = self._headroom_factor(headroom)
        if h_factor <= 0.0:
            return 0  # circuit breaker
        rpc = atr_risk_per_contract(atr, stop_mult, self.instrument.point_value)
        n = contracts_for_budget(self.base_budget, rpc, self.rules.max_contracts)
        n = int(math.floor(n * h_factor * self._daily_factor(day_loss)))
        return max(0, min(self.rules.max_contracts, n))
