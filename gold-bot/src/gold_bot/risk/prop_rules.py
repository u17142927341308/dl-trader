"""Tradovate funded-account rule simulators: trailing drawdown + daily loss.

These two mechanics are *path-dependent* — they depend on the order in which
PnL arrives, not just the total — which is exactly why a vectorised backtest
alone cannot model them and the event-driven engine is mandatory.

TrailingDrawdown
----------------
A floor that sits ``amount`` dollars below the running equity peak. The peak
**ratchets up** with new highs and **never moves down**, so the floor only ever
rises. If equity touches the floor, the account is dead — permanently.

The brief specifies an **end-of-day** trailing drawdown: the peak is raised from
*settled end-of-day* equity, so an intraday spike that retraces by the close
does NOT raise the floor (the conservative, account-protecting interpretation).
``ratchet_mode="intraday"`` is provided for prop firms that trail the unrealised
intraday high instead. Breaches are always checked intraday (on every bar's
mark-to-market), because that is when a real account blows up.

DailyLossLimit
--------------
Measured from the equity at the start of the trading day. Once the day's loss
reaches the limit, trading halts for the rest of the day (the engine flattens
and blocks new entries until the next session).
"""

from __future__ import annotations

_EPS = 1e-9


class TrailingDrawdown:
    """End-of-day trailing max drawdown with an only-rising floor."""

    def __init__(
        self, start_equity: float, amount: float, ratchet_mode: str = "eod"
    ) -> None:
        if ratchet_mode not in {"eod", "intraday"}:
            raise ValueError("ratchet_mode must be 'eod' or 'intraday'")
        self.amount = float(amount)
        self.peak = float(start_equity)
        self.ratchet_mode = ratchet_mode
        self.dead = False
        self.breach_equity: float | None = None

    @property
    def floor(self) -> float:
        """The kill level: account dies if equity touches or drops below it."""
        return self.peak - self.amount

    def update_intraday(self, equity: float) -> None:
        """Raise the peak on an intraday high (only in ``intraday`` mode)."""
        if self.ratchet_mode == "intraday" and equity > self.peak:
            self.peak = equity

    def on_day_close(self, eod_equity: float) -> None:
        """Ratchet the peak from settled end-of-day equity. Never lowers it."""
        if eod_equity > self.peak:
            self.peak = eod_equity

    def check(self, equity: float) -> bool:
        """Return True (and latch ``dead``) if equity has breached the floor."""
        self.update_intraday(equity)
        if not self.dead and equity <= self.floor + _EPS:
            self.dead = True
            self.breach_equity = equity
        return self.dead

    def headroom(self, equity: float) -> float:
        """Dollars between current equity and the kill floor (>= 0 if alive)."""
        return equity - self.floor


class DailyLossLimit:
    """Intraday daily loss limit, measured from the day's starting equity."""

    def __init__(self, limit: float) -> None:
        self.limit = float(limit)

    def loss(self, day_start_equity: float, equity: float) -> float:
        """Realised+unrealised loss so far today (positive number = a loss)."""
        return day_start_equity - equity

    def breached(self, day_start_equity: float, equity: float) -> bool:
        return self.loss(day_start_equity, equity) >= self.limit - _EPS
