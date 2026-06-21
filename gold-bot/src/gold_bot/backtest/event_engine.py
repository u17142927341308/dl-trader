"""Event-driven backtest verifier — the source of truth for the finalist.

vectorbt (and the fast runner) cannot model the *path-dependent* Tradovate
trailing-drawdown and daily-loss rules, so this engine re-simulates a single
strategy bar-by-bar with realistic costs and those prop-firm mechanics. What it
tests is what the live signal generator (Phase 7) will trade, because both use
the same risk code path.

Execution convention (documented so look-ahead is auditable)
-----------------------------------------------------------
The signal at bar ``t`` is decided from data up to and including bar ``t`` and
is executed **market-on-close at bar t** — i.e. the position held over the move
``close[t] -> close[t+1]`` equals ``signal[t]``. The close is known at the
instant the order is submitted, so this is not look-ahead; slippage of at least
one tick is charged to account for MOC fill realism. Stops and prop-rule
breaches are detected *intrabar* using each bar's high/low.

Costs: commission per contract per side + slippage (>= 1 tick) per side,
charged as dollars on every contract traded. No frictionless fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from config.settings import (
    AccountRules,
    CostModel,
    InstrumentSpec,
    Settings,
    get_settings,
)

from ..risk.prop_rules import DailyLossLimit, TrailingDrawdown
from ..strategies.base import StrategyResult


@dataclass
class EventConfig:
    """Everything the engine needs beyond the price data and the signal."""

    instrument: InstrumentSpec
    costs: CostModel
    rules: AccountRules
    fixed_size: int = 1  # contracts; Phase 6 swaps in dynamic sizing
    use_stops: bool = True
    enforce_prop: bool = True
    ratchet_mode: str = "eod"

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **overrides: object) -> EventConfig:
        s = settings or get_settings()
        cfg = cls(instrument=s.instrument, costs=s.cost_model, rules=s.account_rules)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    @property
    def cost_per_contract_side(self) -> float:
        return self.costs.commission_per_side + self.costs.slippage_ticks * self.instrument.tick_value


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    size: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    reason: str  # signal | stop | daily_halt | dd_dead | end


@dataclass
class BacktestResult:
    equity: pd.Series  # marked at each bar's close
    pnl: pd.Series  # per-bar change in equity ($)
    position: pd.Series  # held direction per bar in {-1, 0, +1} (for exposure)
    trades: list[Trade] = field(default_factory=list)
    dd_dead: bool = False
    dd_breach_time: pd.Timestamp | None = None
    daily_halt_days: int = 0
    final_equity: float = 0.0
    total_gross: float = 0.0
    total_costs: float = 0.0
    total_net: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_time", "exit_time", "direction", "size", "entry_price",
                    "exit_price", "gross_pnl", "costs", "net_pnl", "reason",
                ]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


class _Sim:
    """Mutable per-run state; keeps the public function readable."""

    def __init__(self, df: pd.DataFrame, result: StrategyResult, cfg: EventConfig) -> None:
        self.idx = df.index
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.sig = result.signal.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
        self.atr = result.atr.reindex(df.index).to_numpy(dtype=float)
        self.stop_mult = result.atr_stop_mult
        self.cfg = cfg
        self.pv = cfg.instrument.point_value

        self.account = cfg.rules.account_size
        self.tdd = TrailingDrawdown(self.account, cfg.rules.trailing_drawdown, cfg.ratchet_mode)
        self.dll = DailyLossLimit(cfg.rules.daily_loss_limit)

        self.realized = 0.0
        self.side = 0  # -1/0/+1
        self.size = 0
        self.entry_price = 0.0
        self.entry_time: pd.Timestamp | None = None
        self.entry_cost = 0.0
        self.stop = float("nan")
        self.trades: list[Trade] = []

    # --- position transitions ---------------------------------------------
    def open_position(self, direction: int, price: float, size: int, time: pd.Timestamp, atr_at_entry: float) -> None:
        cost = size * self.cfg.cost_per_contract_side
        self.realized -= cost
        self.side = direction
        self.size = size
        self.entry_price = price
        self.entry_time = time
        self.entry_cost = cost
        if np.isnan(atr_at_entry):
            self.stop = float("nan")
        else:
            self.stop = price - direction * atr_at_entry * self.stop_mult

    def close_position(
        self, price: float, time: pd.Timestamp, reason: str, charge_exit_cost: bool = True
    ) -> None:
        gross = self.side * self.size * (price - self.entry_price) * self.pv
        # An end-of-data mark-to-market closeout is not a real fill, so it is
        # not charged exit costs (keeps it consistent with the fast runner).
        exit_cost = self.size * self.cfg.cost_per_contract_side if charge_exit_cost else 0.0
        self.realized += gross - exit_cost
        self.trades.append(
            Trade(
                entry_time=self.entry_time,  # type: ignore[arg-type]
                exit_time=time,
                direction=self.side,
                size=self.size,
                entry_price=self.entry_price,
                exit_price=price,
                gross_pnl=gross,
                costs=self.entry_cost + exit_cost,
                net_pnl=gross - self.entry_cost - exit_cost,
                reason=reason,
            )
        )
        self.side = 0
        self.size = 0
        self.entry_cost = 0.0
        self.stop = float("nan")

    def unrealized(self, mark: float) -> float:
        if self.side == 0:
            return 0.0
        return self.side * self.size * (mark - self.entry_price) * self.pv


def run_event_backtest(
    df: pd.DataFrame, result: StrategyResult, cfg: EventConfig | None = None
) -> BacktestResult:
    """Simulate ``result``'s signal over ``df`` under the funded-account rules."""
    cfg = cfg or EventConfig.from_settings()
    n = len(df)
    sim = _Sim(df, result, cfg)
    equity_out = np.empty(n, dtype=float)
    pnl_out = np.zeros(n, dtype=float)
    pos_out = np.zeros(n, dtype=float)

    equity = sim.account
    prev_equity = sim.account
    day_start_equity = sim.account
    daily_halt = False
    halt_days: set = set()
    dead = False
    dead_time: pd.Timestamp | None = None

    for i in range(n):
        time = sim.idx[i]
        new_day = (i == 0) or (sim.idx[i].date() != sim.idx[i - 1].date())
        if new_day:
            day_start_equity = equity
            daily_halt = False

        # Decide target from the previous bar's signal (MOC at close[i-1]).
        if i == 0 or dead:
            target = 0
        else:
            target = 0 if daily_halt else int(sim.sig[i - 1])

        decision_price = sim.c[i - 1] if i >= 1 else sim.o[0]
        decision_atr = sim.atr[i - 1] if i >= 1 else sim.atr[i]

        # Adjust to target position at the decision price.
        if not dead and target != sim.side:
            if sim.side != 0:
                sim.close_position(decision_price, sim.idx[i - 1] if i >= 1 else time, "signal")
            if target != 0:
                size = max(1, int(cfg.fixed_size))
                sim.open_position(target, decision_price, size, sim.idx[i - 1] if i >= 1 else time, decision_atr)

        # The bar's exposure is the position held into it (before any intrabar exit).
        pos_out[i] = sim.side

        # Intrabar stop check (only meaningful from bar 1 onward).
        if i >= 1 and cfg.use_stops and sim.side != 0 and not dead and not np.isnan(sim.stop):
            if sim.side == 1 and sim.low[i] <= sim.stop:
                sim.close_position(sim.stop, time, "stop")
            elif sim.side == -1 and sim.h[i] >= sim.stop:
                sim.close_position(sim.stop, time, "stop")

        # Mark equity. Worst intrabar mark is used for breach detection.
        if i >= 1:
            if sim.side == 1:
                worst_mark = sim.low[i]
            elif sim.side == -1:
                worst_mark = sim.h[i]
            else:
                worst_mark = sim.c[i]
            close_mark = sim.c[i]
        else:
            worst_mark = close_mark = sim.c[i]

        equity_worst = sim.account + sim.realized + sim.unrealized(worst_mark)

        # Prop-rule checks (intraday, on the worst mark).
        if cfg.enforce_prop and not dead:
            if sim.tdd.check(equity_worst):
                dead = True
                dead_time = time
                if sim.side != 0:
                    sim.close_position(worst_mark, time, "dd_dead")
            elif not daily_halt and sim.dll.breached(day_start_equity, equity_worst):
                daily_halt = True
                halt_days.add(sim.idx[i].date())
                if sim.side != 0:
                    sim.close_position(worst_mark, time, "daily_halt")

        equity = sim.account + sim.realized + sim.unrealized(close_mark)
        equity_out[i] = equity
        pnl_out[i] = equity - prev_equity
        prev_equity = equity

        # End-of-day ratchet of the trailing drawdown.
        eod = (i == n - 1) or (sim.idx[i + 1].date() != sim.idx[i].date())
        if eod and cfg.enforce_prop:
            sim.tdd.on_day_close(equity)

        if dead:
            equity_out[i + 1 :] = equity
            pos_out[i + 1 :] = 0.0
            break

    # Close any still-open position at the final close (mark-to-market) so it is
    # recorded as a trade. No exit cost charged (not a real fill).
    if not dead and sim.side != 0 and n > 0:
        sim.close_position(sim.c[-1], sim.idx[-1], "end", charge_exit_cost=False)

    total_gross = sum(t.gross_pnl for t in sim.trades)
    total_costs = sum(t.costs for t in sim.trades)
    final_equity = equity_out[-1] if n else sim.account

    return BacktestResult(
        equity=pd.Series(equity_out, index=sim.idx, name="equity"),
        pnl=pd.Series(pnl_out, index=sim.idx, name="pnl"),
        position=pd.Series(pos_out, index=sim.idx, name="position"),
        trades=sim.trades,
        dd_dead=dead,
        dd_breach_time=dead_time,
        daily_halt_days=len(halt_days),
        final_equity=float(final_equity),
        total_gross=float(total_gross),
        total_costs=float(total_costs),
        total_net=float(final_equity - sim.account),
    )
