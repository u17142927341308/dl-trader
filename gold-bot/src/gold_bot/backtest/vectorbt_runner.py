"""Fast vectorised backtest for the strategy SEARCH.

The search evaluates thousands of (family, params) combinations, so it needs a
fast, vectorised gross-PnL pass — it does not need the path-dependent prop-firm
rules (the event engine applies those to the single finalist). This module is
that fast pass.

It deliberately mirrors the event engine's execution convention exactly
(market-on-close, one-bar information lag, slippage+commission per contract per
side), so the two reconcile to the cent on any strategy when the event engine
runs with stops and prop rules disabled. That reconciliation (tested) is what
lets us trust the fast pass for ranking.

NOTE ON vectorbt: the project pins ``vectorbt`` as the canonical fast engine,
but this runner is implemented in pure NumPy to keep CI light, deterministic and
free of the numba/numpy version pinning that vectorbt imposes. The public
function :func:`run_fast_backtest` is the seam: a vectorbt-backed implementation
can replace the body without changing any caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from config.settings import CostModel, InstrumentSpec, get_settings

from ..strategies.base import StrategyResult


@dataclass
class FastResult:
    equity: pd.Series
    total_gross: float
    total_costs: float
    total_net: float
    n_trades: int


def run_fast_backtest(
    df: pd.DataFrame,
    result: StrategyResult,
    instrument: InstrumentSpec | None = None,
    costs: CostModel | None = None,
    size: int = 1,
    start_equity: float | None = None,
) -> FastResult:
    """Vectorised close-to-close simulation with costs (no stops, no prop rules)."""
    s = get_settings()
    instrument = instrument or s.instrument
    costs = costs or s.cost_model
    start_equity = s.account_rules.account_size if start_equity is None else start_equity

    pv = instrument.point_value
    cost_per_contract_side = costs.commission_per_side + costs.slippage_ticks * instrument.tick_value

    close = df["close"].to_numpy(dtype=float)
    sig = result.signal.reindex(df.index).fillna(0.0).to_numpy(dtype=float)

    # Position held over close[i-1] -> close[i] is signal[i-1] (one-bar lag).
    pos = np.zeros(len(close), dtype=float)
    pos[1:] = sig[:-1]
    pos_contracts = pos * size

    # Price PnL per bar (close-to-close).
    price_diff = np.zeros(len(close), dtype=float)
    price_diff[1:] = close[1:] - close[:-1]
    pnl_price = pos_contracts * price_diff * pv

    # Costs on turnover (contracts traded each bar, both entries and exits).
    turnover = np.abs(np.diff(pos_contracts, prepend=0.0))
    cost = turnover * cost_per_contract_side

    pnl = pnl_price - cost
    equity = start_equity + np.cumsum(pnl)

    # Trades = number of position "episodes" = entries into a nonzero position
    # (from flat or a flip). The event engine closes every entry exactly once
    # (in-loop or via an end closeout), so this matches its closed-trade count.
    prev = np.concatenate(([0.0], pos[:-1]))
    n_trades = int(np.sum((pos != 0) & (pos != prev)))

    return FastResult(
        equity=pd.Series(equity, index=df.index, name="equity"),
        total_gross=float(pnl_price.sum()),
        total_costs=float(cost.sum()),
        total_net=float(pnl.sum()),
        n_trades=n_trades,
    )
