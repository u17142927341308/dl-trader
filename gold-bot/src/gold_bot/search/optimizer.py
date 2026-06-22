"""Grid optimiser over the search space, scored on out-of-sample WFO results.

Every trial is logged (not just winners) because the *total trial count* is what
the Deflated Sharpe Ratio uses to penalise multiple testing. Scoring uses the
FAST vectorised runner so thousands of trials over years of intraday bars stay
cheap; the orchestrator then verifies the single finalist with the event-driven
engine (full prop-rule + cost accounting).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.event_engine import EventConfig
from ..backtest.metrics import returns_from_equity, sharpe_ratio
from ..strategies.registry import iter_search_space
from .walk_forward import FastEval, Window, evaluate_candidate_fast


@dataclass
class TrialResult:
    family: str
    params: dict
    oos_sharpe: float
    oos_return_pct: float
    n_trades: int
    is_return_pct: float
    score: float  # ranking score (OOS Sharpe; -inf if it busts the account)
    dd_breached: bool

    def summary(self) -> dict:
        return {
            "family": self.family,
            "params": self.params,
            "oos_sharpe": round(self.oos_sharpe, 3),
            "oos_return_pct": round(self.oos_return_pct, 4),
            "n_trades": self.n_trades,
            "dd_breached": self.dd_breached,
            "score": None if self.score == float("-inf") else round(self.score, 3),
        }


@dataclass
class SearchLedger:
    trials: list[TrialResult] = field(default_factory=list)
    trial_sharpe_std: float = 0.0

    @property
    def n_trials(self) -> int:
        return len(self.trials)


def _fast_to_trial(fe: FastEval, periods_per_year: int) -> TrialResult:
    rets = returns_from_equity(fe.oos_equity)
    sharpe = sharpe_ratio(rets, periods_per_year)
    ret_pct = (
        float(fe.oos_equity.iloc[-1] / fe.oos_equity.iloc[0] - 1.0) if len(fe.oos_equity) > 1 else 0.0
    )
    score = float("-inf") if fe.dd_breached else sharpe
    return TrialResult(
        family=fe.family,
        params=fe.params,
        oos_sharpe=sharpe,
        oos_return_pct=ret_pct,
        n_trades=fe.oos_n_trades,
        is_return_pct=fe.is_return_pct,
        score=score,
        dd_breached=fe.dd_breached,
    )


def run_grid_search(
    df: pd.DataFrame,
    windows: list[Window],
    *,
    max_trials_per_family: int = 200,
    event_cfg: EventConfig | None = None,
    periods_per_year: int = 252,
) -> SearchLedger:
    """Run every (family, params) combination through fast walk-forward scoring."""
    per_family: dict[str, int] = defaultdict(int)
    ledger = SearchLedger()
    for family, params in iter_search_space():
        if per_family[family] >= max_trials_per_family:
            continue
        per_family[family] += 1
        fe = evaluate_candidate_fast(df, family, params, windows, event_cfg)
        ledger.trials.append(_fast_to_trial(fe, periods_per_year))

    # DSR works in per-observation Sharpe units, so convert the dispersion of
    # the (annualised) trial Sharpes to per-observation before storing it.
    sharpes = np.array([t.oos_sharpe for t in ledger.trials], dtype=float)
    ann_std = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    ledger.trial_sharpe_std = ann_std / (periods_per_year**0.5) if periods_per_year > 0 else 0.0
    return ledger
