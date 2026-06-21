"""Grid optimiser over the search space, scored on out-of-sample WFO results.

Every trial is logged (not just winners) because the *total trial count* is what
the Deflated Sharpe Ratio uses to penalise multiple testing. An Optuna/TPE
backend can be slotted in behind :func:`run_grid_search` later; the grid is the
honest default because it makes the trial count explicit and reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.event_engine import EventConfig
from ..backtest.metrics import Metrics, compute_metrics
from ..strategies.registry import iter_search_space
from .walk_forward import CandidateEval, Window, evaluate_candidate


@dataclass
class TrialResult:
    family: str
    params: dict
    oos_metrics: Metrics
    is_return_pct: float
    score: float  # ranking score (OOS Sharpe; -inf if it busts the account)
    dd_breached: bool

    @property
    def oos_sharpe(self) -> float:
        return self.oos_metrics.sharpe

    def summary(self) -> dict:
        return {
            "family": self.family,
            "params": self.params,
            "oos_sharpe": round(self.oos_metrics.sharpe, 3),
            "oos_profit_factor": round(self.oos_metrics.profit_factor, 3),
            "oos_return_pct": round(self.oos_metrics.total_return_pct, 4),
            "n_trades": self.oos_metrics.n_trades,
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


def _eval_to_trial(ev: CandidateEval, periods_per_year: int) -> TrialResult:
    oos_metrics = compute_metrics(
        ev.oos, periods_per_year=periods_per_year, n_trials=1, trial_sharpe_std=0.0
    )
    is_return = ev.is_.total_net / max(1.0, ev.is_.final_equity - ev.is_.total_net)
    breached = bool(ev.oos.dd_dead)
    score = float("-inf") if breached else oos_metrics.sharpe
    return TrialResult(
        family=ev.family,
        params=ev.params,
        oos_metrics=oos_metrics,
        is_return_pct=is_return,
        score=score,
        dd_breached=breached,
    )


def run_grid_search(
    df: pd.DataFrame,
    windows: list[Window],
    *,
    max_trials_per_family: int = 200,
    event_cfg: EventConfig | None = None,
    periods_per_year: int = 252,
) -> SearchLedger:
    """Run every (family, params) combination through walk-forward evaluation."""
    per_family: dict[str, int] = defaultdict(int)
    ledger = SearchLedger()
    for family, params in iter_search_space():
        if per_family[family] >= max_trials_per_family:
            continue
        per_family[family] += 1
        ev = evaluate_candidate(df, family, params, windows, event_cfg)
        ledger.trials.append(_eval_to_trial(ev, periods_per_year))

    # DSR works in per-observation Sharpe units, so convert the dispersion of
    # the (annualised) trial Sharpes to per-observation before storing it.
    sharpes = np.array([t.oos_metrics.sharpe for t in ledger.trials], dtype=float)
    ann_std = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    ledger.trial_sharpe_std = ann_std / (periods_per_year**0.5) if periods_per_year > 0 else 0.0
    return ledger
