"""The disciplined search loop: search -> gate -> holdout.

This is the "keep searching until accepted" loop done *honestly*:

1. Carve off a holdout block (default last 20%) that is touched exactly once.
2. Run the grid over the train+validation block with walk-forward evaluation,
   logging every trial.
3. Rank non-busting candidates by OOS Sharpe; take the best.
4. Apply the full acceptance gate (incl. the multiple-testing-deflated Sharpe).
5. Only if it passes, run the finalist ONCE on the untouched holdout and apply a
   (lighter) "does it still work" gate. If it still passes, it becomes the live
   strategy; otherwise it is discarded and logged.

If nothing is accepted, the outcome says so and no strategy ships.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.event_engine import BacktestResult, EventConfig, run_event_backtest
from ..backtest.metrics import Metrics, compute_metrics, walk_forward_efficiency
from ..risk.manager import DynamicRiskManager
from ..strategies.base import make_strategy_id
from ..strategies.registry import build
from .gating import GateConfig, GateResult, evaluate_gate
from .optimizer import SearchLedger, TrialResult, run_grid_search
from .walk_forward import WFOConfig, evaluate_candidate, make_windows


@dataclass
class SearchOutcome:
    ledger: SearchLedger
    accepted: bool
    best: TrialResult | None = None
    best_oos_result: BacktestResult | None = None
    best_oos_metrics: Metrics | None = None
    gate_result: GateResult | None = None
    holdout_result: BacktestResult | None = None
    holdout_metrics: Metrics | None = None
    holdout_gate: GateResult | None = None
    walk_forward_efficiency: float = 0.0
    search_range: tuple[str, str] = ("", "")
    holdout_range: tuple[str, str] = ("", "")
    note: str = ""

    @property
    def n_trials(self) -> int:
        return self.ledger.n_trials

    @property
    def current_strategy(self) -> dict | None:
        if not self.accepted or self.best is None or self.holdout_metrics is None:
            return None
        return {
            "strategy_id": make_strategy_id(self.best.family, self.best.params),
            "family": self.best.family,
            "params": self.best.params,
            "holdout": {
                "sharpe": round(self.holdout_metrics.sharpe, 3),
                "profit_factor": round(self.holdout_metrics.profit_factor, 3),
            },
        }


def _holdout_gate_config(base: GateConfig, holdout_frac: float) -> GateConfig:
    """A lighter 'does it still work out-of-sample' gate for the holdout block."""
    return GateConfig(
        min_sharpe=0.25,
        min_profit_factor=1.05,
        min_trades=max(5, int(base.min_trades * holdout_frac)),
        min_walk_forward_efficiency=-1e9,  # not meaningful on a single block
        min_deflated_sharpe=0.0,  # single test, no multiple-testing here
        max_bust_probability=0.10,
        max_daily_loss=base.max_daily_loss,
    )


def run_search(
    df: pd.DataFrame,
    *,
    wfo_cfg: WFOConfig | None = None,
    gate_cfg: GateConfig | None = None,
    event_cfg: EventConfig | None = None,
    holdout_frac: float = 0.2,
    periods_per_year: int = 252,
    max_trials_per_family: int = 200,
    verify_top_k: int = 10,
) -> SearchOutcome:
    wfo_cfg = wfo_cfg or WFOConfig()
    gate_cfg = gate_cfg or GateConfig()
    n = len(df)
    holdout_n = int(n * holdout_frac)
    search_df = df.iloc[: n - holdout_n]
    holdout_df = df.iloc[n - holdout_n :]

    windows = make_windows(len(search_df), wfo_cfg)
    if not windows:
        return SearchOutcome(
            ledger=SearchLedger(),
            accepted=False,
            note="insufficient data for the configured walk-forward windows",
        )

    ledger = run_grid_search(
        search_df,
        windows,
        max_trials_per_family=max_trials_per_family,
        event_cfg=event_cfg,
        periods_per_year=periods_per_year,
    )

    ranked = sorted(
        (t for t in ledger.trials if t.score != float("-inf")),
        key=lambda t: t.score,
        reverse=True,
    )
    sr = (str(search_df.index[0].date()), str(search_df.index[-1].date()))
    hr = (str(holdout_df.index[0].date()), str(holdout_df.index[-1].date())) if holdout_n else ("", "")

    if not ranked:
        return SearchOutcome(
            ledger=ledger, accepted=False, search_range=sr, holdout_range=hr,
            note="no scoreable candidate found",
        )

    # Verify the top-K fast-ranked candidates with the EVENT engine (ATR stops +
    # real DD accounting). Take the first that clears the full gate; otherwise
    # keep the best non-busting one for honest display.
    verified: list[tuple] = []
    for t in ranked[: max(1, verify_top_k)]:
        ev = evaluate_candidate(search_df, t.family, t.params, windows, event_cfg)
        wfe = walk_forward_efficiency(t.is_return_pct, ev.oos.total_net / ev.oos.equity.iloc[0])
        metrics = compute_metrics(
            ev.oos,
            periods_per_year=periods_per_year,
            n_trials=ledger.n_trials,
            trial_sharpe_std=ledger.trial_sharpe_std,
            is_return=t.is_return_pct,
        )
        gate = evaluate_gate(
            metrics,
            n_trials=ledger.n_trials,
            trial_sharpe_std=ledger.trial_sharpe_std,
            walk_forward_efficiency=wfe,
            periods_per_year=periods_per_year,
            cfg=gate_cfg,
        )
        verified.append((t, ev, metrics, gate, wfe))
        if gate.passed:
            break

    passing = [v for v in verified if v[3].passed]
    chosen = (
        passing[0]
        if passing
        else max(verified, key=lambda v: (not v[2].dd_breached, v[2].sharpe))
    )
    best, ev, best_metrics, gate_result, wfe = chosen

    outcome = SearchOutcome(
        ledger=ledger,
        accepted=False,
        best=best,
        best_oos_result=ev.oos,
        best_oos_metrics=best_metrics,
        gate_result=gate_result,
        walk_forward_efficiency=wfe,
        search_range=sr,
        holdout_range=hr,
    )

    if not gate_result.passed:
        outcome.note = "best candidate failed the acceptance gate"
        return outcome

    # Finalist passed -> touch the holdout exactly once, using the dynamic risk
    # manager (the same code path the live signal generator uses).
    strat = build(best.family, best.params).generate(holdout_df)
    cfg = event_cfg or EventConfig.from_settings()
    cfg = EventConfig(
        instrument=cfg.instrument, costs=cfg.costs, rules=cfg.rules,
        fixed_size=cfg.fixed_size, use_stops=True, enforce_prop=True,
        risk_manager=DynamicRiskManager(rules=cfg.rules, instrument=cfg.instrument),
    )
    holdout_bt = run_event_backtest(holdout_df, strat, cfg)
    holdout_metrics = compute_metrics(holdout_bt, periods_per_year=periods_per_year)
    holdout_gate = evaluate_gate(
        holdout_metrics,
        n_trials=1,
        trial_sharpe_std=0.0,
        walk_forward_efficiency=1.0,
        periods_per_year=periods_per_year,
        cfg=_holdout_gate_config(gate_cfg, holdout_frac),
    )

    outcome.holdout_result = holdout_bt
    outcome.holdout_metrics = holdout_metrics
    outcome.holdout_gate = holdout_gate
    outcome.accepted = holdout_gate.passed
    outcome.note = "accepted" if holdout_gate.passed else "finalist failed on holdout"
    return outcome
