"""Acceptance gate — a candidate ships ONLY if it passes ALL criteria.

The thresholds default to the brief's values. They are never loosened to force a
winner: if nothing passes, nothing ships and the dashboard says so. That honesty
is the whole point — a mediocre strategy that survives out-of-sample is worth
far more than a beautiful overfit curve that blows a funded account.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backtest.metrics import Metrics


@dataclass
class GateConfig:
    min_sharpe: float = 1.0
    min_profit_factor: float = 1.3
    min_trades: int = 100
    min_walk_forward_efficiency: float = 0.5
    min_deflated_sharpe: float = 0.95  # DSR is a probability in [0, 1]
    max_bust_probability: float = 0.05
    max_daily_loss: float = 1_000.0  # worst single day must stay within the limit
    require_dd_survival: bool = True


@dataclass
class Criterion:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


@dataclass
class GateResult:
    passed: bool
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [c.name for c in self.criteria if not c.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "criteria": [c.__dict__ for c in self.criteria],
        }


def evaluate_gate(
    metrics: Metrics,
    *,
    n_trials: int,
    trial_sharpe_std: float,
    walk_forward_efficiency: float,
    periods_per_year: int = 252,
    cfg: GateConfig | None = None,
) -> GateResult:
    """Apply every acceptance criterion. ALL must pass for the gate to open."""
    cfg = cfg or GateConfig()

    # Use the Deflated Sharpe Ratio already computed in compute_metrics (it uses
    # the proper per-observation Sharpe, sample length, skew/kurtosis, the trial
    # count and the trial-Sharpe dispersion). Recomputing here would risk unit
    # mismatches, so we consume the authoritative value.
    dsr = metrics.deflated_sharpe

    crit = [
        Criterion("oos_sharpe", metrics.sharpe >= cfg.min_sharpe, metrics.sharpe, cfg.min_sharpe),
        Criterion(
            "profit_factor",
            metrics.profit_factor >= cfg.min_profit_factor,
            metrics.profit_factor,
            cfg.min_profit_factor,
        ),
        Criterion("min_trades", metrics.n_trades >= cfg.min_trades, metrics.n_trades, cfg.min_trades),
        Criterion(
            "walk_forward_efficiency",
            walk_forward_efficiency >= cfg.min_walk_forward_efficiency,
            walk_forward_efficiency,
            cfg.min_walk_forward_efficiency,
        ),
        Criterion(
            "deflated_sharpe",
            dsr >= cfg.min_deflated_sharpe,
            dsr,
            cfg.min_deflated_sharpe,
            note=f"n_trials={n_trials}",
        ),
        Criterion(
            "monte_carlo_survival",
            metrics.monte_carlo_bust_prob <= cfg.max_bust_probability,
            metrics.monte_carlo_bust_prob,
            cfg.max_bust_probability,
        ),
        Criterion(
            "daily_loss_within_limit",
            metrics.worst_daily_loss <= cfg.max_daily_loss,
            metrics.worst_daily_loss,
            cfg.max_daily_loss,
        ),
        Criterion(
            "trailing_dd_survival",
            (not metrics.dd_breached) if cfg.require_dd_survival else True,
            0.0 if metrics.dd_breached else 1.0,
            1.0,
            note="never breached the trailing drawdown",
        ),
    ]
    return GateResult(passed=all(c.passed for c in crit), criteria=crit)
