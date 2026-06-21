"""Performance & robustness metrics.

Every metric is a small, independently testable pure function so it can be
checked against a hand-computed fixture (see tests/test_metrics.py). The
aggregator :func:`compute_metrics` bundles them into a :class:`Metrics` record
for a single :class:`BacktestResult`.

Two metrics deserve special attention because they are the anti-overfitting
backbone of the project:

* **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado. Discounts a
  strategy's Sharpe for (a) the number of trials it took to find it, (b)
  non-normal returns (skew/kurtosis), and (c) sample length. A strategy that
  only looks good because we tried thousands of variants gets a low DSR.

* **Monte-Carlo bust probability** — resamples the trade PnL stream and applies
  the real trailing-drawdown mechanic to each path, estimating the probability
  the account would have blown up. The gate (Phase 5) requires high survival.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from config.settings import AccountRules
from scipy.stats import norm

from ..risk.prop_rules import TrailingDrawdown

if TYPE_CHECKING:
    from .event_engine import BacktestResult

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Return / ratio metrics
# --------------------------------------------------------------------------- #
def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Simple per-period returns from an equity curve."""
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> tuple[float, float]:
    """(dollars, fraction) total return over the equity curve."""
    if len(equity) < 2:
        return 0.0, 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    return end - start, (end / start - 1.0) if start > 0 else 0.0


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return -1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe (risk-free 0). Uses sample std (ddof=1)."""
    r = returns.to_numpy(dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < _EPS:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sortino: mean / downside deviation (target 0)."""
    r = returns.to_numpy(dtype=float)
    if len(r) < 2:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd < _EPS:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """(dollars, fraction) maximum drawdown — both reported as positive magnitudes."""
    if len(equity) == 0:
        return 0.0, 0.0
    eq = equity.to_numpy(dtype=float)
    run_max = np.maximum.accumulate(eq)
    dd_dollars = eq - run_max
    dd_pct = eq / run_max - 1.0
    return float(-dd_dollars.min()), float(-dd_pct.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    _, mdd_pct = max_drawdown(equity)
    if mdd_pct < _EPS:
        return 0.0
    return cagr(equity, periods_per_year) / mdd_pct


def trailing_dd_min_headroom(
    equity: pd.Series, amount: float, ratchet_mode: str = "eod"
) -> float:
    """Smallest dollar gap between equity and the trailing-DD floor over the path.

    Negative means the floor was breached. Treats each bar as an end-of-day mark
    (correct for daily bars), reusing the real :class:`TrailingDrawdown`.
    """
    if len(equity) == 0:
        return amount
    tdd = TrailingDrawdown(float(equity.iloc[0]), amount, ratchet_mode)
    worst = float("inf")
    for value in equity.to_numpy(dtype=float):
        worst = min(worst, tdd.headroom(value))
        tdd.on_day_close(value)
    return worst


def worst_daily_loss(pnl: pd.Series) -> float:
    """Most negative single-day PnL (returned as a positive magnitude)."""
    if len(pnl) == 0:
        return 0.0
    daily = pnl.groupby(pnl.index.date).sum()
    return float(max(0.0, -daily.min()))


# --------------------------------------------------------------------------- #
# Trade-based metrics
# --------------------------------------------------------------------------- #
def _trade_pnls(trades: list) -> np.ndarray:
    return np.array([t.net_pnl for t in trades], dtype=float)


def profit_factor(trades: list) -> float:
    p = _trade_pnls(trades)
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses < _EPS:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(trades: list) -> float:
    p = _trade_pnls(trades)
    if len(p) == 0:
        return 0.0
    return float(np.mean(p > 0))


def payoff_ratio(trades: list) -> float:
    p = _trade_pnls(trades)
    wins = p[p > 0]
    losses = p[p < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / -losses.mean())


def expectancy(trades: list) -> float:
    p = _trade_pnls(trades)
    return float(p.mean()) if len(p) else 0.0


def longest_losing_streak(trades: list) -> int:
    streak = best = 0
    for t in trades:
        if t.net_pnl < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def avg_trade_duration_hours(trades: list) -> float:
    if not trades:
        return 0.0
    secs = [(t.exit_time - t.entry_time).total_seconds() for t in trades]
    return float(np.mean(secs) / 3600.0)


def exposure(position: pd.Series) -> float:
    """Fraction of bars spent in a position (time in market)."""
    if len(position) == 0:
        return 0.0
    return float(np.mean(position.to_numpy() != 0))


# --------------------------------------------------------------------------- #
# Deflated / Probabilistic Sharpe Ratio
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sr_hat: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """PSR: probability the true (per-obs) Sharpe exceeds ``sr_benchmark``.

    ``sr_hat`` and ``sr_benchmark`` are per-observation (NOT annualised).
    ``kurt`` is the non-excess kurtosis (normal == 3).
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom <= _EPS:
        return 0.0
    z = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(trial_sharpe_std: float, n_trials: int) -> float:
    """Expected maximum (per-obs) Sharpe across ``n_trials`` independent trials.

    Bailey & López de Prado's closed form. This is the benchmark the Deflated
    Sharpe Ratio must beat.
    """
    if n_trials <= 1 or trial_sharpe_std <= _EPS:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trial_sharpe_std * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    trial_sharpe_std: float,
) -> float:
    """DSR = PSR evaluated against the expected-maximum-Sharpe benchmark."""
    sr_star = expected_max_sharpe(trial_sharpe_std, n_trials)
    return probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, sr_star)


# --------------------------------------------------------------------------- #
# Monte-Carlo bust probability
# --------------------------------------------------------------------------- #
def monte_carlo_bust_probability(
    trade_pnls: np.ndarray | list[float],
    account_size: float,
    trailing_dd: float,
    n_paths: int = 2000,
    method: str = "bootstrap",
    seed: int = 7,
) -> float:
    """Probability a resampled trade sequence would breach the trailing DD.

    ``method``: "bootstrap" (sample with replacement) or "permutation" (reorder
    the same trades). Each path applies the real trailing-DD mechanic at trade
    granularity. Returns the fraction of paths that bust (0..1).
    """
    pnls = np.asarray(trade_pnls, dtype=float)
    if len(pnls) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    busts = 0
    for _ in range(n_paths):
        if method == "permutation":
            path = rng.permutation(pnls)
        else:
            path = rng.choice(pnls, size=len(pnls), replace=True)
        tdd = TrailingDrawdown(account_size, trailing_dd, ratchet_mode="eod")
        equity = account_size
        for pnl in path:
            equity += pnl
            if tdd.check(equity):
                busts += 1
                break
            tdd.on_day_close(equity)
    return busts / n_paths


def walk_forward_efficiency(is_return: float, oos_return: float) -> float:
    """OOS return / IS return. < 1 means the edge decays out of sample."""
    if abs(is_return) < _EPS:
        return 0.0
    return oos_return / is_return


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    total_return_dollars: float
    total_return_pct: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd_dollars: float
    max_dd_pct: float
    trailing_dd_min_headroom: float
    worst_daily_loss: float
    profit_factor: float
    win_rate: float
    payoff_ratio: float
    expectancy: float
    avg_trade_duration_hours: float
    exposure: float
    n_trades: int
    longest_losing_streak: int
    deflated_sharpe: float
    walk_forward_efficiency: float | None
    monte_carlo_bust_prob: float
    dd_breached: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    result: BacktestResult,
    *,
    periods_per_year: int = 252,
    account_rules: AccountRules | None = None,
    n_trials: int = 1,
    trial_sharpe_std: float = 0.0,
    is_return: float | None = None,
    mc_paths: int = 2000,
) -> Metrics:
    """Bundle the full metric suite for one backtest result."""
    rules = account_rules or AccountRules()
    equity = result.equity
    pnl = result.pnl
    trades = result.trades
    rets = returns_from_equity(equity)

    tr_dollars, tr_pct = total_return(equity)
    mdd_dollars, mdd_pct = max_drawdown(equity)

    sr_ann = sharpe_ratio(rets, periods_per_year)
    # Per-observation Sharpe for the (de)flated calc.
    sr_obs = sr_ann / np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    skew = float(rets.skew()) if len(rets) > 2 else 0.0
    kurt = float(rets.kurt() + 3.0) if len(rets) > 3 else 3.0  # pandas kurt is excess
    dsr = deflated_sharpe_ratio(sr_obs, len(rets), skew, kurt, n_trials, trial_sharpe_std)

    bust = monte_carlo_bust_probability(
        _trade_pnls(trades), rules.account_size, rules.trailing_drawdown, n_paths=mc_paths
    )

    wfe = None
    if is_return is not None:
        wfe = walk_forward_efficiency(is_return, tr_pct)

    return Metrics(
        total_return_dollars=tr_dollars,
        total_return_pct=tr_pct,
        cagr=cagr(equity, periods_per_year),
        sharpe=sr_ann,
        sortino=sortino_ratio(rets, periods_per_year),
        calmar=calmar_ratio(equity, periods_per_year),
        max_dd_dollars=mdd_dollars,
        max_dd_pct=mdd_pct,
        trailing_dd_min_headroom=trailing_dd_min_headroom(equity, rules.trailing_drawdown),
        worst_daily_loss=worst_daily_loss(pnl),
        profit_factor=profit_factor(trades),
        win_rate=win_rate(trades),
        payoff_ratio=payoff_ratio(trades),
        expectancy=expectancy(trades),
        avg_trade_duration_hours=avg_trade_duration_hours(trades),
        exposure=exposure(result.position),
        n_trades=len(trades),
        longest_losing_streak=longest_losing_streak(trades),
        deflated_sharpe=dsr,
        walk_forward_efficiency=wfe,
        monte_carlo_bust_prob=bust,
        dd_breached=bool(result.dd_dead),
    )
