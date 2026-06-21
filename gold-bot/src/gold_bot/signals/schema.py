"""Pydantic models for every dashboard JSON artifact.

These are the single source of truth shared between the back-end (which writes
``docs/data/*.json``) and the front-end contract (which reads them). Tests
validate exported JSON against these models so the dashboard never receives a
malformed payload.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StatusArtifact(BaseModel):
    generated_at: str | None = None
    data_as_of: str | None = None
    search_state: str
    account_headroom_to_trailing_dd: float | None = None
    daily_loss_state: str = "n/a"
    n_trials: int = 0
    note: str = ""


class HoldoutSummary(BaseModel):
    sharpe: float | None = None
    profit_factor: float | None = None


class CurrentStrategyArtifact(BaseModel):
    strategy_id: str
    family: str
    params: dict[str, Any]
    holdout: HoldoutSummary = HoldoutSummary()


class EquityCurveArtifact(BaseModel):
    timestamps: list[str]
    equity: list[float]
    trailing_dd_floor: list[float]


class MetricRow(BaseModel):
    name: str
    oos: float | int | str | None = None
    holdout: float | int | str | None = None


class MetricsArtifact(BaseModel):
    rows: list[MetricRow]


class TradesArtifact(BaseModel):
    rows: list[dict[str, Any]]


class FoldSummary(BaseModel):
    family: str
    oos_sharpe: float | None = None
    oos_profit_factor: float | None = None
    n_trades: int = 0


class WalkForwardArtifact(BaseModel):
    total_trials: int
    deflated_sharpe: float | None = None
    walk_forward_efficiency: float | None = None
    folds: list[FoldSummary] = []
    top_trials: list[dict[str, Any]] = []


class BestCandidateArtifact(BaseModel):
    strategy_id: str
    family: str
    params: dict[str, Any]
    accepted: bool
    oos_sharpe: float | None = None
    oos_profit_factor: float | None = None
    oos_return_pct: float | None = None
    n_trades: int = 0
    gate_failures: list[str] = []


class SignalArtifact(BaseModel):
    generated_at: str
    instrument: str
    timeframe: str
    signal: Literal["LONG", "SHORT", "FLAT"]
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_contracts: int = 0
    risk_dollars: float | None = None
    account_headroom_to_trailing_dd: float | None = None
    confidence_notes: str = ""
    strategy_id: str | None = None
    valid_until: str | None = None
    auto_execution: bool = False
