"""Tests for configuration: defaults, env overrides, instrument economics."""

from __future__ import annotations

import pytest
from config.settings import INSTRUMENTS, Settings


def test_defaults_match_50k_plan() -> None:
    s = Settings()
    rules = s.account_rules
    assert rules.account_size == 50_000
    assert rules.trailing_drawdown == 2_000
    assert rules.daily_loss_limit == 1_000
    assert rules.profit_target == 3_000
    assert rules.max_contracts == 10


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLDBOT_TRAILING_DRAWDOWN", "2000")
    monkeypatch.setenv("GOLDBOT_MAX_CONTRACTS", "5")
    s = Settings()
    assert s.account_rules.trailing_drawdown == 2_000
    assert s.account_rules.max_contracts == 5


def test_instrument_economics() -> None:
    gc = INSTRUMENTS["GC"]
    mgc = INSTRUMENTS["MGC"]
    # GC: 0.10 tick worth $10 -> $100 per full point.
    assert gc.tick_value == 10.0
    assert gc.point_value == 100.0
    assert gc.ticks_per_point == pytest.approx(10.0)
    # 10 MGC == 1 GC in dollar terms.
    assert mgc.point_value * 10 == gc.point_value
    assert mgc.tick_value * 10 == gc.tick_value


def test_cost_model_slippage_floor() -> None:
    s = Settings()
    assert s.cost_model.slippage_ticks >= 1.0  # no frictionless fills
