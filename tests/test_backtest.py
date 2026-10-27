from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbv_forecasts.backtest import (
    BacktestConfig,
    forecast_weeks,
    load_config,
    run_development,
    run_recovery,
    saturdays,
    score_by_truth_vintage,
    truth_changes,
)
from bdbv_forecasts.models import get_model

ROOT = Path(__file__).resolve().parents[1]


def test_load_config() -> None:
    cfg = load_config(ROOT / "config" / "backtest.toml")
    assert cfg.n_samples == 100
    # The development set ends at the latest model choice date.
    assert cfg.development_until == "2026-10-10"
    assert cfg.is_heldout("2026-10-17")
    assert not cfg.is_heldout("2026-10-10")
    assert cfg.reconstructed_dates[0] == "2026-06-06"
    assert all(pd.Timestamp(d).dayofweek == 5 for d in cfg.simulated_dates)


def test_forecast_weeks() -> None:
    weeks = forecast_weeks("2026-10-31", BacktestConfig())
    assert weeks[0] == "2026-10-03"
    assert weeks[-1] == "2026-11-21"
    assert len(weeks) == 8


def test_saturdays() -> None:
    assert saturdays("2026-10-01", "2026-10-17") == (
        "2026-10-03",
        "2026-10-10",
        "2026-10-17",
    )


def test_development_never_uses_heldout_vintages(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, scores = run_development(
        [get_model("persistence")], target, delays, config
    )
    # Vintages after the model choice date are held out.
    assert (forecasts["vintage"] <= "2026-10-10").all()
    real = forecasts.loc[forecasts["kind"] == "real"]
    assert set(real["vintage"]) == {"2026-08-18", "2026-10-09"}
    rec = forecasts.loc[forecasts["kind"] == "reconstructed"]
    assert set(rec["vintage"]) == {"2026-08-29", "2026-10-10"}
    assert set(scores["kind"]) == {"real", "reconstructed"}
    # Every week is scored, up to the last week of the truth vintage.
    assert scores["target_end_date"].max() == "2026-10-24"
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    assert (scores["n_samples"] == 100).all()


def test_score_by_truth_vintage(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, latest = run_development(
        [get_model("persistence")], target, delays, config
    )
    scores = score_by_truth_vintage(forecasts, target)
    # Real forecasts only, each against every later real vintage.
    assert set(scores["kind"]) == {"real"}
    pairs = set(zip(scores["vintage"], scores["truth_vintage"], strict=True))
    assert pairs == {
        ("2026-08-18", "2026-10-09"),
        ("2026-08-18", "2026-10-17"),
        ("2026-08-18", "2026-10-24"),
        ("2026-10-09", "2026-10-17"),
        ("2026-10-09", "2026-10-24"),
    }
    # Against the latest vintage it matches the development scores.
    at_latest = scores.loc[scores["truth_vintage"] == "2026-10-24"]
    real = latest.loc[latest["kind"] == "real"]
    assert len(at_latest) == len(real)
    assert at_latest["wis"].sum() == pytest.approx(real["wis"].sum())


def test_truth_changes(target: pd.DataFrame) -> None:
    changes = truth_changes(target, locations=("CD",))
    week = changes.loc[changes["target_end_date"] == "2026-10-10"]
    assert list(week["truth_vintage"]) == [
        "2026-10-09",
        "2026-10-17",
        "2026-10-24",
    ]
    last = week.iloc[-1]
    assert last["change"] == 0
    assert (week["change"] == week["truth"] - last["truth"]).all()


def test_recovery_on_simulated_vintages(config: BacktestConfig) -> None:
    _, scores = run_recovery([get_model("persistence")], config)
    assert set(scores["kind"]) == {"simulated"}
    assert len(scores) > 0
