from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbv_hub_renewal.backtest import (
    BacktestConfig,
    forecast_weeks,
    load_config,
    make_tasks,
    run_development,
    run_lopo,
    run_recovery,
    saturdays,
    score_by_truth_vintage,
    truth_changes,
)
from bdbv_hub_renewal.models import get_model

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


class _MaskAware:
    """Persistence that forecasts masked provinces from the others."""

    name = "mask_aware"
    supports_mask = True

    def forecast(self, task):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        seen = task.target.loc[~task.target["location"].isin(task.mask)]
        # Fill masked provinces with the mean of the visible ones.
        mean = (
            seen.loc[seen["location"] != "CD"]
            .groupby("target_end_date", as_index=False)["total"]
            .mean()
        )
        filled = pd.concat(
            [seen]
            + [mean.assign(location=p, location_name=p) for p in task.mask],
            ignore_index=True,
        )
        return get_model("persistence").forecast(replace(task, target=filled))


def test_tasks_carry_an_empty_mask(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    tasks = make_tasks(
        target.loc[target["vintage"] == "2026-10-09"], delays, config
    )
    assert tasks[0].mask == ()


def test_lopo_masks_one_province_at_a_time(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    models = [get_model("persistence"), _MaskAware()]
    forecasts, scores = run_lopo(
        models,
        target,
        delays,
        config,
        provinces=("CD-IT", "CD-NK"),
        include_reconstructed=False,
    )
    # Persistence cannot honour a mask, so it is skipped.
    assert set(forecasts["model"]) == {"mask_aware"}
    # Only the masked province is kept, from development vintages only.
    assert (forecasts["location"] == forecasts["masked"]).all()
    assert set(forecasts["masked"]) == {"CD-IT", "CD-NK"}
    assert set(forecasts["vintage"]) == {"2026-08-18", "2026-10-09"}
    assert set(scores["location"]) == {"CD-IT", "CD-NK"}
    assert set(scores["truth_vintage"]) == {"2026-10-24"}


def test_lopo_without_mask_aware_models_is_empty(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, scores = run_lopo(
        [get_model("persistence")],
        target,
        delays,
        config,
        include_reconstructed=False,
    )
    assert forecasts.empty
    assert scores.empty


def test_vintage_without_delays_takes_reconstructed_rows(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    from bdbv_hub_renewal.vintages import reconstruct_delays

    assert not (delays["vintage"] == "2026-08-18").any()
    tasks = make_tasks(
        target.loc[target["vintage"] == "2026-08-18"], delays, config
    )
    got = tasks[0].delays
    assert got is not None
    assert len(got) > 0
    assert set(got["vintage"]) == {"2026-08-18"}
    # The next vintage's rows, kept if published by the cut-off.
    nxt = delays.loc[delays["vintage"] == "2026-10-09"]
    want = reconstruct_delays(nxt, "2026-08-18")
    assert len(got) == len(want)
