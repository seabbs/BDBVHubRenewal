from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from bdbv_hub_renewal.evaluation.backtest import (
    BacktestConfig,
    forecast_weeks,
    load_config,
    make_tasks,
    run_development,
    run_lopo,
    run_recovery,
    saturdays,
    score_by_truth_vintage,
    score_tasks,
    truth_changes,
)
from bdbv_hub_renewal.forecasters import get_model

ROOT = Path(__file__).resolve().parents[2]


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


def test_scores_cover_only_weeks_in_the_truth(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, _ = run_development(
        [get_model("persistence")], target, delays, config
    )
    scores = score_by_truth_vintage(forecasts, target)
    # A week past the end of a truth vintage is not scored, not taken as 0.
    weeks = target.groupby("vintage")["target_end_date"].max()
    last = scores["truth_vintage"].map(weeks)
    assert (scores["target_end_date"] <= last).all()
    assert scores["target_end_date"].max() < forecasts["target_end_date"].max()
    index = ["vintage", "location", "target_end_date"]
    truth = target.set_index(index)["total"]
    keys = zip(
        scores["truth_vintage"],
        scores["location"],
        scores["target_end_date"],
        strict=True,
    )
    assert list(scores["truth"]) == [truth[k] for k in keys]


def _forecast_rows(as_of: str | None) -> pd.DataFrame:
    weeks = ["2026-09-26", "2026-10-03", "2026-10-10"]
    return pd.DataFrame(
        {
            "model": "m",
            "kind": "real",
            "vintage": "2026-10-09",
            "reference_date": "2026-10-17",
            "location": "CD",
            "target_end_date": weeks * 2,
            "horizon": [-3, -2, -1] * 2,
            "sample": [1, 1, 1, 2, 2, 2],
            "value": [10.0, 20.0, 30.0, 12.0, 22.0, 32.0],
            "as_of": as_of,
        }
    )


def test_score_tasks_marks_complete_weeks() -> None:
    truth = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": ["2026-09-26", "2026-10-03", "2026-10-10"],
            "truth": [11.0, 21.0, 25.0],
        }
    )
    forecasts = _forecast_rows(None)
    scores = score_tasks(forecasts, truth, "2026-10-24", complete_after_weeks=3)
    # Weeks that ended at least 3 weeks before the truth vintage.
    done = dict(zip(scores["target_end_date"], scores["complete"], strict=True))
    assert done == {"2026-09-26": True, "2026-10-03": True, "2026-10-10": False}
    assert set(scores["scoring"]) == {"as reported"}


def test_score_tasks_scores_vintage_matched_rows() -> None:
    truth = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": ["2026-09-26", "2026-10-03", "2026-10-10"],
            "truth": [11.0, 21.0, 25.0],
        }
    )
    forecasts = pd.concat(
        [
            _forecast_rows(None),
            _forecast_rows("2026-10-17").assign(value=0.0),
            _forecast_rows("2026-10-24").assign(value=25.0),
        ],
        ignore_index=True,
    )
    scores = score_tasks(forecasts, truth, "2026-10-24", complete_after_weeks=3)
    # Final rows against the truth as reported, and the rows predicting
    # this truth vintage against it. Rows for other vintages are left out.
    assert sorted(scores["scoring"].unique()) == [
        "as reported",
        "vintage-matched",
    ]
    matched = scores.loc[scores["scoring"] == "vintage-matched"]
    assert len(matched) == 3
    week = matched.loc[matched["target_end_date"] == "2026-10-10"]
    assert week["ae_median"].iloc[0] == 0.0
    # Forecasts without an as_of column are all final counts.
    plain = score_tasks(
        _forecast_rows(None).drop(columns="as_of"), truth, "2026-10-24"
    )
    assert set(plain["scoring"]) == {"as reported"}
    assert plain["complete"].all()


def test_make_tasks_asks_for_later_vintages(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    vintages = ("2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24")
    tasks = make_tasks(target, delays, config, report_by=vintages)
    by = {t.vintage: t.report_by for t in tasks}
    assert by["2026-08-18"] == vintages[1:]
    assert by["2026-10-24"] == ()


def test_development_scores_both_ways(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    config = dataclasses.replace(config, complete_after_weeks=3)
    forecasts, scores = run_development(
        [get_model("persistence")], target, delays, config
    )
    assert {"as reported", "vintage-matched"} <= set(scores["scoring"])
    assert scores["complete"].any()
    assert not scores["complete"].all()
    by_truth = score_by_truth_vintage(forecasts, target, 3)
    matched = by_truth.loc[by_truth["scoring"] == "vintage-matched"]
    assert set(matched["truth_vintage"]) == {
        "2026-10-09",
        "2026-10-17",
        "2026-10-24",
    }


def test_origins_without_a_lag_estimate_are_marked(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, scores = run_development(
        [get_model("persistence")], target, delays, config
    )
    # Origins with no earlier pair of delay files take the publication
    # lag from a prior informed by later files.
    marked = scores.loc[scores["lag_prior"], ["kind", "vintage"]]
    assert set(map(tuple, marked.drop_duplicates().to_numpy())) == {
        ("real", "2026-08-18"),
        ("real", "2026-10-09"),
        ("reconstructed", "2026-08-29"),
    }
    by_truth = score_by_truth_vintage(forecasts, target)
    assert set(by_truth.loc[by_truth["lag_prior"], "vintage"]) == {
        "2026-08-18",
        "2026-10-09",
    }


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
    from bdbv_hub_renewal.data.reconstruct import reconstruct_delays

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


def test_tasks_carry_earlier_delay_files_for_the_publication_lag(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    tasks = {
        t.vintage: t
        for t in make_tasks(
            target.loc[target["kind"] == "real"],
            delays.assign(kind="real"),
            config,
        )
    }
    history = tasks["2026-10-17"].delay_history
    assert history is not None
    assert set(history["vintage"]) == {"2026-10-09", "2026-10-17"}
    assert tasks["2026-08-18"].delay_history is None
