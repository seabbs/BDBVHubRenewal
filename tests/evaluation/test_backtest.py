from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.releases import cutoff_lags
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    development_tasks,
    energy_tasks,
    forecast_weeks,
    load_config,
    lopo_tasks,
    make_tasks,
    recovery_tasks,
    run_development,
    run_forecasts,
    run_lopo,
    run_recovery,
    saturdays,
    score_tasks,
    truth_changes,
)
from bdbvhubrenewal.forecasters import get_model

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.usefixtures("flat_model")


def test_load_config() -> None:
    cfg = load_config(ROOT / "config" / "backtest.toml")
    assert cfg.n_samples == 100
    # The development set ends at the latest model choice date.
    assert cfg.development_until == "2026-10-28"
    assert not cfg.is_heldout("2026-10-24")
    assert cfg.is_heldout("2026-10-31")
    assert all(pd.Timestamp(d).dayofweek == 5 for d in cfg.simulated_dates)
    # A generous convergence gate on BVDOutbreakSize releases.
    assert cfg.bvd_max_rhat == 1.1
    assert BacktestConfig().bvd_max_rhat == 1.1


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
    forecasts = run_development([get_model("flat")], target, delays, config)
    # Vintages after the model choice date are held out.
    assert (forecasts["vintage"] <= "2026-10-10").all()
    real = forecasts.loc[forecasts["kind"] == "real"]
    assert set(real["vintage"]) == {"2026-08-18", "2026-10-09"}


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


def test_energy_tasks_score_complete_weeks_as_one_vector() -> None:
    truth = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": ["2026-09-26", "2026-10-03", "2026-10-10"],
            "truth": [11.0, 21.0, 25.0],
        }
    )
    forecasts = _forecast_rows(None)
    every = energy_tasks(forecasts, truth, "2026-10-24")
    done = energy_tasks(forecasts, truth, "2026-10-24", complete_after_weeks=3)
    assert list(every["n_cells"]) == [3]
    assert list(done["n_cells"]) == [2]
    # Two samples (10, 20) and (12, 22) against (11, 21).
    spread = np.hypot(2.0, 2.0) / 2
    assert done["energy"].iloc[0] == pytest.approx(np.hypot(1, 1) - spread)


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
    # Each later release with its cut-off, its latest onset day.
    assert by["2026-08-18"] == {
        "2026-10-09": "2026-10-07",
        "2026-10-17": "2026-10-16",
        "2026-10-24": "2026-10-24",
    }
    assert by["2026-10-24"] == {}


def test_truth_changes(target: pd.DataFrame) -> None:
    changes = truth_changes(target, locations=("CD",))
    week = changes.loc[changes["target_end_date"] == "2026-10-10"]
    assert list(week["truth_vintage"]) == [
        "2026-10-09",
        "2026-10-17",
        "2026-10-24",
        "2026-11-01",
    ]
    last = week.iloc[-1]
    assert last["change"] == 0
    assert (week["change"] == week["truth"] - last["truth"]).all()


def test_recovery_on_simulated_vintages(config: BacktestConfig) -> None:
    _, scores, energy = run_recovery([get_model("flat")], config)
    assert set(scores["kind"]) == {"simulated"}
    assert len(scores) > 0
    # One energy score per vintage and scale, over every scored cell.
    assert set(energy["scale"]) == {"national", "provinces"}
    unit = ["vintage", "scale"]
    scale = (
        scores["location"].str.count("-").map({0: "national", 1: "provinces"})
    )
    cells = scores.assign(scale=scale).groupby(unit).size()
    assert (
        energy.set_index(unit)["n_cells"]
        .sort_index()
        .equals(cells.rename("n_cells").sort_index())
    )


class _MaskAware:
    """RandomWalk that forecasts masked provinces from the others."""

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
        return get_model("flat").forecast(replace(task, target=filled))


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
    models = [get_model("flat"), _MaskAware()]
    forecasts, scores = run_lopo(
        models, target, delays, config, provinces=("CD-IT", "CD-NK")
    )
    # Random walk cannot honour a mask, so it is skipped.
    assert set(forecasts["model"]) == {"mask_aware"}
    # Only the masked province is kept, from development vintages only.
    assert (forecasts["location"] == forecasts["masked"]).all()
    assert set(forecasts["masked"]) == {"CD-IT", "CD-NK"}
    assert set(forecasts["vintage"]) == {"2026-08-18", "2026-10-09"}
    assert set(scores["location"]) == {"CD-IT", "CD-NK"}
    latest = target.loc[target["kind"] == "real", "vintage"].max()
    assert set(scores["truth_vintage"]) == {latest}


def test_lopo_without_mask_aware_models_is_empty(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, scores = run_lopo([get_model("flat")], target, delays, config)
    assert forecasts.empty
    assert scores.empty


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


def test_tasks_carry_earlier_target_vintages(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = {t.vintage: t for t in make_tasks(real, delays, config)}
    history = tasks["2026-10-17"].target_history
    assert history is not None
    assert set(history["vintage"]) == {"2026-08-18", "2026-10-09", "2026-10-17"}
    # One vintage's rows can take their history from a fuller table.
    rows = real.loc[real["vintage"] == "2026-10-09"]
    (task,) = make_tasks(rows, delays, config, history=real)
    assert task.target_history is not None
    assert set(task.target_history["vintage"]) == {"2026-08-18", "2026-10-09"}


class _Counting:
    name = "randomwalk"

    def __init__(self) -> None:
        self.calls = 0

    def forecast(self, task):  # type: ignore[no-untyped-def]
        self.calls += 1
        return get_model("flat").forecast(task)


class _Fitted(_Counting):
    def forecast(self, task):  # type: ignore[no-untyped-def]
        self.fit_diagnostics = [{"fit": "all", "max_rhat": 1.01}]
        return super().forecast(task)


def test_run_forecasts_collects_fit_summaries(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    tasks = make_tasks(
        target.loc[target["vintage"] >= "2026-10-17"], delays, config
    )
    fits: list[pd.DataFrame] = []
    run_forecasts([_Fitted()], tasks, fits=fits)
    table = pd.concat(fits)
    assert set(table["vintage"]) == {t.vintage for t in tasks}
    assert (table["max_rhat"] == 1.01).all()


class _LearntDelay(_Counting):
    def forecast(self, task):  # type: ignore[no-untyped-def]
        self.delay_summary = pd.DataFrame(
            {"kind": ["cumulative", "onset_week"], "vintage": task.vintage}
        )
        return super().forecast(task)


def test_run_forecasts_keeps_the_learnt_delay_kind(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    tasks = make_tasks(
        target.loc[target["vintage"] >= "2026-10-17"], delays, config
    )
    learnt: list[pd.DataFrame] = []
    run_forecasts([_LearntDelay()], tasks, learnt=learnt)
    table = pd.concat(learnt)
    assert set(table["kind"]) == {"cumulative", "onset_week"}


class _Masking(_Counting):
    supports_mask = True


def test_lopo_scores_only_the_masked_province(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    model = _Masking()
    kwargs = {"provinces": ("CD-IT",)}
    again, scores = run_lopo([model], target, delays, config, **kwargs)
    calls = model.calls
    assert calls > 0
    assert set(again["masked"]) == {"CD-IT"}
    assert set(again["location"]) == {"CD-IT"}
    assert len(scores) > 0
    tasks = lopo_tasks(target, delays, config, **kwargs)
    assert all(t.mask == ("CD-IT",) for t in tasks)
    assert len(tasks) == calls


def test_backtest_sampler_comes_from_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bdbvhubrenewal.evaluation.backtest import apply_sampler
    from bdbvhubrenewal.forecasters.renewal import (
        SAMPLER_ENV,
        sampler_settings,
    )

    cfg = load_config(ROOT / "config" / "backtest.toml")
    assert cfg.sampler == {
        "num_chains": 2,
        "num_warmup": 500,
        "num_samples": 500,
        "target_accept": 0.8,
        "max_tree_depth": 10,
        "chain_method": "parallel",
    }
    # setenv, not delenv, so the variable apply_sampler writes is undone.
    monkeypatch.setenv(SAMPLER_ENV, "")
    apply_sampler(cfg)
    assert sampler_settings()["num_warmup"] == 500
    # Settings already in the environment win.
    monkeypatch.setenv(SAMPLER_ENV, '{"num_warmup": 50}')
    apply_sampler(cfg)
    assert sampler_settings()["num_warmup"] == 50


def test_scenarios_are_scored_final_and_as_of(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    from bdbvhubrenewal.data import load_health_zones
    from bdbvhubrenewal.evaluation.backtest import run_scenarios

    data = Path(__file__).resolve().parents[2] / "data"
    zones = load_health_zones(data)
    forecasts, scores, energy, truth = run_scenarios(
        [get_model("flat")],
        config,
        target,
        delays,
        zones,
        as_of="2026-10-24",
        dates=("2026-10-10", "2026-10-17"),
        names=("flat",),
    )
    assert set(scores["scenario"]) == {"calibrated_flat"}
    assert set(scores["truth_view"]) == {"final", "as of 2 weeks later"}
    assert {"CD", "CD-IT"} <= set(scores["location"])
    assert set(scores["vintage"]) == {"2026-10-10", "2026-10-17"}
    final = scores.loc[scores["truth_view"] == "final"]
    assert final["complete"].any()
    assert set(forecasts["kind"]) == {"simulated"}
    # Zones: the model's province samples split across zones, scored on
    # the simulated zone counts.
    zone = scores.loc[scores["level"] == "zone"]
    assert set(zone["model"]) == {"flat"}
    assert zone["location"].str.count("-").eq(2).all()
    assert set(zone["truth_view"]) == {"final", "as of 2 weeks later"}
    assert set(scores["level"]) == {"national", "province", "zone"}
    assert set(energy["scale"]) == {"national", "provinces", "zones"}
    assert set(energy["truth_view"]) == {"final", "as of 2 weeks later"}
    # The final counts cover the weeks before the first forecast date too.
    assert set(truth["scenario"]) == {"calibrated_flat"}
    assert truth["target_end_date"].min() < scores["target_end_date"].min()


def test_tasks_carry_the_daily_releases_up_to_their_vintage(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from bdbvhubrenewal.data.tidy import load_daily

    daily = load_daily()
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"] == "2026-10-17"]
    task = make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )[0]
    assert task.daily_history is not None
    assert set(task.daily_history["vintage"]) == {
        "2026-08-18",
        "2026-10-09",
        "2026-10-17",
    }
    assert {"location", "onset_date", "observed", "imputed"} <= set(
        task.daily_history.columns
    )
    # Without daily data, or for another kind, there is none.
    assert make_tasks(rows, delays, BacktestConfig())[0].daily_history is None
    other = make_tasks(
        rows.assign(kind="simulated"), delays, BacktestConfig(), daily=daily
    )[0]
    assert other.daily_history is None


def test_development_tasks_pass_the_daily_releases(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from bdbvhubrenewal.data.tidy import load_daily

    tasks = development_tasks(
        target, delays, BacktestConfig(), daily=load_daily(),
    )  # fmt: skip
    assert all(t.daily_history is not None for t in tasks)


def test_simulated_tasks_carry_daily_releases(config: BacktestConfig) -> None:
    from bdbvhubrenewal.data.tidy import load_daily

    lags = cutoff_lags(load_daily())
    tasks = recovery_tasks(config, lags=lags)
    assert tasks
    for task in tasks:
        hist = task.daily_history
        assert hist is not None
        assert set(hist["kind"]) == {"simulated"}
        assert hist["vintage"].max() == task.vintage
        assert hist["vintage"].nunique() > 4
        last = hist.loc[hist["vintage"] == task.vintage]
        # The vintage's weekly target is cut at the same day.
        assert last["onset_date"].max() <= last["cutoff"].iloc[0]
