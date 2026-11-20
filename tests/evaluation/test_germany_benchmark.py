"""The German hospitalisation nowcast benchmark."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bdbvhubrenewal.data.germany import HORIZONS, hub_target, read_triangle
from bdbvhubrenewal.evaluation.germany import (
    PAPER_COLUMNS,
    QUANTILE_LEVELS,
    SCORE_COLUMNS,
    asof_sums,
    germany_forecasts,
    load_germany_config,
    nowcast_dates,
    paper_forecasts,
    relative_scores,
    run_models,
    score_germany,
    team_forecasts,
    with_truth,
)
from bdbvhubrenewal.forecasters import list_models
from nowcasteval.nowcast import rolling_truth

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def forecasts(triangle: pd.DataFrame) -> pd.DataFrame:
    drawn = germany_forecasts(
        ["randomwalk", "randomwalknowcast"],
        triangle,
        ["2021-10-04", "2021-10-05"],
        n_samples=30,
        seed=1,
    )
    return with_truth(drawn, triangle, truth_delay=10)


@pytest.fixture(scope="module")
def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    return score_germany(forecasts, truth_delay=10)


def test_config_gives_the_paper_s_dates_and_an_explicit_model_set() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    every = nowcast_dates(config, held_out=False) + nowcast_dates(
        config, held_out=True
    )
    assert (every[0], every[-1], len(every)) == (
        "2021-11-22",
        "2022-04-29",
        159,
    )
    assert config.models == (
        "randomwalk",
        "randomwalknowcast",
        *(
            f"fm_chronos_{source}_{layout}"
            for layout in ("complete", "ragged")
            for source in ("daily", "weekly")
        ),
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
    )
    assert config.truth_delay == 40
    assert config.submissions
    assert config.baseline == "randomwalknowcast"
    assert config.paper_baseline == "KIT-frozen_baseline"


def test_german_trained_models_run_on_the_held_out_half_only() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    german = ("fm_chronos_daily_ragged_real", "fm_chronos_weekly_ragged_real")
    assert config.held_out_models == german
    # They were fine-tuned on the development half, so never score on it.
    assert not set(german) & set(run_models(config, held_out=False))
    # They join once their checkpoint is pinned and so registered.
    known = {s.name for s in list_models()}
    joined = tuple(m for m in german if m in known)
    assert run_models(config, held_out=True) == (*config.models, *joined)


def test_the_later_half_of_the_dates_is_held_out() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    assert config.split_date == "2022-02-10"
    development = nowcast_dates(config)
    held = nowcast_dates(config, held_out=True)
    assert development[-1] < config.split_date <= held[0]
    assert len(development) - len(held) in (0, 1)


def test_our_quantiles_are_the_samples_quantiles_at_the_hub_s_levels(
    forecasts: pd.DataFrame,
) -> None:
    from bdbvhubrenewal.evaluation.germany import FORECAST_LEVELS

    per = forecasts.groupby(["model", "origin", "location", "horizon"])
    assert (per["quantile"].apply(tuple) == FORECAST_LEVELS).all()
    assert per["value"].is_monotonic_increasing.all()


def test_scores_are_quantile_scores_on_the_d_day_truth(
    triangle: pd.DataFrame, scores: pd.DataFrame
) -> None:
    assert set(SCORE_COLUMNS) <= set(scores.columns)
    assert "log_wis" not in scores.columns
    assert set(scores["origin"]) == {"2021-10-04", "2021-10-05"}
    assert set(scores["horizon"]) <= set(HORIZONS)
    assert set(scores["pool"]) == {"national", "states", "age"}
    ends = sorted(set(scores["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), 10)
    both = scores.merge(truth, on=["location", "target_end_date"])
    assert len(both) == len(scores)
    assert (both["truth_x"] == both["truth_y"]).all()
    assert (both["truth_vintage_x"] == both["truth_vintage_y"]).all()


def test_sums_with_no_release_d_days_on_are_not_scored(
    scores: pd.DataFrame,
) -> None:
    # The fixture ends on 2021-10-20, so sums ending after 2021-10-10
    # have no truth.
    assert scores["target_end_date"].max() <= "2021-10-10"


def test_relative_scores_are_against_the_baseline_given(
    scores: pd.DataFrame,
) -> None:
    for base in ("randomwalknowcast", "randomwalk"):
        rel = relative_scores(scores, ("pool",), base)
        rel = rel.set_index(["pool", "model"])
        assert rel.loc[("national", base), "rel_wis"] == pytest.approx(1)
    assert (rel["n"] > 0).all()
    assert {"coverage_50", "coverage_95", "rel_ae_median"} <= set(rel.columns)


def _team(model: str, target: int, values: dict[float, float], **kw) -> list:
    end = (pd.Timestamp("2021-10-04") + pd.Timedelta(days=target)).date()
    return [
        {
            "location": kw.get("location", "DE"),
            "age_group": kw.get("age_group", "00+"),
            "forecast_date": "2021-10-04",
            "target_end_date": str(end),
            "target": f"{target} day ahead inc hosp",
            "type": "quantile",
            "quantile": q,
            "value": v,
            "pathogen": "COVID-19",
            "model": model,
            "retrospective": False,
        }
        for q, v in values.items()
    ]


def _levels(scale: float) -> dict[float, float]:
    return {q: scale * (1 + i) for i, q in enumerate(QUANTILE_LEVELS)}


@pytest.fixture(scope="module")
def submissions() -> pd.DataFrame:
    no_tenth = {q: v for q, v in _levels(100).items() if q != 0.1}
    rki = _levels(100) | {0.5: 2e9}
    rows = [
        *_team("KIT-simple_nowcast", 0, _levels(100)),
        *_team("KIT-simple_nowcast", -7, _levels(100)),
        *_team("KIT-simple_nowcast", -3, _levels(100)),
        *_team("KIT-simple_nowcast", 0, _levels(10), age_group="80+"),
        *_team("KIT-simple_nowcast", 0, _levels(10), location="DE-BY"),
        *_team("Epiforecasts-independent", 0, no_tenth),
        *_team("RKI-weekly_report", 0, rki),
        *_team("RKI-weekly_report", -1, _levels(90)),
    ]
    mean = _team("KIT-simple_nowcast", 0, {0.5: 1.0})[0] | {
        "type": "mean",
        "quantile": np.nan,
    }
    return pd.DataFrame([*rows, mean])


def test_team_forecasts_are_the_hub_s_quantiles_at_our_horizons(
    submissions: pd.DataFrame,
) -> None:
    got = team_forecasts(submissions)
    assert set(got["model"]) == {"KIT", "Epiforecasts", "RKI"}
    # Every one of the hub's 29 target days is kept.
    assert set(got["horizon"]) == {0, -1, -3, -7}
    kit = got.loc[got["model"] == "KIT"]
    assert set(kit["location"]) == {"DE", "80+", "DE-BY"}
    assert (got.groupby(["model", "location", "horizon"]).size() == 7).all()
    assert set(got["origin"]) == {"2021-10-04"}


def test_team_forecasts_fill_epiforecasts_tenth_percentile(
    submissions: pd.DataFrame,
) -> None:
    # As the paper: the normal through the 2.5% and 25% quantiles.
    got = team_forecasts(submissions).set_index(["model", "quantile"])
    lo, hi = 100.0, 300.0
    sd = (hi - lo) / (norm.ppf(0.25) - norm.ppf(0.025))
    want = lo + sd * (norm.ppf(0.1) - norm.ppf(0.025))
    assert got.loc[("Epiforecasts", 0.1), "value"] == pytest.approx(want)


def test_only_epiforecasts_tenth_percentile_is_filled(
    submissions: pd.DataFrame,
) -> None:
    no_tenth = {q: v for q, v in _levels(100).items() if q != 0.1}
    other = pd.DataFrame(_team("SU-hier_bayes", 0, no_tenth))
    got = team_forecasts(pd.concat([submissions, other]))
    assert got.loc[got["model"] == "SU", "quantile"].nunique() == 6


def test_team_quantiles_are_sorted_within_a_forecast(
    submissions: pd.DataFrame,
) -> None:
    crossed = _levels(100) | {0.1: 50.0}
    other = pd.DataFrame(_team("SU-hier_bayes", 0, crossed))
    got = team_forecasts(pd.concat([submissions, other]))
    su = got.loc[got["model"] == "SU"].sort_values("quantile")
    assert su["value"].is_monotonic_increasing
    assert sorted(su["value"]) == sorted(crossed.values())


def test_unknown_teams_are_an_error(submissions: pd.DataFrame) -> None:
    other = pd.DataFrame(_team("New-model", 0, _levels(100)))
    with pytest.raises(ValueError, match="New-model"):
        team_forecasts(pd.concat([submissions, other]))


def test_team_forecasts_replace_rki_errors_by_the_minus_one_day_nowcast(
    submissions: pd.DataFrame,
) -> None:
    got = team_forecasts(submissions)
    got = got.loc[got["horizon"] == 0].set_index(["model", "quantile"])
    assert got.loc[("RKI", 0.5), "value"] == pytest.approx(90 * 4)
    assert got.loc[("RKI", 0.25), "value"] == pytest.approx(300)


def test_teams_are_scored_as_our_models(
    triangle: pd.DataFrame,
    submissions: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    quantiles = team_forecasts(submissions, ["2021-10-04"])
    teams = score_germany(with_truth(quantiles, triangle, 10), 10)
    assert set(SCORE_COLUMNS) <= set(teams.columns)
    assert list(teams.columns) == list(scores.columns)
    cell = ["origin", "location", "target_end_date"]
    both = teams.merge(scores, on=cell)
    assert len(both)
    assert (both["truth_x"] == both["truth_y"]).all()
    assert (both["truth_vintage_x"] == both["truth_vintage_y"]).all()
    assert set(teams["pool"]) == {"national", "states", "age"}


def test_team_forecasts_keep_the_dates_given(
    submissions: pd.DataFrame,
) -> None:
    assert team_forecasts(submissions, ["2021-10-05"]).empty


def test_paper_table_follows_the_paper_schema(
    forecasts: pd.DataFrame,
) -> None:
    paper = paper_forecasts(forecasts)
    assert list(paper.columns) == list(PAPER_COLUMNS)
    assert set(paper["study"]) == {"germany"}
    assert paper["sample_id"].isna().all()
    ages = paper.loc[paper["stratum"] != "all"]
    assert set(ages["location"]) == {"DE"}
    assert set(ages["stratum"]) == {
        "00-04", "05-14", "15-34", "35-59", "60-79", "80+"
    }  # fmt: skip
    days = pd.to_datetime(paper["target_date"]) - pd.to_datetime(
        paper["origin_date"]
    )
    assert (days.dt.days == paper["horizon"]).all()
    hub = forecasts["quantile"].round(6).isin(QUANTILE_LEVELS)
    assert len(paper) == hub.sum()


def test_asof_sums_are_the_data_each_nowcast_date_saw(
    triangle: pd.DataFrame,
) -> None:
    got = asof_sums(triangle, ["2021-10-04"], weeks=1)
    assert set(got.columns) == {
        "origin", "location", "target_end_date", "total"
    }  # fmt: skip
    assert got["target_end_date"].min() == "2021-09-27"
    assert got["target_end_date"].max() == "2021-10-04"
    ends = sorted(set(got["target_end_date"]))
    want = hub_target(triangle, ends)
    want = want.loc[want["vintage"] == "2021-10-04"]
    both = got.merge(want, on=["location", "target_end_date"])
    assert len(both) == len(got) == len(want)
    assert (both["total_x"] == both["total_y"]).all()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_a_quantile_model_s_german_scores_do_not_depend_on_the_seed(
    triangle: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nowcastfoundation import forecaster as foundation

    class Pipeline:
        quantiles = (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)

        def predict_quantiles(
            self, groups, prediction_length, quantile_levels, **_kw
        ):
            import torch

            level = 1000 * torch.tensor(quantile_levels, dtype=torch.float32)
            return [
                level[None, None, :].expand(
                    np.atleast_2d(g["target"]).shape[0], prediction_length, -1
                )
                for g in groups
            ], None

    monkeypatch.setattr(foundation, "_chronos", lambda *_: Pipeline())
    runs = [
        germany_forecasts(
            ["fm_chronos_weekly_ragged"],
            triangle,
            ["2021-10-04"],
            n_samples=30,
            seed=seed,
        )
        for seed in (1, 2)
    ]
    assert not runs[0].empty
    pd.testing.assert_frame_equal(*runs)


class _Extrapolate:
    """Chronos-2 stand-in: each step continues the context's last slope."""

    quantiles = (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)

    def predict_quantiles(
        self, groups, prediction_length, quantile_levels, **_kw
    ):
        import torch

        out = []
        for g in groups:
            t = np.atleast_2d(g["target"]).astype(np.float64)
            step = np.arange(1, prediction_length + 1)
            path = t[:, -1:] + step * (t[:, -1:] - t[:, -2:-1])
            out.append(
                torch.tensor(path, dtype=torch.float32)[:, :, None].expand(
                    -1, -1, len(quantile_levels)
                )
            )
        return out, None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_each_scored_row_is_the_sum_of_its_own_seven_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nowcastfoundation import forecaster as foundation

    # Counts rise by a known step a day and are reported in full three
    # days on, so the last three days are forecast and a sum one day out
    # of place is off by seven steps. The weeks cross ISO 2020's week 53.
    first, last = pd.Timestamp("2020-10-01"), pd.Timestamp("2021-01-20")
    days = pd.date_range(first, last)
    i = np.arange(len(days))
    places = {
        ("DE-BE", "00+"): 100 + 3 * i,
        ("DE-BY", "00+"): 50 + 2 * i,
        ("DE", "00+"): 150 + 5 * i,
        ("DE", "00-04"): 20 + i,
        ("DE", "05-14"): 40 + 4 * i,
    }
    report = days + pd.Timedelta(days=3)
    triangle = pd.concat(
        pd.DataFrame(
            {
                "reference_date": days.strftime("%Y-%m-%d"),
                "location": loc,
                "age_group": age,
                "confirm": counts,
                "report_date": report.strftime("%Y-%m-%d"),
            }
        )
        for (loc, age), counts in places.items()
    ).reset_index(drop=True)
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Extrapolate())
    drawn = germany_forecasts(
        ["fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"],
        triangle,
        ["2021-01-06"],
        n_samples=10,
        seed=1,
    )
    scores = score_germany(with_truth(drawn, triangle, 10), truth_delay=10)
    count = {
        name: pd.Series(c, index=days)
        for name, c in {
            "DE-BE": places[("DE-BE", "00+")],
            "DE-BY": places[("DE-BY", "00+")],
            "DE": places[("DE", "00+")],
            "00-04": places[("DE", "00-04")],
            "05-14": places[("DE", "05-14")],
        }.items()
    }
    end = pd.to_datetime(scores["target_end_date"])
    want = [
        count[p].loc[e - pd.Timedelta(days=6) : e].sum()
        for p, e in zip(scores["location"], end, strict=True)
    ]
    assert set(scores["model"]) == {
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged",
    }
    assert set(scores["horizon"]) == set(HORIZONS)
    assert (
        end
        == pd.Timestamp("2021-01-06")
        + pd.to_timedelta(scores["horizon"], unit="D")
    ).all()
    np.testing.assert_array_equal(scores["truth"], want)
    np.testing.assert_array_equal(scores["ae_median"], 0)
    np.testing.assert_array_equal(scores["wis"], 0)


def test_a_sample_model_s_german_quantiles_use_the_draws_given(
    triangle: pd.DataFrame,
) -> None:
    runs = [
        germany_forecasts(
            ["randomwalknowcast"],
            triangle,
            ["2021-10-04"],
            n_samples=30,
            seed=seed,
        )
        for seed in (1, 1, 2)
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0]["value"], runs[2]["value"])


def test_held_out_models_wait_until_they_are_registered(caplog) -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    config = dataclasses.replace(
        config, held_out_models=("randomwalk", "not_registered_de")
    )
    with caplog.at_level("WARNING"):
        held = run_models(config, held_out=True)
    assert held == (*config.models, "randomwalk")
    assert "not_registered_de" in caplog.text


def test_models_without_chronos_hold_one_dates_tasks_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bdbvhubrenewal.evaluation import germany as de

    # Each German task holds its own copy of the release history, so a
    # model that is not predicted ahead builds and forecasts one date's
    # tasks before the next date's are built.
    events: list[tuple[str, str]] = []

    def tasks(_triangle, date, _n, _seed, _root, _horizons):
        events.append(("build", date))
        return [date]

    def quantiles(_model, task, *_a, **_k):
        events.append(("forecast", task))
        return pd.DataFrame()

    monkeypatch.setattr(de, "_tasks", tasks)
    monkeypatch.setattr(de, "get_model", lambda name: name)
    monkeypatch.setattr(de, "forecast_quantiles", quantiles)
    dates = ["2021-10-04", "2021-10-05", "2021-10-06"]
    de.germany_forecasts(["randomwalknowcast"], pd.DataFrame(), dates, 10, 1)
    assert events == [e for d in dates for e in (("build", d), ("forecast", d))]


def test_forecasts_carry_the_90_interval_beside_the_hub_levels(
    forecasts: pd.DataFrame,
) -> None:
    from bdbvhubrenewal.evaluation.germany import (
        FORECAST_LEVELS,
        QUANTILE_LEVELS,
        paper_forecasts,
    )

    assert set(FORECAST_LEVELS) == {*QUANTILE_LEVELS, 0.05, 0.95}
    assert set(forecasts["quantile"].round(6)) == set(FORECAST_LEVELS)
    # The paper's table keeps the hub's levels, which it scores.
    paper = paper_forecasts(forecasts)
    assert set(paper["quantile_level"].round(6)) == set(QUANTILE_LEVELS)


def test_scores_add_90_coverage_and_keep_wis_on_the_hub_levels(
    forecasts: pd.DataFrame, scores: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.germany import QUANTILE_LEVELS

    hub = forecasts[forecasts["quantile"].round(6).isin(QUANTILE_LEVELS)]
    on_hub = score_germany(hub, truth_delay=10)
    keys = ["model", "origin", "location", "target_end_date"]
    both = scores.merge(on_hub, on=keys, suffixes=("", "_hub"))
    assert len(both) == len(scores)
    np.testing.assert_allclose(both["wis"], both["wis_hub"])
    # The 90% interval is the 5% and 95% quantiles.
    wide = forecasts.pivot_table(
        index=keys, columns="quantile", values="value"
    ).reset_index()
    got = scores.merge(wide, on=keys)
    inside = (got[0.05] <= got["truth"]) & (got["truth"] <= got[0.95])
    np.testing.assert_array_equal(got["coverage_90"], inside.astype(float))
    assert on_hub["coverage_90"].isna().all()


def test_forecasts_pass_their_horizons_and_store_them_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bdbvhubrenewal.evaluation import germany as de

    seen: list[tuple[int, ...]] = []

    def tasks(_triangle, _date, _n, _seed, horizons=HORIZONS):
        seen.append(tuple(horizons))
        return []

    monkeypatch.setattr(de, "nowcast_tasks", tasks)
    monkeypatch.setattr(de, "get_model", lambda name: name)
    wide = tuple(range(28, -29, -1))
    for h in (wide, HORIZONS):
        de.germany_forecasts(
            ["randomwalknowcast"], pd.DataFrame(), ["2021-10-04"], 10, 1,
            tmp_path, horizons=h,
        )  # fmt: skip
    assert seen == [wide, tuple(HORIZONS)]
    # Each set of horizons has its own stored tasks.
    assert len(list(tmp_path.rglob("*.pkl.gz"))) == 2
