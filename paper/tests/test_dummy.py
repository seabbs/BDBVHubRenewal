import pandas as pd
import pytest

from paperfigs.dummy import STUDY_SETUP, dummy_study
from paperfigs.schema import (
    STUDIES,
    validate_forecasts,
    validate_observations,
)
from paperfigs.scores import STUDY_BASELINES


@pytest.fixture(scope="module", params=STUDIES)
def study(request):
    fc, obs = dummy_study(request.param, seed=1, n_samples=20)
    return request.param, fc, obs


def test_dummy_tables_follow_the_schema(study) -> None:
    name, fc, obs = study
    validate_forecasts(fc)
    validate_observations(obs)
    assert set(fc["study"]) == {name}
    assert set(obs["study"]) == {name}


def test_dummy_models_include_baseline_and_foundation_models(study) -> None:
    name, fc, _ = study
    models = set(fc["model"])
    assert STUDY_BASELINES[name] in models
    assert {"randomwalk", "randomwalknowcast"} <= models
    assert any(
        m.startswith("fm_chronos_") and m.endswith("_ragged") for m in models
    )
    assert any(m.endswith("_ft") for m in models)
    if name != "synthetic":
        assert any(m.startswith("hub_") for m in models)


def test_dummy_horizons_and_dates_agree(study) -> None:
    name, fc, _ = study
    setup = STUDY_SETUP[name]
    # Days per unit of the horizon.
    days = 1 if setup.horizon_days else setup.step_days
    steps = fc["horizon"] * days // setup.step_days
    assert steps.between(*setup.horizons).all()
    implied = fc["origin_date"] + fc["horizon"] * pd.Timedelta(days=days)
    assert (implied == fc["target_date"]).all()


def test_weekly_dummy_targets_end_on_saturdays(study) -> None:
    name, fc, _ = study
    if STUDY_SETUP[name].step_days == 7:
        assert (fc["target_date"].dt.day_of_week == 5).all()


def test_dummy_data_as_of_origin_is_at_most_final(study) -> None:
    _, fc, obs = study
    final_date = obs["as_of_date"].max()
    final = obs.loc[obs["as_of_date"] == final_date]
    early = obs.loc[obs["as_of_date"] < final_date]
    joined = early.merge(
        final,
        on=["study", "location", "stratum", "target_date"],
        suffixes=("", "_f"),
    )
    assert len(joined) == len(early)
    assert (joined["value"] <= joined["value_f"]).all()
    assert (joined["target_date"] <= joined["as_of_date"]).all()
    assert set(fc["origin_date"]) <= set(early["as_of_date"])


def test_dummy_truth_matches_final_data(study) -> None:
    _, fc, obs = study
    final = obs.loc[obs["as_of_date"] == obs["as_of_date"].max()]
    truth = fc.drop_duplicates(["location", "stratum", "target_date"])
    joined = truth.merge(
        final, on=["study", "location", "stratum", "target_date"]
    )
    assert len(joined) == len(truth)
    assert (joined["truth"] == joined["value_y"]).all()


def test_dummy_is_reproducible() -> None:
    a, _ = dummy_study("ebola", seed=3, n_samples=10)
    b, _ = dummy_study("ebola", seed=3, n_samples=10)
    c, _ = dummy_study("ebola", seed=4, n_samples=10)
    pd.testing.assert_frame_equal(a, b)
    assert not a["value"].equals(c["value"])


def test_dummy_forms_follow_the_study(study) -> None:
    name, fc, _ = study
    if name == "germany":
        assert fc["sample_id"].isna().all()
    else:
        assert fc["quantile_level"].isna().all()
        assert (
            fc.groupby(["model", "location", "origin_date", "horizon"])[
                "sample_id"
            ]
            .nunique()
            .eq(20)
            .all()
        )


def test_germany_has_age_strata_for_the_nation(study) -> None:
    name, fc, obs = study
    strata = set(fc["stratum"])
    if name == "germany":
        assert len(strata) > 1
        ages = fc.loc[fc["stratum"] != "all", "location"]
        assert set(ages) == {"DE"}
        assert set(obs["stratum"]) == strata
    else:
        assert strata == {"all"}


def test_dummy_has_complete_ragged_and_fine_tuned_chronos(study) -> None:
    _, fc, _ = study
    models = set(fc["model"])
    for m in models:
        if m.startswith("fm_chronos_") and m.endswith("_ragged_ft"):
            family = m.removesuffix("_ragged_ft")
            assert {f"{family}_complete", f"{family}_ragged"} <= models
