"""The grouping arms: the paired arm and the merged German task."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.groupings import (
    GROUPING_ARMS,
    arm_name,
    merged_task,
    pair_and_sum,
)
from bdbvhubrenewal.forecasters.base import NATIONAL, ForecastTask

ZONES = {"P1-a": "P1", "P1-b": "P1", "P2-a": "P2"}
WEEKS = ["2026-10-10", "2026-10-17"]
N = 200


@pytest.fixture
def draws() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {
            "model": "m:bottom",
            "origin": "2026-10-17",
            "location": z,
            "target_end_date": w,
            "sample": s + 1,
            "value": float(rng.poisson(20)),
        }
        for z in ZONES
        for w in WEEKS
        for s in range(N)
    ]
    return pd.DataFrame(rows)


def _no_truth() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["location", "target_end_date", "truth", "truth_vintage"]
    )


def test_arms_are_the_fitted_groupings_and_paired() -> None:
    assert GROUPING_ARMS == ("joint", "level", "bottom", "paired")
    assert arm_name("m", "level") == "m:level"


def test_paired_keeps_each_cells_draws_and_sums_up(draws) -> None:
    out, fits = pair_and_sum(draws, _no_truth(), 3, parent=ZONES.get)
    assert len(fits) == 1
    keys = ["location", "target_end_date"]
    zones = out.loc[out["location"].isin(set(ZONES))]
    for (place, week), own in zones.groupby(keys):
        before = draws.loc[
            (draws["location"] == place) & (draws["target_end_date"] == week)
        ]
        assert sorted(own["value"]) == sorted(before["value"])
    wide = out.pivot_table(
        index=["target_end_date", "sample"], columns="location", values="value"
    )
    np.testing.assert_allclose(wide["P1"], wide["P1-a"] + wide["P1-b"])
    np.testing.assert_allclose(wide["P2"], wide["P2-a"])
    np.testing.assert_allclose(wide[NATIONAL], wide["P1"] + wide["P2"])
    # The copula ties the zones' draws together, unlike independent draws.
    assert np.corrcoef(wide["P1-a"], wide["P1-b"])[0, 1] > 0.3


def test_sums_are_raised_to_their_floors(draws) -> None:
    floors = pd.DataFrame(
        {
            "origin": "2026-10-17",
            "location": ["P1", NATIONAL],
            "target_end_date": WEEKS[0],
            "floor": [1000.0, 5000.0],
        }
    )
    out, _ = pair_and_sum(
        draws, _no_truth(), 3, parent=ZONES.get, floors=floors
    )
    first = out.loc[out["target_end_date"] == WEEKS[0]]
    assert (first.loc[first["location"] == "P1", "value"] == 1000).all()
    assert (first.loc[first["location"] == NATIONAL, "value"] == 5000).all()


def test_no_parent_sums_the_places_to_the_country(draws) -> None:
    out, _ = pair_and_sum(draws, _no_truth(), 3)
    assert set(out["location"]) == {*ZONES, NATIONAL}


def _task(places: tuple[str, ...]) -> ForecastTask:
    rows = pd.DataFrame(
        {"location": [NATIONAL, *places], "total": range(len(places) + 1)}
    )
    return ForecastTask(
        vintage="2022-01-01",
        reference_date=pd.Timestamp("2022-01-01").date(),
        kind="real",
        cutoff="2022-01-01",
        target=rows,
        delays=None,
        locations=places,
        target_end_dates=("2022-01-01",),
        n_samples=10,
        seed=1,
        target_history=rows,
        daily_history=rows,
    )


def test_merged_task_holds_both_tasks_places_once() -> None:
    task = merged_task(_task(("S1", "S2")), _task(("A1",)))
    assert task.locations == ("S1", "S2", "A1")
    for table in (task.target, task.target_history, task.daily_history):
        assert table["location"].tolist() == [NATIONAL, "S1", "S2", "A1"]


def test_options_are_named_for_the_paper() -> None:
    from bdbvhubrenewal.evaluation.groupings import split_arm

    assert split_arm("m:joint") == ("m", "joint")
    assert split_arm("m@abc:bottom") == ("m@abc", "zones_summed")
    assert split_arm("m:paired") == ("m", "zones_copula")
    assert split_arm("randomwalknowcast") == ("randomwalknowcast", "reference")


def test_aggregation_rows_keep_cells_the_reference_scored() -> None:
    from bdbvhubrenewal.evaluation.groupings import (
        AGGREGATION_COLUMNS,
        aggregation_rows,
    )

    scores = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "m:joint", "m:joint"],
            "origin": "2026-10-17",
            "location": ["CD", "CD", "CD-KS"],
            "target_end_date": "2026-10-10",
            "level": ["national", "national", "provinces"],
            "wis": [2.0, 1.0, 3.0],
            "log_wis": [0.2, 0.1, 0.3],
            "cover50": [1.0, 0.0, 1.0],
            "cover95": [1.0, 1.0, 1.0],
        }
    )
    out = aggregation_rows(scores, "ebola")
    assert list(out.columns) == list(AGGREGATION_COLUMNS)
    assert out["location"].tolist() == ["CD", "CD"]
    assert set(out["option"]) == {"reference", "joint"}
    assert (out["study"] == "ebola").all()
    assert out["target_date"].tolist() == ["2026-10-10"] * 2


def test_sample_coverage_is_the_central_interval() -> None:
    from bdbvhubrenewal.evaluation.groupings import sample_coverage

    samples = pd.DataFrame(
        {
            "model": "m",
            "origin": "o",
            "location": "x",
            "target_end_date": "w",
            "sample": range(1, 101),
            "value": np.arange(100.0),
        }
    )
    truth = pd.DataFrame(
        {"location": ["x"], "target_end_date": ["w"], "truth": [98.5]}
    )
    out = sample_coverage(samples, truth, 0.95)
    assert out["cover95"].tolist() == [0.0]
    out = sample_coverage(samples, truth.assign(truth=50.0), 0.95)
    assert out["cover95"].tolist() == [1.0]


def _paper_rows(
    model: str, option: str, wis: float, n: int = 2
) -> pd.DataFrame:
    """Rows of the paper's aggregation table, without a checkpoint."""
    return pd.DataFrame(
        {
            "study": "ebola",
            "model": model,
            "option": option,
            "level": "national",
            "location": "CD",
            "origin_date": "2026-10-17",
            "target_date": [f"2026-10-{d:02d}" for d in range(3, 3 + 7 * n, 7)],
            "wis": wis,
            "log_wis": wis / 10,
            "cover50": 1.0,
            "cover95": 1.0,
        }
    )


@pytest.mark.parametrize(
    ("label", "env", "tag"),
    [
        (
            "slots-0123456789ab",
            "slots-0123456789ab",
            "chronos-slots-0123456789ab",
        ),
        ("de129103cacc", "de129103cacc", "chronos-ft-de129103cacc"),
    ],
)
def test_a_checkpoint_label_gives_its_env_and_release(
    label: str, env: str, tag: str
) -> None:
    from bdbvhubrenewal.evaluation.groupings import checkpoint_label

    assert checkpoint_label(label) == (env, tag)


@pytest.mark.parametrize("label", ["latest", "slots-", "chronos-ft-abc", ""])
def test_a_label_that_is_no_checkpoint_fails(label: str) -> None:
    from bdbvhubrenewal.evaluation.groupings import checkpoint_label

    with pytest.raises(ValueError, match="checkpoint"):
        checkpoint_label(label)


def test_checkpoint_rows_name_the_model_and_keep_the_reference_cells() -> None:
    from bdbvhubrenewal.evaluation.groupings import (
        AGGREGATION_COLUMNS,
        checkpoint_rows,
    )

    reference = _paper_rows("randomwalknowcast", "reference", 2.0, n=1)
    m = "fm_chronos_daily_ragged_ft"
    scores = pd.DataFrame(
        {
            "model": f"{m}@slots-abc:paired",
            "origin": "2026-10-17",
            "location": "CD",
            "target_end_date": ["2026-10-03", "2026-10-10"],
            "level": "national",
            "wis": [1.0, 5.0],
            "log_wis": [0.1, 0.5],
            "cover50": 1.0,
            "cover95": 0.0,
        }
    )
    out = checkpoint_rows(scores, reference, "ebola", "slots-abc")
    assert list(out.columns) == [
        *AGGREGATION_COLUMNS,
        "checkpoint",
        "slots_rule",
    ]
    assert out["slots_rule"].tolist() == ["cases-only"]
    assert out["target_date"].tolist() == ["2026-10-03"]
    assert out["model"].tolist() == [m]
    assert out["option"].tolist() == ["zones_copula"]
    assert out["checkpoint"].tolist() == ["slots-abc"]


def test_merged_rows_keep_zero_shot_and_replace_old_finetunes() -> None:
    from bdbvhubrenewal.evaluation.groupings import merge_checkpoint_rows

    existing = pd.concat(
        [
            _paper_rows("randomwalknowcast", "reference", 2.0),
            _paper_rows("fm_chronos_daily_ragged", "joint", 1.0),
            _paper_rows("fm_chronos_daily_ragged_ft", "joint", 9.0),
            _paper_rows("fm_chronos_daily_ragged_ft_n1000", "joint", 9.0),
        ],
        ignore_index=True,
    )
    new = _paper_rows("fm_chronos_daily_ragged_ft", "joint", 0.5).assign(
        checkpoint="slots-abc"
    )
    out = merge_checkpoint_rows(existing, new, "slots-abc")
    zero = out.loc[out["checkpoint"] == "zero-shot"]
    assert set(zero["model"]) == {
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
    }
    pd.testing.assert_frame_equal(
        zero.drop(columns=["checkpoint", "slots_rule"]).reset_index(drop=True),
        existing.iloc[:4].reset_index(drop=True),
    )
    assert set(zero["slots_rule"]) == {"none"}
    ft = out.loc[out["checkpoint"] == "slots-abc"]
    assert ft["wis"].tolist() == [0.5, 0.5]
    # A second run of the same label replaces its rows; others are kept.
    other = new.assign(checkpoint="slots-def", wis=0.7)
    again = merge_checkpoint_rows(
        merge_checkpoint_rows(out, other, "slots-def"), new, "slots-abc"
    )
    assert again.groupby("checkpoint").size().to_dict() == {
        "slots-abc": 2,
        "slots-def": 2,
        "zero-shot": 4,
    }


def test_rows_must_match_the_reference_in_number() -> None:
    from bdbvhubrenewal.evaluation.groupings import check_checkpoint_rows

    rows = pd.concat(
        [
            _paper_rows("randomwalknowcast", "reference", 2.0),
            _paper_rows("fm_chronos_daily_ragged_ft", "joint", 0.5),
        ],
        ignore_index=True,
    ).assign(checkpoint=["zero-shot"] * 2 + ["slots-abc"] * 2)
    check_checkpoint_rows(rows)
    with pytest.raises(ValueError, match="slots-abc"):
        check_checkpoint_rows(rows.iloc[:3])


def _block(cased: list[bool], lagged: list[str], steps: int = 6) -> dict:
    """A ragged group of places p0, p1, ...; `cased` ones have counts."""
    places = [f"p{i}" for i in range(len(cased))]
    target = np.array([[float(c)] * steps for c in cased])
    names = [f"{p}_lag_{d}" for p in lagged for d in range(2)]
    return {
        "target": target,
        "past_covariates": {n: np.zeros(steps) for n in names},
        "future_covariates": {n: np.zeros(2) for n in names},
        "places": places,
    }


def test_slots_of_a_block_within_the_slots_are_the_usual_ones() -> None:
    from bdbvhubrenewal.evaluation.groupings import cased_group_slots
    from bdbvhubrenewal.forecasters.foundation import _group_slots

    group = _block([True, False, True], ["p2", "p0"])
    assert (
        cased_group_slots(group, 8).tolist() == _group_slots(group, 8).tolist()
    )


def test_a_block_over_the_slots_gives_slots_to_places_with_cases_only() -> None:
    from bdbvhubrenewal.evaluation.groupings import cased_group_slots

    group = _block([False, True, False, True, False], ["p3", "p1"])
    ids = cased_group_slots(group, 3)
    targets, lags = ids[:5], ids[5:]
    assert (targets[[0, 2, 4]] == -1).all()
    assert targets[1] >= 0
    assert targets[3] >= 0
    assert targets[1] != targets[3]
    # Chronos-2's rows: the lags by sorted name, p1's then p3's.
    assert lags.tolist() == [targets[1]] * 2 + [targets[3]] * 2


def test_more_places_with_cases_than_slots_fails() -> None:
    from bdbvhubrenewal.evaluation.groupings import cased_group_slots

    with pytest.raises(ValueError, match="slots"):
        cased_group_slots(_block([True] * 4, []), 3)


def test_slots_rule_of_each_row() -> None:
    from bdbvhubrenewal.evaluation.groupings import slots_rule

    rows = pd.DataFrame(
        {
            "study": ["ebola"] * 5 + ["germany"],
            "option": ["joint", "level", "level", "zones_summed",
                       "zones_copula", "joint"],
            "level": ["national", "provinces", "zones", "national",
                      "provinces", "national"],
        }
    )  # fmt: skip
    assert slots_rule(rows, slots=True).tolist() == [
        "cases-only", "all", "cases-only", "cases-only", "cases-only", "all",
    ]  # fmt: skip
    assert set(slots_rule(rows, slots=False)) == {"none"}


def test_slots_over_the_slots_follow_chronos_rows_with_hub_names() -> None:
    # Chronos-2 stacks covariates by sorted name, and "-" sorts before
    # "_": CD-KN's lags come before CD's. CD-EQ has no cases and no lags.
    from bdbvhubrenewal.evaluation.groupings import cased_group_slots

    names = ["CD_lag_0", "CD_lag_1", "CD-KN_lag_0", "CD-KN_lag_1"]
    group = {
        "target": np.array([[3.0] * 4, [1.0] * 4, [0.0] * 4]),
        "past_covariates": {n: np.zeros(4) for n in names},
        "future_covariates": {n: np.zeros(2) for n in names},
        "places": ["CD", "CD-KN", "CD-EQ"],
    }
    ids = cased_group_slots(group, 2)
    cd, kn, eq = ids[:3]
    assert eq == -1
    assert {cd, kn} == {0, 1}
    assert ids[3:].tolist() == [kn, kn, cd, cd]


def test_the_runner_checkpoint_overrides_the_pin(monkeypatch) -> None:
    # config/finetune.toml pins 2c2b22a03107 on this branch; the runner's
    # BDBV_CHRONOS_FT must win for every model and arm it fits.
    from bdbvhubrenewal.evaluation import groupings
    from bdbvhubrenewal.forecasters import foundation, get_model

    @functools.cache
    def fake(spec: str) -> foundation.Checkpoint:
        return foundation.Checkpoint(spec, ())

    label = "slots-0123456789ab"
    monkeypatch.setenv(foundation.FINETUNED_ENV, label)
    monkeypatch.setattr(foundation, "resolve_checkpoint", fake)
    monkeypatch.setattr(groupings, "resolve_checkpoint", fake)
    assert foundation.finetuned_spec() == label
    for model in ("fm_chronos_daily_ragged_ft", "fm_chronos_weekly_ragged_ft"):
        assert get_model(model).checkpoint.hash == label
        for grouping in groupings.FITTED_GROUPINGS:
            built = groupings.grouping_model(model, grouping, label)
            assert built.checkpoint.hash == label
            assert built.sample.keywords["checkpoint"].hash == label
            assert built.sample.keywords["group_slots"] is (
                groupings.cased_group_slots
            )


def test_a_place_with_lags_but_no_cases_in_its_counts_has_a_slot() -> None:
    # Cases only in the newest, partly reported weeks show in the lag
    # series alone; such a place keeps a slot.
    from bdbvhubrenewal.evaluation.groupings import cased_group_slots

    group = _block([True, False, False], ["p0", "p1"])
    ids = cased_group_slots(group, 2)
    assert ids[0] >= 0
    assert ids[1] >= 0
    assert ids[2] == -1


def _rel(label: str, init: str, n: int) -> dict:
    return {"label": label, "init": init, "n_outbreaks": n}


def test_the_newest_random_init_checkpoint_is_run_not_a_newer_zero_init() -> (
    None
):
    from bdbvhubrenewal.evaluation.groupings import choose_checkpoint

    releases = [
        _rel("slots-aaaaaaaaaaaa", "normal", 250),
        _rel("slots-bbbbbbbbbbbb", "zero", 500),
        _rel("slots-cccccccccccc", "normal", 1000),
    ]
    chosen, notes = choose_checkpoint("slots-aaaaaaaaaaaa", releases, set())
    assert chosen == "slots-cccccccccccc"
    text = " ".join(notes)
    assert "slots-aaaaaaaaaaaa" in text
    assert "zero-init" in text
    # A zero-init release with the most outbreaks is still not taken.
    releases.append(_rel("slots-dddddddddddd", "zero", 8000))
    assert choose_checkpoint("slots-dddddddddddd", releases, set())[0] == (
        "slots-cccccccccccc"
    )


def test_only_zero_init_checkpoints_run_nothing() -> None:
    from bdbvhubrenewal.evaluation.groupings import choose_checkpoint

    releases = [
        _rel("slots-aaaaaaaaaaaa", "zero", 250),
        _rel("slots-bbbbbbbbbbbb", "zero", 500),
    ]
    chosen, notes = choose_checkpoint("slots-bbbbbbbbbbbb", releases, set())
    assert chosen is None
    assert notes == ["no slots-run checkpoint yet (zero-init ones skipped)"]


def test_a_done_or_older_checkpoint_runs_nothing() -> None:
    from bdbvhubrenewal.evaluation.groupings import choose_checkpoint

    releases = [
        _rel("slots-aaaaaaaaaaaa", "normal", 250),
        _rel("slots-bbbbbbbbbbbb", "normal", 500),
    ]
    done = {"slots-bbbbbbbbbbbb"}
    assert choose_checkpoint("slots-bbbbbbbbbbbb", releases, done)[0] is None
    assert choose_checkpoint("slots-aaaaaaaaaaaa", releases, done)[0] is None
    # An old label exits at once even with a newer one waiting; the run
    # that finishes asks for the next one (no label) and takes it.
    releases.append(_rel("slots-cccccccccccc", "normal", 1000))
    assert choose_checkpoint("slots-aaaaaaaaaaaa", releases, done)[0] is None
    assert choose_checkpoint(None, releases, done)[0] == "slots-cccccccccccc"
    assert (
        choose_checkpoint(None, releases, done | {"slots-cccccccccccc"})[0]
        is None
    )
