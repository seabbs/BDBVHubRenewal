from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    check_distinct,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import (
    BASELINE,
    complete_weeks,
    release_backtest,
    summarise,
)
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.chain import MissingSnapshots


@pytest.fixture(scope="module")
def result(target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame):  # type: ignore[no-untyped-def]
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real,
        delays.assign(kind="real"),
        BacktestConfig(n_samples=50, seed=2),
        daily=daily,
    )
    return release_backtest(tasks, daily, target)


def test_origins_without_a_delay_pair_are_listed_with_no_forecasts(
    result,  # type: ignore[no-untyped-def]
) -> None:
    o = result.origins
    early = o.loc[o["origin"].isin(["2026-08-18", "2026-10-09"])]
    assert set(early["status"]) == {"no forecast"}
    assert (early["n_next"] == 0).all()
    assert (early["n_complete"] == 0).all()
    ran = o.loc[(o["origin"] == "2026-10-17") & (o["status"] == "ran")]
    assert set(ran["model"]) == {"randomwalk", "randomwalknowcast"}
    assert (ran["n_next"] > 0).all()


def test_scores_cover_both_targets_and_the_floor(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert set(s["target"]) == {"next", "complete"}
    assert set(s["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        BASELINE,
    }
    assert set(s["origin"]) == {"2026-10-17", "2026-10-24"}
    nxt = s.loc[s["target"] == "next"]
    assert set(nxt["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Every model is scored on the same forecasts.
    n = summarise(s)
    for _, g in n.groupby("target"):
        assert g["n"].nunique() == 1


def test_complete_target_only_has_complete_weeks(
    result,  # type: ignore[no-untyped-def]
    daily: pd.DataFrame,
) -> None:
    s = result.scores.loc[result.scores["target"] == "complete"]
    done = complete_weeks(daily, "2026-11-01", sorted(set(s["location"])))
    ok = set(
        zip(
            done.loc[done["complete"], "location"],
            done.loc[done["complete"], "target_end_date"],
            strict=True,
        )
    )
    assert set(zip(s["location"], s["target_end_date"], strict=True)) <= ok
    assert s["complete_at_latest"].all()


def test_next_target_uses_the_chain_over_the_gap(
    result,  # type: ignore[no-untyped-def]
) -> None:
    o = result.origins
    row = o.loc[
        (o["origin"] == "2026-10-17") & (o["model"] == "randomwalknowcast")
    ]
    assert row["mismatch_2026-10-24"].iloc[0] == 1


def test_quantiles_carry_the_reported_count_and_truth(
    result,  # type: ignore[no-untyped-def]
) -> None:
    q = result.quantiles
    assert {"q5", "q25", "q50", "q75", "q95", "reported", "truth"} <= set(q)
    assert (q["q5"] <= q["q95"]).all()


def test_missing_snapshots_stop_the_backtest(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(real, delays.assign(kind="real"), BacktestConfig())
    with pytest.raises(MissingSnapshots):
        release_backtest(tasks, daily, target)
    with pytest.raises(MissingSnapshots):
        run_forecasts([get_model("randomwalk")], tasks)


def test_identical_samples_from_two_models_stop_the_run() -> None:
    base = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-10-17",
            "location": "CD",
            "target_end_date": "2026-10-10",
            "sample": np.arange(1, 4),
            "value": [1.0, 2.0, 3.0],
        }
    )
    a, b = base.assign(model="a"), base.assign(model="b")
    with pytest.raises(ValueError, match="identical"):
        check_distinct(pd.concat([a, b]))
    check_distinct(
        pd.concat([a, dataclasses.replace(b) if False else b.assign(value=4.0)])
    )
