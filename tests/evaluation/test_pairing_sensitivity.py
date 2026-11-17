"""Sensitivity of the walk models' scores to pairing."""

from __future__ import annotations

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.pairing import (
    VARIANTS,
    summarise_pairing,
    variant_forecasts,
    variant_scores,
)
from bdbvhubrenewal.forecasters import NATIONAL


@pytest.fixture(scope="module")
def setup(target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame):
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"] == sorted(real["vintage"].unique())[-3]]
    config = BacktestConfig(seed=3, n_samples=300)
    task = make_tasks(rows, delays, config, history=real, daily=daily)[0]
    truth = rolling_truth(real, 7)
    return task, truth


@pytest.fixture(scope="module")
def scores(setup) -> pd.DataFrame:
    task, truth = setup
    drawn = variant_forecasts("randomwalknowcast", task)
    return pd.concat(
        [
            variant_scores(
                s,
                truth,
                study="bdbv",
                model="randomwalknowcast",
                variant=label,
                origin=task.vintage,
            )
            for label, s in drawn.items()
        ],
        ignore_index=True,
    )


def test_one_forecast_gives_every_variant_with_the_country(setup) -> None:
    task, _ = setup
    drawn = variant_forecasts("randomwalknowcast", task)
    assert list(drawn) == [label for label, _ in VARIANTS]
    for samples in drawn.values():
        assert NATIONAL in set(samples["location"])


def test_per_cell_scores_do_not_move_and_sums_do(scores) -> None:
    cell = scores.loc[
        (scores["target"] == "cell") & (scores["level"] == "provinces")
    ].pivot_table(index="metric", columns="variant", values="value")
    assert cell.nunique(axis=1).max() == 1
    national = scores.loc[
        (scores["level"] == "national") & (scores["metric"] == "crps")
        & (scores["target"] == "cell")
    ].set_index("variant")["value"]  # fmt: skip
    # The country is the provinces' sum, so its score moves with pairing.
    assert national["off"] != national["prior"]


def test_summary_is_relative_to_no_pairing(scores) -> None:
    out = summarise_pairing(scores)
    off = out.loc[out["variant"] == "off", "relative"]
    # Undefined only where no pairing scores 0 (0 / 0).
    assert (off.dropna() == 1.0).all()
    assert off.notna().any()
    assert set(out["variant"]) == {label for label, _ in VARIANTS}
