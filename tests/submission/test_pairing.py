"""Pairing a submission's draws from the model's past forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.submission.pairing import (
    archived_forecasts,
    pair_submission,
    past_forecasts,
    stored_forecasts,
)


def _hub(origin: str, value: float, model: str = "m") -> pd.DataFrame:
    """A hub table of two places, a zone and two weeks, as archived."""
    rows = []
    for loc in (NATIONAL, "CD-IT", "CD-IT-01"):
        for week in ("2026-10-10", "2026-10-17"):
            for sample in (1, 2):
                rows.append(
                    {
                        "model": model,
                        "vintage": origin,
                        "reference_date": "2026-10-24",
                        "location": loc,
                        "target_end_date": week,
                        "output_type": "sample",
                        "output_type_id": sample,
                        "value": value + sample,
                    }
                )
    return pd.DataFrame(rows)


def _archive(root: Path, ref: str, run: str, table: pd.DataFrame) -> None:
    path = root / ref / run
    path.mkdir(parents=True)
    table.to_csv(path / "candidates.csv.gz", index=False)


def test_archive_gives_each_origins_latest_run(tmp_path: Path) -> None:
    _archive(tmp_path, "2026-10-17", "100", _hub("2026-10-17", 1.0))
    _archive(tmp_path, "2026-10-24", "200", _hub("2026-10-24", 5.0))
    _archive(tmp_path, "2026-10-24", "300", _hub("2026-10-24", 9.0))
    _archive(
        tmp_path, "2026-10-24", "301", _hub("2026-10-24", 0.0, model="other")
    )
    out = archived_forecasts(tmp_path, "m")
    assert set(out.columns) >= {
        "model",
        "origin",
        "location",
        "target_end_date",
        "sample",
        "value",
    }
    assert set(out["origin"]) == {"2026-10-17", "2026-10-24"}
    # Zones are left out, and a round run twice gives its latest run.
    assert set(out["location"]) == {NATIONAL, "CD-IT"}
    late = out.loc[out["origin"] == "2026-10-24", "value"]
    assert set(late) == {10.0, 11.0}


def test_no_archive_gives_no_forecasts(tmp_path: Path) -> None:
    assert archived_forecasts(tmp_path / "none", "m").empty
    assert archived_forecasts(None, "m").empty


def _stored(root: Path, origin: str, fitted_at: str, value: float) -> None:
    forecasts = pd.DataFrame(
        {
            "location": ["CD-IT", "CD-IT"],
            "target_end_date": ["2026-10-10", "2026-10-10"],
            "sample": [1, 2],
            "value": [value, value + 1],
            "as_of": [np.nan, np.nan],
        }
    )
    meta = {
        "model": "m",
        "origin": origin,
        "key": f"k{fitted_at}",
        "fitted_at": fitted_at,
        "status": "ok",
    }
    empty = pd.DataFrame()
    store.write_fit(root, store.StoredFit(meta, forecasts, empty, empty))


def test_store_gives_each_origins_latest_good_fit(tmp_path: Path) -> None:
    _stored(tmp_path, "2026-10-17", "2026-10-18T00:00:00", 1.0)
    _stored(tmp_path, "2026-10-17", "2026-10-20T00:00:00", 7.0)
    meta = {"model": "m", "origin": "2026-10-24", "key": "bad"}
    store.write_fit(tmp_path, store.failed(meta, RuntimeError("no")))
    out = stored_forecasts(tmp_path, "m")
    assert set(out["origin"]) == {"2026-10-17"}
    assert set(out["value"]) == {7.0, 8.0}


def test_past_forecasts_take_the_archive_then_the_store(
    tmp_path: Path,
) -> None:
    archive, root = tmp_path / "archive", tmp_path / "store"
    _archive(archive, "2026-10-24", "1", _hub("2026-10-24", 5.0))
    _stored(root, "2026-10-17", "2026-10-18T00:00:00", 1.0)
    _stored(root, "2026-10-24", "2026-10-25T00:00:00", 99.0)
    out = past_forecasts("m", "2026-10-31", archive=archive, store=root)
    assert set(out["origin"]) == {"2026-10-17", "2026-10-24"}
    # The archive's forecast, as submitted, wins over the store's.
    assert 99.0 not in set(out["value"])
    # Only origins before the forecast's.
    early = past_forecasts("m", "2026-10-24", archive=archive, store=root)
    assert set(early["origin"]) == {"2026-10-17"}


@pytest.fixture(scope="module")
def tasks(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> list:
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        real, delays, BacktestConfig(n_samples=20, seed=3), daily=daily
    )


def _cells(rows: pd.DataFrame) -> pd.Series:
    return rows.groupby(["location", "target_end_date"])["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )


def test_randomwalknowcast_is_recomputed_when_nothing_is_kept(
    tasks: list,
) -> None:
    task = tasks[-1]
    model = get_model("randomwalknowcast")
    samples = model.forecast(task)
    paired, fit = pair_submission(
        "randomwalknowcast", samples, task, truth_delay=7, past_tasks=tasks
    )
    # Each place and week keeps its draws.
    pd.testing.assert_series_equal(_cells(paired), _cells(samples))
    assert fit["origin"] == task.vintage
    assert fit["n_scores"] > 0
    assert fit["source"] == "recomputed"


def test_a_model_with_no_past_forecasts_takes_the_prior(tasks: list) -> None:
    task = tasks[-1]
    samples = get_model("randomwalknowcast").forecast(task)
    paired, fit = pair_submission(
        "fm_chronos_weekly_ragged", samples, task, truth_delay=7
    )
    pd.testing.assert_series_equal(_cells(paired), _cells(samples))
    assert fit["n_scores"] == 0
    assert fit["source"] == "none"
