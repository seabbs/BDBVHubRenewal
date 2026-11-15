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
    # Zones and the country, the sum of the provinces in a hub table, are
    # left out, and a round run twice gives its latest run.
    assert set(out["location"]) == {"CD-IT"}
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


def test_past_forecasts_take_the_store_then_the_archive(
    tmp_path: Path,
) -> None:
    archive, root = tmp_path / "archive", tmp_path / "store"
    _archive(archive, "2026-10-17", "1", _hub("2026-10-17", 50.0))
    _archive(archive, "2026-10-24", "1", _hub("2026-10-24", 5.0))
    _stored(root, "2026-10-24", "2026-10-25T00:00:00", 99.0)
    out, _ = past_forecasts("m", "2026-10-31", archive=archive, store=root)
    sources = out.drop_duplicates("origin").set_index("origin")["source"]
    # The store's forecast, as the backtest scores it, wins, and the
    # archive fills an origin the store lacks.
    assert sources.to_dict() == {"2026-10-24": "store", "2026-10-17": "archive"}
    assert 5.0 + 1 not in set(out["value"])
    # Only origins before the forecast's.
    early, _ = past_forecasts("m", "2026-10-24", archive=archive, store=root)
    assert set(early["origin"]) == {"2026-10-17"}


def test_archived_national_rows_are_left_out(tmp_path: Path) -> None:
    _archive(tmp_path, "2026-10-17", "1", _hub("2026-10-17", 1.0))
    assert NATIONAL not in set(archived_forecasts(tmp_path, "m")["location"])


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
    # Each past origin says where its forecast came from.
    assert fit["sources"]
    assert set(fit["sources"].values()) == {"recomputed"}
    assert all(o < task.vintage for o in fit["sources"])


def test_a_model_with_no_past_forecasts_fails(tasks: list) -> None:
    task = tasks[-1]
    samples = get_model("randomwalknowcast").forecast(task)
    with pytest.raises(LookupError, match="no past forecasts"):
        pair_submission(
            "fm_chronos_weekly_ragged",
            samples,
            task,
            truth_delay=7,
            past_tasks=tasks,
        )


def test_the_first_origin_takes_the_prior(tasks: list) -> None:
    # With no earlier origin, the backtest has no errors either.
    task = tasks[-1]
    samples = get_model("randomwalknowcast").forecast(task)
    paired, fit = pair_submission(
        "fm_chronos_weekly_ragged",
        samples,
        task,
        truth_delay=7,
        past_tasks=[t for t in tasks if t.vintage >= task.vintage],
    )
    pd.testing.assert_series_equal(_cells(paired), _cells(samples))
    assert fit["n_scores"] == 0
    assert fit["source"] == "none"
    assert fit["sources"] == {}
    assert fit["rho_time"] == pytest.approx(0.9)


def test_submission_pairs_as_the_release_backtest_does(
    tasks: list, target: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.nowcast import (
        forecast_releases,
        rolling_truth,
    )
    from bdbvhubrenewal.forecasters.copula import pair_registered

    real = target.loc[target["kind"] == "real"]
    run = forecast_releases(tasks, target, ["randomwalknowcast"])
    origin = max(run.samples["origin"])
    task = next(t for t in tasks if t.vintage == origin)
    scored, fits = pair_registered(
        run.samples.astype({"origin": str}),
        rolling_truth(real, 7),
        task.seed,
    )
    submitted, record = pair_submission(
        "randomwalknowcast",
        get_model("randomwalknowcast").forecast(task),
        task,
        truth_delay=7,
        past_tasks=tasks,
    )
    fit = fits.loc[fits["origin"] == origin].iloc[0]
    assert record["n_scores"] == fit["n_scores"] > 0
    assert record["rho_time"] == pytest.approx(fit["rho_time"])
    assert record["rho_provinces"] == pytest.approx(fit["rho_provinces"])
    keys = ["location", "target_end_date", "sample"]
    mine = scored.loc[
        (scored["origin"] == origin) & (scored["location"] != NATIONAL)
    ]
    pd.testing.assert_frame_equal(
        mine[[*keys, "value"]].sort_values(keys).reset_index(drop=True),
        submitted[[*keys, "value"]].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )


def test_an_unfitted_model_with_no_forecast_at_earlier_origins_takes_the_prior(
    tasks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As in the backtest, where the model has no forecast there either.
    from bdbvhubrenewal.forecasters import ModelUnavailableError
    from bdbvhubrenewal.submission import pairing

    task = tasks[-1]
    samples = get_model("randomwalknowcast").forecast(task)

    class Unavailable:
        def forecast(self, _task):  # type: ignore[no-untyped-def]
            raise ModelUnavailableError("no delay pair")

    monkeypatch.setattr(pairing, "get_model", lambda _name: Unavailable())
    _, fit = pair_submission(
        "randomwalknowcast", samples, task, truth_delay=7, past_tasks=tasks
    )
    assert fit["n_scores"] == 0
    assert fit["source"] == "none"
    # Each earlier origin is listed as one the model cannot forecast.
    assert fit["unavailable"]
    assert all(o < task.vintage for o in fit["unavailable"])


def test_an_unfitted_model_whose_past_forecasts_are_missing_fails(
    tasks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Forecasting again gives nothing, though the model could forecast
    # there, so the submission cannot pair as the backtest does.
    from bdbvhubrenewal.submission import pairing

    task = tasks[-1]
    samples = get_model("randomwalknowcast").forecast(task)
    monkeypatch.setattr(
        pairing, "_recomputed", lambda *_: (pairing._empty(), set())
    )
    with pytest.raises(LookupError, match="no past forecasts"):
        pair_submission(
            "randomwalknowcast", samples, task, truth_delay=7, past_tasks=tasks
        )
