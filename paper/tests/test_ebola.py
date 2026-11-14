"""Tests of the Ebola hub loader and scoring."""

import gzip
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from paperdata import ebola


def test_hub_samples_keep_weekly_target_only(hub):
    s = ebola.hub_samples(hub)
    assert set(s["model"]) == {"team-a"}
    assert len(s) == 200
    assert s["origin_date"].iloc[0] == pd.Timestamp("2026-10-31")


def test_hub_samples_keep_latest_forecast_date(hub):
    path = hub / "model-output/team-a/2026-10-31-team-a.csv"
    first = pd.read_csv(path)
    later = first.assign(forecast_date="2026-10-28", value=first["value"] * 2)
    pd.concat([first, later]).to_csv(path, index=False)
    s = ebola.hub_samples(hub)
    assert len(s) == 200
    assert s["value"].max() == 200


def test_load_adds_candidates(hub, tmp_path):
    cand = tmp_path / "candidates.csv.gz"
    pd.DataFrame(
        {
            "model": "renewal",
            "kind": "real",
            "vintage": "2026-10-25",
            "reference_date": "2026-10-31",
            "location": "CD",
            "target_end_date": "2026-11-07",
            "sample": np.arange(1, 101),
            "value": 50.0,
            "as_of": None,
        }
    ).to_csv(cand, index=False)
    table = ebola.load(hub, candidates=[cand])
    assert set(table["model"]) == {"team-a", "renewal"}


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _fit(repo, model, origin, key, fitted_at, value):
    rows = [
        [model, "real", origin, "2026-10-31", "CD", "2026-11-07", s, value]
        for s in range(1, 101)
    ]
    fit = {
        "meta": {
            "model": model,
            "origin": origin,
            "key": key,
            "fitted_at": fitted_at,
            "status": "ok",
        },
        "forecasts": {
            "columns": [
                "model",
                "kind",
                "vintage",
                "reference_date",
                "location",
                "target_end_date",
                "sample",
                "value",
            ],
            "data": rows,
        },
    }
    path = repo / model / origin / f"{key}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(fit).encode()))


@pytest.fixture
def store(tmp_path):
    """Build a fit store with a prospective fit later refitted."""
    repo = tmp_path / "store"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "fit-store")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _fit(repo, "renewal", "2026-10-25", "a", "2026-10-26T10:00:00+00:00", 5)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    (repo / "renewal/2026-10-25/a.json.gz").unlink()
    _fit(repo, "renewal", "2026-10-25", "b", "2026-11-05T10:00:00+00:00", 9)
    # A fit made after its round closed is never prospective.
    _fit(
        repo, "fm_chronos_x", "2026-10-25", "c", "2026-11-12T10:00:00+00:00", 1
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "refit")
    return repo


def test_store_keeps_first_prospective_fit(store):
    s = ebola.store_samples(store, "fit-store")
    assert set(s["model"]) == {"renewal"}
    assert (s["value"] == 5).all()


def test_round_close_is_tuesday_before():
    close = ebola.round_close(pd.Timestamp("2026-11-14"))
    assert close == pd.Timestamp("2026-11-10 22:59", tz="UTC")


def test_store_skips_fits_with_no_forecasts(store):
    _fit(store, "pp", "2026-10-25", "d", "2026-10-26T09:00:00+00:00", 3)
    _git(store, "add", "-A")
    _git(store, "commit", "-q", "-m", "pp")
    _fit(store, "pp", "2026-10-25", "e", "2026-10-26T08:00:00+00:00", 0)
    path = store / "pp/2026-10-25/e.json.gz"
    fit = json.loads(gzip.decompress(path.read_bytes()))
    fit["forecasts"]["data"] = []
    path.write_bytes(gzip.compress(json.dumps(fit).encode()))
    _git(store, "add", "-A")
    _git(store, "commit", "-q", "-m", "empty")
    s = ebola.store_samples(store, "fit-store")
    assert set(s["model"]) == {"renewal", "pp"}
    assert (s.loc[s["model"] == "pp", "value"] == 3).all()


def test_hub_samples_keep_trajectory_ids(hub):
    s = ebola.hub_samples(hub)
    assert sorted(s["sample_id"].unique()) == list(range(1, 101))


def test_load_gives_sample_schema(hub):
    table = ebola.load(hub)
    assert list(table.columns) == list(ebola.COLUMNS)
    assert "sample_id" in table
    assert table["quantile_level"].isna().all()
    assert set(table["study"]) == {"ebola"}
    assert set(table["horizon"]) == {-1, 1}
    assert len(table) == 200
    assert table["truth"].isna().all()


def test_hub_baseline_is_the_hub_model():
    assert ebola.HUB_BASELINE == "BDBVHub-baseline"


def test_reporting_delay_is_first_week_age_reaching_level(hub):
    # Against the 2026-11-08 release, the 2026-10-24 release has week
    # 2026-10-17 (age 1 week) at 3 of 9. The 2026-11-01 release has it at
    # 8 of 9 (age 3 weeks, rounded up) and week 2026-10-24 at 10 of 30
    # (age 2 weeks, rounded up).
    assert ebola.reporting_delay(hub, level=0.8) == 21
    assert ebola.reporting_delay(hub, level=0.95) is None


def test_reporting_delay_skips_releases(hub):
    assert ebola.reporting_delay(hub, level=0.3, skip=("2026-11-01",)) == 7
    assert ebola.reporting_delay(hub, level=0.8, skip=("2026-11-01",)) is None


def test_reporting_delay_ignores_weeks_the_latest_release_lacks(hub):
    path = hub / "target-data/time-series.csv"
    data = pd.read_csv(path)
    gone = data.iloc[[0]].assign(
        as_of="2026-10-24", target_end_date="2026-10-10", observation=1000
    )
    pd.concat([data, gone]).to_csv(path, index=False)
    assert ebola.reporting_delay(hub, level=0.8) == 21
