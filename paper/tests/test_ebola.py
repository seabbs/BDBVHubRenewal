"""Tests of the Ebola hub loader and scoring."""

import gzip
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from paperdata import ebola

TARGET = "wk inc confirmed cases"


def _hub_file(path, ref, location, weeks, values, forecast_date):
    """Write one hub sample file with `values[week][sample]`."""
    rows = [
        {
            "reference_date": ref,
            "forecast_date": forecast_date,
            "target": TARGET,
            "target_end_date": week,
            "location": location,
            "output_type": "sample",
            "output_type_id": s + 1,
            "value": v,
        }
        for week, draws in zip(weeks, values, strict=True)
        for s, v in enumerate(draws)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def hub(tmp_path):
    """Build a small hub with one weekly round and an old cumulative file."""
    root = tmp_path / "hub"
    draws = [list(range(1, 101))]
    _hub_file(
        root / "model-output/team-a/2026-10-31-team-a.csv",
        "2026-10-31",
        "CD",
        ["2026-11-07"],
        draws,
        "2026-10-27",
    )
    old = root / "model-output/team-a/2026-06-13-team-a.csv"
    pd.DataFrame(
        {
            "reference_date": ["2026-06-13"],
            "target": ["cumulative cases"],
            "location": ["CD"],
            "output_type": ["quantile"],
            "output_type_id": [0.5],
            "value": [100.0],
        }
    ).to_csv(old, index=False)
    (root / "target-data").mkdir(parents=True)
    pd.DataFrame(
        {
            "as_of": ["2026-11-01", "2026-11-08"],
            "target_end_date": ["2026-11-07", "2026-11-07"],
            "target": [TARGET, TARGET],
            "location": ["CD", "CD"],
            "location_name": ["DRC", "DRC"],
            "location_type": ["country", "country"],
            "observation": [10, 50],
        }
    ).to_csv(root / "target-data/time-series.csv", index=False)
    return root


def test_quantiles_from_samples():
    samples = pd.DataFrame(
        {
            "model": "m",
            "location": "CD",
            "origin_date": pd.Timestamp("2026-10-31"),
            "target_date": pd.Timestamp("2026-11-07"),
            "sample": np.arange(1, 102),
            "value": np.arange(0.0, 101.0),
        }
    )
    q = ebola.to_quantiles(samples)
    assert list(q["quantile_level"]) == list(ebola.QUANTILE_LEVELS)
    assert q.loc[q["quantile_level"] == 0.5, "value"].item() == 50.0
    assert q.loc[q["quantile_level"] == 0.025, "value"].item() == 2.5


def test_hub_samples_keep_weekly_target_only(hub):
    s = ebola.hub_samples(hub)
    assert set(s["model"]) == {"team-a"}
    assert len(s) == 100
    assert s["origin_date"].iloc[0] == pd.Timestamp("2026-10-31")


def test_hub_samples_keep_latest_forecast_date(hub):
    path = hub / "model-output/team-a/2026-10-31-team-a.csv"
    first = pd.read_csv(path)
    later = first.assign(forecast_date="2026-10-28", value=first["value"] * 2)
    pd.concat([first, later]).to_csv(path, index=False)
    s = ebola.hub_samples(hub)
    assert len(s) == 100
    assert s["value"].max() == 200


def test_truth_is_latest_vintage(hub):
    truth = ebola.hub_truth(hub)
    assert truth["truth"].item() == 50
    assert truth["as_of"].item() == pd.Timestamp("2026-11-08")


def test_load_gives_schema(hub):
    table = ebola.load(hub)
    assert list(table.columns) == list(ebola.COLUMNS)
    assert set(table["study"]) == {"ebola"}
    assert set(table["horizon"]) == {1}
    assert (table["truth"] == 50).all()
    assert len(table) == len(ebola.QUANTILE_LEVELS)


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


def test_wis_of_point_mass_at_truth_is_zero():
    q = pd.DataFrame(
        {
            "model": "m",
            "location": "CD",
            "origin_date": pd.Timestamp("2026-10-31"),
            "target_date": pd.Timestamp("2026-11-07"),
            "horizon": 1,
            "quantile_level": ebola.QUANTILE_LEVELS,
            "value": 7.0,
            "truth": 7.0,
        }
    )
    scores = ebola.score(q)
    assert scores["wis"].item() == 0
    assert scores["ae_median"].item() == 0
    assert scores["cover_50"].item()


def test_wis_matches_interval_definition():
    levels = np.array(ebola.QUANTILE_LEVELS)
    values = np.array([0.0, 1, 2, 3, 4, 5, 6])
    y = 10.0
    q = pd.DataFrame(
        {
            "model": "m",
            "location": "CD",
            "origin_date": pd.Timestamp("2026-10-31"),
            "target_date": pd.Timestamp("2026-11-07"),
            "horizon": 1,
            "quantile_level": levels,
            "value": values,
            "truth": y,
        }
    )
    # Intervals (alpha): 0.05 [0, 6], 0.2 [1, 5], 0.5 [2, 4].
    terms = [
        a / 2 * ((u - lo) + 2 / a * max(y - u, 0))
        for a, lo, u in [(0.05, 0, 6), (0.2, 1, 5), (0.5, 2, 4)]
    ]
    expected = (0.5 * abs(y - 3) + sum(terms)) / 3.5
    assert ebola.score(q)["wis"].item() == pytest.approx(expected)


def test_score_drops_rows_without_truth():
    q = pd.DataFrame(
        {
            "model": "m",
            "location": "CD",
            "origin_date": pd.Timestamp("2026-10-31"),
            "target_date": pd.Timestamp("2026-11-21"),
            "horizon": 3,
            "quantile_level": ebola.QUANTILE_LEVELS,
            "value": 1.0,
            "truth": np.nan,
        }
    )
    assert ebola.score(q).empty


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


def test_hub_baseline_is_the_hub_model():
    assert ebola.HUB_BASELINE == "BDBVHub-baseline"


def test_relative_wis_uses_shared_forecasts():
    def rows(model, location, wis):
        return {
            "model": model,
            "location": location,
            "origin_date": pd.Timestamp("2026-11-07"),
            "target_date": pd.Timestamp("2026-11-07"),
            "horizon": 0,
            "wis": wis,
        }

    scores = pd.DataFrame(
        [
            rows(ebola.HUB_BASELINE, "CD", 4.0),
            rows(ebola.HUB_BASELINE, "CD-IT", 100.0),
            rows("m", "CD", 2.0),
        ]
    )
    rel = ebola.relative_wis(scores).set_index("model")
    assert rel.loc["m", "relative_wis"] == 0.5
    assert rel.loc["m", "n"] == 1
    assert rel.loc[ebola.HUB_BASELINE, "relative_wis"] == 1
