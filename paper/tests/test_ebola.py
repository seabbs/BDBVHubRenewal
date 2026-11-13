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
    draws = [list(range(1, 101)), list(range(1, 101))]
    _hub_file(
        root / "model-output/team-a/2026-10-31-team-a.csv",
        "2026-10-31",
        "CD",
        ["2026-10-24", "2026-11-07"],
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
    releases = [
        ("2026-10-24", "2026-10-17", 3),
        ("2026-11-01", "2026-10-17", 8),
        ("2026-11-01", "2026-10-24", 10),
        ("2026-11-08", "2026-10-17", 9),
        ("2026-11-08", "2026-10-24", 30),
        ("2026-11-08", "2026-11-07", 50),
    ]
    pd.DataFrame(
        {
            "as_of": [r[0] for r in releases],
            "target_end_date": [r[1] for r in releases],
            "target": TARGET,
            "location": "CD",
            "location_name": "DRC",
            "location_type": "country",
            "observation": [r[2] for r in releases],
        }
    ).to_csv(root / "target-data/time-series.csv", index=False)
    return root


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


def test_rolling_truth_is_release_d_days_after_week(hub):
    truth = ebola.rolling_truth(hub, delay_days=14).set_index("target_date")
    # 2026-10-17 + 14 days is 2026-10-31: the 2026-10-24 release.
    assert truth.loc["2026-10-17", "truth"] == 3
    # 2026-10-24 + 14 days is 2026-11-07: the 2026-11-01 release.
    assert truth.loc["2026-10-24", "truth"] == 10
    # 2026-11-07 + 14 days is not released yet.
    assert pd.Timestamp("2026-11-07") not in truth.index


def test_rolling_truth_needs_a_release_by_d_days(hub):
    truth = ebola.rolling_truth(hub, delay_days=3)
    # 2026-10-17 + 3 is before the first release.
    assert pd.Timestamp("2026-10-17") not in set(truth["target_date"])


def test_load_gives_sample_schema(hub):
    table = ebola.load(hub, delay_days=14)
    assert list(table.columns) == list(ebola.COLUMNS)
    assert "sample_id" in table
    assert set(table["study"]) == {"ebola"}
    assert set(table["horizon"]) == {-1, 1}
    assert len(table) == 200
    week = table.set_index("target_date")["truth"]
    assert (week.loc["2026-10-24"] == 10).all()
    assert week.loc["2026-11-07"].isna().all()


def _forecast(model, location, values, truth, target="2026-10-24"):
    return pd.DataFrame(
        {
            "study": "ebola",
            "model": model,
            "location": location,
            "origin_date": pd.Timestamp("2026-10-31"),
            "target_date": pd.Timestamp(target),
            "horizon": -1,
            "sample_id": np.arange(1, len(values) + 1),
            "value": np.asarray(values, dtype=float),
            "truth": truth,
        }
    )


def test_score_uses_the_repo_sample_scores():
    from bdbvhubrenewal.evaluation.scoring import (
        bias_sample,
        coverage_sample,
        crps_sample,
    )

    x = np.arange(0.0, 20.0)
    scores = ebola.score(_forecast("m", "CD", x, 7.0))
    row = scores.iloc[0]
    assert row["crps"] == pytest.approx(crps_sample(7.0, x))
    assert row["log_crps"] == pytest.approx(
        crps_sample(np.log1p(7.0), np.log1p(x))
    )
    assert row["bias"] == pytest.approx(bias_sample(7.0, x))
    assert row["coverage_50"] == coverage_sample(7.0, x, 0.5)
    assert row["coverage_90"] == coverage_sample(7.0, x, 0.9)
    assert row["n_samples"] == 20


def test_score_drops_forecasts_without_truth():
    assert ebola.score(_forecast("m", "CD", [1.0, 2.0], np.nan)).empty


def test_hub_baseline_is_the_hub_model():
    assert ebola.HUB_BASELINE == "BDBVHub-baseline"


def test_relative_scores_use_forecasts_all_models_share():
    base = ebola.HUB_BASELINE
    table = pd.concat(
        [
            _forecast(base, "CD", [4.0, 4.0], 0.0),
            _forecast(base, "CD-IT", [100.0, 100.0], 0.0),
            _forecast("a", "CD", [2.0, 2.0], 0.0),
            _forecast("a", "CD-IT", [1.0, 1.0], 0.0),
            _forecast("b", "CD", [8.0, 8.0], 0.0),
        ]
    )
    rel = ebola.relative_scores(ebola.score(table)).set_index("model")
    # Only CD is shared by all three models.
    assert (rel["n"] == 1).all()
    assert rel.loc["a", "relative_crps"] == pytest.approx(0.5)
    assert rel.loc["b", "relative_crps"] == pytest.approx(2.0)
    assert rel.loc[base, "relative_crps"] == 1
    assert "relative_log_crps" in rel
