"""Tests of the synthetic study's loader of the runner's outputs."""

import pandas as pd
import pytest

from paperdata import synthetic
from paperpipe.config import seed_of
from paperpipe.studies import Sources, load_data, run_forecasts, source_of

LOCATIONS = ["9501-0001-1_CD", "9501-0001-1_CD-01", "9502-0003-1_CD"]


@pytest.fixture
def runner(tmp_path):
    """The runner's three outputs for two seeds."""
    out = tmp_path / "synthetic-dev.csv.gz"
    pd.DataFrame(
        {
            "study": "synthetic",
            "model": "m",
            "location": LOCATIONS,
            "origin_date": "2026-01-10",
            "target_date": "2026-01-10",
            "horizon": 0,
            "quantile_level": None,
            "sample_id": 0,
            "value": 1.0,
            "truth": 1.0,
        }
    ).to_csv(out, index=False)
    pd.DataFrame(
        {
            "study": "synthetic",
            "location": LOCATIONS,
            "target_date": "2026-01-03",
            "as_of_date": "2026-01-10",
            "value": [4.0, 2.0, 7.0],
        }
    ).to_csv(tmp_path / "synthetic-dev-observations.csv", index=False)
    pd.DataFrame(
        {
            "location": LOCATIONS,
            "process": ["growth", "growth", "renewal_7"],
            "delay_q98": [12.0, 12.0, 30.0],
        }
    ).to_csv(tmp_path / "synthetic-dev-factors.csv", index=False)
    return out


def test_runner_locations_carry_their_seed() -> None:
    assert seed_of("9501-0001-1_CD-01") == 9501
    assert seed_of("s7001-sim01") == 7001


def test_load_keeps_the_seeds_and_takes_d_from_the_98pct_delay(runner):
    obs, locations = synthetic.load(runner, [9501])
    assert set(obs["location"]) == set(LOCATIONS[:2])
    assert locations["delay_days"].tolist() == [12, 12]
    assert set(locations["process"]) == {"growth"}


def test_missing_names_absent_outputs(runner, tmp_path):
    assert synthetic.missing(runner) is None
    (tmp_path / "synthetic-dev-factors.csv").unlink()
    assert "factors" in synthetic.missing(runner)


def test_the_synthetic_study_reads_the_runner(runner):
    sources = Sources(synthetic_forecasts=runner)
    assert source_of("synthetic", sources) == "real"
    obs, _ = load_data("synthetic", seeds=(9502,), sources=sources)
    assert obs["location"].tolist() == [LOCATIONS[2]]
    fc = run_forecasts(
        "synthetic", seeds=(9502,), sources=sources, source="real"
    )
    # Only the run's seeds, with the truth left for the workflow.
    assert fc["location"].tolist() == [LOCATIONS[2]]
    assert fc["truth"].isna().all()
