from pathlib import Path

import pandas as pd
import pytest

from paperdata.ebola import HUB_BASELINE
from paperpipe.config import seed_of
from paperpipe.studies import Sources, load_data, run_forecasts, source_of

SPLIT = pd.Timestamp("2022-02-10")


def test_synthetic_data_has_one_set_of_outbreaks_per_seed() -> None:
    obs, locations = load_data("synthetic", seeds=(7001, 7002))
    assert {seed_of(loc) for loc in obs["location"]} == {7001, 7002}
    assert set(locations["location"]) == set(obs["location"])
    assert {"delay_days", "delay_q98", "gt_mean"} <= set(locations.columns)
    assert (locations["delay_days"] == locations["delay_q98"]).all()


def test_synthetic_forecasts_match_the_data() -> None:
    obs, _ = load_data("synthetic", seeds=(7001,))
    fc = run_forecasts("synthetic", seeds=(7001,), n_samples=10)
    assert set(fc["location"]) == set(obs["location"])
    assert fc["sample_id"].nunique() == 10


def test_germany_before_drops_later_dates() -> None:
    obs, locations = load_data("germany", before=SPLIT, delay_days=40)
    fc = run_forecasts("germany", before=SPLIT)
    assert fc["origin_date"].max() < SPLIT
    assert obs["target_date"].max() < SPLIT
    assert (locations["delay_days"] == 40).all()


def test_germany_without_before_keeps_both_halves() -> None:
    fc = run_forecasts("germany")
    assert fc["origin_date"].min() < SPLIT <= fc["origin_date"].max()


def test_ebola_delay_is_set_per_location() -> None:
    obs, locations = load_data("ebola")
    assert set(locations["location"]) == set(obs["location"])
    assert (locations["delay_days"] > 0).all()


@pytest.fixture
def hub_with_baseline(hub):
    team = hub / "model-output" / "team-a" / "2026-10-31-team-a.csv"
    base = hub / "model-output" / HUB_BASELINE
    base.mkdir()
    (base / f"2026-10-31-{HUB_BASELINE}.csv").write_text(team.read_text())
    return hub


def test_ebola_reads_the_hub_once_its_baseline_is_scorable(
    hub_with_baseline,
) -> None:
    sources = Sources(hub=hub_with_baseline)
    assert source_of("ebola", sources, delay_days=14) == "real"
    obs, locations = load_data("ebola", delay_days=14, sources=sources)
    assert set(obs["study"]) == {"ebola"}
    assert len(obs) == 6
    assert list(locations["location"]) == ["CD"]
    assert locations["delay_days"].iloc[0] == 14
    fc = run_forecasts("ebola", sources=sources, source="real")
    assert set(fc["model"]) == {"team-a", HUB_BASELINE}
    assert fc["truth"].isna().all()


def test_ebola_stays_dummy_until_its_baseline_is_scorable(hub) -> None:
    assert source_of("ebola", Sources(hub=hub), delay_days=14) == "dummy"
    assert source_of("ebola", Sources()) == "dummy"


def test_synthetic_runs_on_dummy_data() -> None:
    assert source_of("synthetic", Sources()) == "dummy"


def _triangle(path: Path) -> Path:
    dates = pd.date_range("2021-11-01", "2022-03-31").strftime("%Y-%m-%d")
    rows = [
        {"date": d, "location": loc, "age_group": age}
        | {f"value_{k}d": 1 for k in range(4)}
        for d in dates
        for loc, age in [("DE", "00+"), ("DE", "80+"), ("DE-BY", "00+")]
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _paper_table(path: Path) -> Path:
    origins = pd.date_range("2022-02-01", "2022-02-20")
    pd.DataFrame(
        {
            "study": "germany",
            "model": "m",
            "location": "DE",
            "stratum": "all",
            "origin_date": origins,
            "target_date": origins,
            "horizon": 0,
            "quantile_level": 0.5,
            "sample_id": None,
            "value": 1.0,
            "truth": 7.0,
        }
    ).to_csv(path, index=False)
    return path


def test_germany_reads_the_benchmark(tmp_path) -> None:
    sources = Sources(
        germany_forecasts=(_paper_table(tmp_path / "paper.csv"),),
        germany_triangle=_triangle(tmp_path / "triangle.csv"),
    )
    assert source_of("germany", sources) == "real"
    obs, locations = load_data(
        "germany", before=SPLIT, delay_days=10, sources=sources
    )
    assert obs["target_date"].max() < SPLIT
    # Releases run to D days after the split, for the last truths.
    assert obs["as_of_date"].max() == SPLIT + pd.Timedelta(days=9)
    assert set(obs["stratum"]) == {"all", "80+"}
    assert set(locations["location"]) == {"DE", "DE-BY"}
    fc = run_forecasts("germany", before=SPLIT, sources=sources, source="real")
    assert fc["origin_date"].max() < SPLIT
    assert fc["truth"].isna().all()


def test_germany_needs_its_truth_delay() -> None:
    with pytest.raises(ValueError, match="delay_days"):
        load_data("germany")
