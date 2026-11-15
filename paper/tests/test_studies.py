import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from paperdata.ebola import HUB_BASELINE
from paperpipe.config import seed_of
from paperpipe.studies import (
    Sources,
    load_data,
    missing,
    run_forecasts,
    source_of,
    study_baseline,
)

SPLIT = pd.Timestamp("2022-02-10")
CONFIG = Path(__file__).parents[1] / "workflow" / "config.yaml"


def test_dummy_synthetic_data_is_one_study_for_the_first_seed() -> None:
    obs, locations = load_data("synthetic", seeds=(7001, 7002))
    assert {seed_of(loc) for loc in obs["location"]} == {7001}
    assert set(locations["location"]) == set(obs["location"])
    assert {"delay_days", "delay_q98", "process"} <= set(locations.columns)
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


def _data(path: Path) -> Path:
    """Main's data folder with a target table of three releases."""
    path.mkdir()
    pd.DataFrame(
        {
            "kind": "real",
            "vintage": ["2026-10-26", "2026-11-03", "2026-11-10"],
            "location": "CD",
            "location_name": "DRC",
            "target_end_date": "2026-10-24",
            "observed": 1,
            "imputed": 0,
            "total": [3, 10, 12],
            "cutoff": ["2026-10-24", "2026-11-01", "2026-11-08"],
        }
    ).to_csv(path / "target.csv", index=False)
    return path


def _backtest(path: Path) -> Path:
    """Release-backtest samples of one model from one release."""
    pd.DataFrame(
        {
            "model": "randomwalknowcast",
            "kind": ["real"] * 3 + ["simulated"],
            "vintage": "2026-11-01",
            "reference_date": "2026-11-07",
            "location": "CD",
            "target_end_date": "2026-10-24",
            "sample": [1, 2, 3, 1],
            "value": [4.0, 5.0, 6.0, 7.0],
            "as_of": None,
            "horizon": -2,
        }
    ).to_csv(path, index=False)
    return path


def test_ebola_reads_the_release_backtest_then_the_hub(
    hub_with_baseline, tmp_path
) -> None:
    sources = Sources(
        hub=hub_with_baseline,
        ebola_data=_data(tmp_path / "data"),
        ebola_backtest=_backtest(tmp_path / "q.csv"),
    )
    assert source_of("ebola", sources, delay_days=14) == "real"
    obs, locations = load_data("ebola", delay_days=14, sources=sources)
    assert len(obs) == 3
    assert locations["delay_days"].iloc[0] == 14
    # The hub's baseline can be scored, so it is the baseline and the
    # hub's forecasts join.
    assert study_baseline("ebola", sources, delay_days=14) is None
    fc = run_forecasts("ebola", sources=sources, source="real")
    assert {"randomwalknowcast", "team-a", HUB_BASELINE} <= set(fc["model"])
    # Samples throughout, so the study is scored by CRPS.
    assert fc["sample_id"].notna().all()
    assert fc["quantile_level"].isna().all()


def test_ebola_is_relative_to_randomwalknowcast_until_the_hub_can_be(
    hub, tmp_path
) -> None:
    sources = Sources(
        hub=hub,
        ebola_data=_data(tmp_path / "data"),
        ebola_backtest=_backtest(tmp_path / "q.csv"),
    )
    assert (
        study_baseline("ebola", sources, delay_days=14) == "randomwalknowcast"
    )
    fc = run_forecasts(
        "ebola", sources=sources, source="real", baseline="randomwalknowcast"
    )
    assert set(fc["model"]) == {"randomwalknowcast"}
    # Only real releases' samples.
    assert len(fc) == 3
    assert fc["truth"].isna().all()


def test_ebola_without_a_backtest_or_hub_is_dummy(hub) -> None:
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


def _paper_table(path: Path, last: str = "2022-02-09") -> Path:
    origins = pd.date_range("2022-02-01", last)
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


def test_germany_fails_on_tables_with_heldout_dates(tmp_path) -> None:
    sources = Sources(
        germany_forecasts=(_paper_table(tmp_path / "p.csv", "2022-02-20"),),
        germany_triangle=_triangle(tmp_path / "triangle.csv"),
    )
    with pytest.raises(ValueError, match="must not read"):
        run_forecasts("germany", before=SPLIT, sources=sources, source="real")


def test_germany_needs_its_truth_delay() -> None:
    with pytest.raises(ValueError, match="delay_days"):
        load_data("germany")


def test_missing_names_why_a_study_cannot_use_real_data() -> None:
    assert "runner" in missing("synthetic", Sources())
    assert "release data" in missing("ebola", Sources())
    assert "German" in missing("germany", Sources())


def test_ebola_without_a_backtest_names_it(tmp_path) -> None:
    sources = Sources(ebola_data=_data(tmp_path / "data"))
    assert "backtest" in missing("ebola", sources, delay_days=14)


def test_a_paper_run_fails_on_dummy_data(tmp_path) -> None:
    from paperpipe.__main__ import main

    cfg = yaml.safe_load(CONFIG.read_text())
    out = tmp_path / "x"
    args = ["data", "synthetic", "--seeds", "9001", "--mode", "paper"]
    args += ["--config", json.dumps(cfg), "--meta", str(out / "m.json")]
    args += [
        "--observations",
        str(out / "o.csv"),
        "--locations",
        str(out / "l.csv"),
    ]
    with pytest.raises(ValueError, match="synthetic"):
        main(args)


def test_a_development_run_may_use_labelled_dummy_data(tmp_path) -> None:
    from paperpipe.__main__ import main
    from paperpipe.meta import read_meta

    cfg = yaml.safe_load(CONFIG.read_text())
    out = tmp_path / "x"
    args = ["data", "synthetic", "--seeds", "7001", "--mode", "development"]
    args += ["--config", json.dumps(cfg), "--meta", str(out / "m.json")]
    args += [
        "--observations",
        str(out / "o.csv"),
        "--locations",
        str(out / "l.csv"),
    ]
    main(args)
    assert read_meta(out / "m.json")["source"] == "dummy"
