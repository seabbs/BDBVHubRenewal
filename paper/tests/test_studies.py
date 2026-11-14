import pandas as pd

from paperpipe.config import seed_of
from paperpipe.studies import DUMMY_STUDIES, load_data, run_forecasts

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


def test_ebola_reads_the_hub_when_given_one(hub) -> None:
    obs, locations = load_data("ebola", hub=hub)
    assert set(obs["study"]) == {"ebola"}
    assert len(obs) == 6
    assert set(obs["as_of_date"].astype(str)) == {
        "2026-10-24",
        "2026-11-01",
        "2026-11-08",
    }
    assert list(locations["location"]) == ["CD"]
    assert locations["delay_days"].iloc[0] > 0


def test_ebola_forecasts_come_from_the_hub_when_given_one(hub) -> None:
    fc = run_forecasts("ebola", hub=hub)
    assert set(fc["model"]) == {"team-a"}
    assert fc["truth"].isna().all()
    assert len(fc) == 200


def test_studies_whose_loaders_are_not_merged_run_on_dummy_data() -> None:
    assert set(DUMMY_STUDIES) == {"synthetic", "germany"}
