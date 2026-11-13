import pandas as pd

from paperpipe.config import seed_of
from paperpipe.studies import load_data, run_forecasts

SPLIT = pd.Timestamp("2022-02-10")


def test_synthetic_data_has_one_set_of_outbreaks_per_seed() -> None:
    obs, locations = load_data("synthetic", seeds=(7001, 7002))
    assert {seed_of(loc) for loc in obs["location"]} == {7001, 7002}
    assert set(locations["location"]) == set(obs["location"])
    assert {"delay_days", "delay_q99", "gt_mean"} <= set(locations.columns)
    assert (locations["delay_days"] == locations["delay_q99"]).all()


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
