import pandas as pd

from paperfigs.dummy import dummy_factors, dummy_study
from paperfigs.nowcasts import (
    other_places,
    pick_origins,
    pick_outbreaks,
    place_models,
)


def test_origins_are_seeded_and_do_not_overlap() -> None:
    fc, _ = dummy_study("germany", seed=1, n_samples=5)
    a = pick_origins(fc, 3, seed=7)
    assert a == pick_origins(fc, 3, seed=7)
    assert len(a) == 3
    gaps = pd.Series(a).diff().dropna()
    span = (fc["target_date"] - fc["origin_date"]).agg(["min", "max"])
    assert (gaps > span["max"] - span["min"]).all()


def test_other_places_are_states_then_age_groups() -> None:
    fc, _ = dummy_study("germany", seed=1, n_samples=5)
    places = other_places(fc, "DE")
    locations = [p[0] for p in places]
    assert "DE" not in [loc for loc, st, _ in places if st == "all"]
    assert any(st == "80+" for _, st, _ in places)
    assert locations.index("DE-BY") < locations.index("DE")


def test_other_places_leave_out_deeper_levels() -> None:
    fc = pd.DataFrame(
        {"location": ["CD", "CD-IT", "CD-IT-01"], "stratum": "all"}
    )
    assert [p[0] for p in other_places(fc, "CD")] == ["CD-IT"]


def test_outbreaks_cover_factor_levels_and_name_their_cell() -> None:
    fc, _ = dummy_study("synthetic", seed=1, n_samples=5)
    cells = dummy_factors(fc, seed=1)
    panels = pick_outbreaks(cells, ["observation", "strata"], 4, seed=3)
    assert panels == pick_outbreaks(cells, ["observation", "strata"], 4, 3)
    assert len(panels) == 4
    assert all("Observation:" in title for _, _, title in panels)


def test_models_without_forecasts_are_left_out() -> None:
    assert place_models(["a", "c"], ["c", "b", "a"]) == ["c", "a"]
