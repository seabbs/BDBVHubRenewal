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


def test_pick_places_follows_the_rule() -> None:
    from paperfigs.nowcasts import pick_places

    dates = pd.date_range("2026-01-03", periods=5, freq="7D")
    series = {
        "big": [100, 100, 100, 100, 100],
        "mid": [10, 10, 10, 10, 10],
        "tiny": [1, 1, 1, 1, 1],
        "up": [2, 4, 8, 16, 32],
        "down": [32, 16, 8, 4, 2],
        "none": [0, 0, 0, 0, 0],
    }
    truth = pd.DataFrame(
        [
            {"location": k, "stratum": "all", "target_date": d, "truth": v}
            for k, vals in series.items()
            for d, v in zip(dates, vals, strict=True)
        ]
    )
    places = [(k, "all", k) for k in series]
    picked = pick_places(truth, places, dates[-1], seed=1)
    reasons = [p[2].split("(")[1].rstrip(")") for p in picked]
    assert reasons == ["largest", "median", "small", "rising", "falling"]
    assert picked[0][0] == "big"
    assert picked[2][0] == "tiny"
    assert len({p[0] for p in picked}) == 5
