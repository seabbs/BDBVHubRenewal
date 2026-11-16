"""German nowcasts by checkpoint, for the configured runs and references."""

import csv
import gzip

import pytest
from flowlib.dashboard import germany, runs

LEVELS = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")


def _write(path, model, location="DE", truth=10.0, base=8.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(
            f,
            [
                "model",
                "location",
                "target_end_date",
                "quantile",
                "value",
                "truth",
                "truth_vintage",
            ],
        )
        w.writeheader()
        for i, q in enumerate(LEVELS):
            w.writerow(
                {
                    "model": model,
                    "location": location,
                    "target_end_date": "2022-02-01",
                    "quantile": q,
                    "value": base + i,
                    "truth": truth,
                    "truth_vintage": "2022-03-01",
                }
            )


def _setup(tmp_path):
    cache = tmp_path / "germany"
    _write(cache / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged")
    _write(cache / "reference" / "2022-02-10.csv.gz", "randomwalknowcast")
    _write(
        cache / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
    )
    _write(cache / "bbbbbb" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged_ft")
    rs = runs.load_runs([], "slots=slots-,label=slots,colour=#111", [])
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#0a0",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)",
                "checkpoint": "zero-shot",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            },
        ],
        [],
    )
    return cache, rs, refs


def test_wis_from_seven_quantiles():
    q = [8, 9, 10, 11, 12, 13, 14]
    assert germany.wis(q, 11) == pytest.approx(
        (0.5 * 0 + 0.025 * 6 + 0.1 * 4 + 0.25 * 2) / 3.5
    )


def test_series_are_the_configured_references_and_runs(tmp_path):
    cache, rs, refs = _setup(tmp_path)
    built = germany.build(
        cache, rs, refs, levels=(0.025, 0.25, 0.5, 0.75, 0.975)
    )
    assert set(built["sets"]) == {"zero", "rw", "aaaaaa"}
    one = built["series"]["DE"]["aaaaaa"]["daily"]["2022-02-09"]
    # The paper's five levels, from the cached seven.
    assert one["q"] == [[8.0, 10.0, 11.0, 12.0, 14.0]]
    assert built["truth"]["DE"]["2022-02-01"] == ("2022-03-01", 10.0)
    assert built["origins"] == ["2022-02-09", "2022-02-10"]


def test_labels_and_colours_come_from_the_configuration(tmp_path):
    cache, rs, refs = _setup(tmp_path)
    built = germany.build(cache, rs, refs)
    meta = {m["id"]: m for m in germany.meta(built, rs, refs, {"aaaaaa": 250})}
    assert meta["aaaaaa"]["label"] == "slots n250 aaaaaa"
    assert meta["aaaaaa"]["run"] == "slots"
    assert meta["zero"]["label"] == "zero-shot ragged"
    assert meta["rw"]["color"] == "#666"
    # References first, then each run's checkpoints by size.
    assert [m["id"] for m in germany.meta(built, rs, refs, {})][:2] == [
        "zero",
        "rw",
    ]


def test_one_location_as_page_data(tmp_path):
    cache, rs, refs = _setup(tmp_path)
    built = germany.build(cache, rs, refs)
    got = germany.location(built, "DE")
    assert set(got) == {"series", "wis", "truth"}
    assert got["truth"] == {"t": ["2022-02-01"], "y": [10.0]}
    assert germany.location(built, "XX") == {}


def test_a_build_is_reused_until_a_cache_file_changes(tmp_path):
    cache, rs, refs = _setup(tmp_path)
    store = {}
    first = germany.cached_build(cache, rs, refs, store)
    assert germany.cached_build(cache, rs, refs, store) is first
    _write(cache / "reference" / "2022-02-11.csv.gz", "randomwalknowcast")
    again = germany.cached_build(cache, rs, refs, store)
    assert again is not first
    assert "2022-02-11" in again["origins"]
