"""Checkpoint labels from the repository's releases."""

from flowlib import releases

LISTING = [
    {"tagName": "chronos-ft-old", "publishedAt": "2026-11-10T00:00:00Z"},
    {"tagName": "chronos-ft-aaa", "publishedAt": "2026-11-16T12:00:00Z"},
    {"tagName": "chronos-slots-bbb", "publishedAt": "2026-11-16T15:00:00Z"},
    {"tagName": "chronos-slots-ccc", "publishedAt": "2026-11-16T10:00:00Z"},
    {"tagName": "sims-r4", "publishedAt": "2026-11-16T11:00:00Z"},
    {"tagName": "germany-ft-de", "publishedAt": "2026-11-16T11:00:00Z"},
]


def test_slots_checkpoints_first_then_oldest_first():
    got = releases.labels(LISTING, "2026-11-16T09:00:00Z")
    assert got == ["slots-ccc", "slots-bbb", "aaa"]


def test_earlier_releases_are_left_out():
    got = releases.labels(LISTING, "2026-11-01T00:00:00Z")
    assert got[-1] == "aaa"
    assert "old" in got


def test_a_batch_is_released_once_both_its_files_are():
    names = [
        "counts-30900.csv.gz",
        "settings-30900.csv",
        "counts-30901.csv.gz",
        "settings-30902.csv",
        "qc.csv",
    ]
    assert releases.released_seeds(names) == {30900}
