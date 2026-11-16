"""Stored forecasts of the fine-tuned model on the German benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from bdbvhubrenewal.evaluation.germany_store import (
    STORED_MODEL,
    choose_checkpoint,
    development_rows,
    load_stored,
    release_tag,
    write_stored,
)

from bdbvhubrenewal.evaluation.germany import load_germany_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_germany_config(ROOT / "config" / "germany.toml")
SPLIT = CONFIG.split_date


def _cache(origins: tuple[str, ...], model: str = STORED_MODEL) -> pd.DataFrame:
    rows = []
    for origin in origins:
        for h in (0, -7):
            end = pd.Timestamp(origin) + pd.Timedelta(days=h)
            for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975):
                rows.append(
                    {
                        "model": model,
                        "origin": origin,
                        "location": "DE",
                        "target_end_date": str(end)[:10],
                        "horizon": h,
                        "quantile": q,
                        "value": 100.0 * q,
                        "truth": 50.0,
                        "truth_vintage": str(end + pd.Timedelta(days=40))[:10],
                    }
                )
    return pd.DataFrame(rows)


def _release(tag: str, published: str, draft: bool = False) -> dict:
    return {"tag_name": tag, "published_at": published, "draft": draft}


RELEASES = [
    _release("chronos-ft-aaa", "2026-11-15T10:00:00Z"),
    _release("chronos-ft-bbb", "2026-11-16T10:00:00Z"),
    _release("realdata-ft-ccc", "2026-11-16T12:00:00Z"),
]


def test_release_tag_maps_scorer_labels() -> None:
    assert release_tag("7962fd718a47") == "chronos-ft-7962fd718a47"
    assert release_tag("slots-1234abcd") == "chronos-slots-1234abcd"
    for bad in ("reference", "", "slots-", "../x", "chronos-ft-abc"):
        with pytest.raises(ValueError, match="label"):
            release_tag(bad)


def test_latest_pin_takes_the_newest_ft_release() -> None:
    stored = {"chronos-ft-aaa", "chronos-ft-bbb"}
    assert choose_checkpoint(stored, RELEASES, "latest") == (
        "chronos-ft-bbb",
        "chronos-ft-bbb",
    )


def test_a_hash_pin_takes_its_release() -> None:
    stored = {"chronos-ft-aaa", "chronos-ft-bbb"}
    assert choose_checkpoint(stored, RELEASES, "aaa") == (
        "chronos-ft-aaa",
        "chronos-ft-aaa",
    )


def test_the_newest_stored_slots_checkpoint_wins() -> None:
    releases = [
        *RELEASES,
        _release("chronos-slots-s1", "2026-11-17T10:00:00Z"),
        _release("chronos-slots-s2", "2026-11-18T10:00:00Z"),
        _release("chronos-slots-s3", "2026-11-19T10:00:00Z", draft=True),
    ]
    # s2 is newer but has no stored forecasts yet; s3 is a draft.
    stored = {"chronos-ft-bbb", "chronos-slots-s1", "chronos-slots-s3"}
    assert choose_checkpoint(stored, releases, "latest") == (
        "chronos-slots-s1",
        "chronos-ft-bbb",
    )
    stored |= {"chronos-slots-s2"}
    assert choose_checkpoint(stored, releases, "latest")[0] == (
        "chronos-slots-s2"
    )


def test_an_unstored_pin_falls_back_to_the_newest_stored_ft() -> None:
    # The pinned release is out but the scorer has not stored it yet.
    assert choose_checkpoint({"chronos-ft-aaa"}, RELEASES, "latest") == (
        "chronos-ft-aaa",
        "chronos-ft-bbb",
    )
    assert choose_checkpoint(set(), RELEASES, "latest") == (
        None,
        "chronos-ft-bbb",
    )


def test_development_rows_keep_the_model_before_the_split() -> None:
    frame = pd.concat(
        [
            _cache(("2021-11-22", "2022-02-09")),
            _cache(("2021-11-22",), model="fm_chronos_weekly_ragged_ft"),
        ]
    )
    out = development_rows(frame, SPLIT)
    assert set(out["model"]) == {STORED_MODEL}
    assert out["origin"].max() == "2022-02-09"
    assert (out["target_end_date"] < SPLIT).all()


def test_development_rows_refuse_held_out_dates() -> None:
    frame = _cache(("2022-02-09", SPLIT))
    with pytest.raises(ValueError, match="held-out"):
        development_rows(frame, SPLIT)


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    frame = _cache(("2021-11-22", "2021-11-23"))
    store = tmp_path / "store"
    write_stored(store, "chronos-ft-bbb", frame, SPLIT, label="bbb")
    manifest = json.loads(
        (store / "chronos-ft-bbb" / "manifest.json").read_text()
    )
    assert manifest["model"] == STORED_MODEL
    assert manifest["first_origin"] == "2021-11-22"
    assert manifest["last_origin"] == "2021-11-23"
    assert manifest["split_date"] == SPLIT
    back = load_stored(store, "chronos-ft-bbb", SPLIT)
    pd.testing.assert_frame_equal(
        back.reset_index(drop=True),
        development_rows(frame, SPLIT).reset_index(drop=True),
        check_dtype=False,
    )


def test_load_refuses_a_store_with_held_out_dates(tmp_path: Path) -> None:
    path = tmp_path / "store" / "chronos-ft-bbb"
    path.mkdir(parents=True)
    _cache(("2022-02-09", "2022-02-11")).to_csv(
        path / "forecasts.csv.gz", index=False
    )
    with pytest.raises(ValueError, match="held-out"):
        load_stored(tmp_path / "store", "chronos-ft-bbb", SPLIT)


def test_the_store_script_writes_a_label_from_the_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    day = cache / "germany" / "bbb"
    day.mkdir(parents=True)
    for origin in ("2021-11-22", "2021-11-23"):
        pd.concat(
            [
                _cache((origin,)),
                _cache((origin,), model="fm_chronos_weekly_ragged_ft"),
            ]
        ).to_csv(day / f"{origin}.csv.gz", index=False)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "germany_ft_store.py"),
            str(cache),
            "bbb",
            "--no-push",
            "--partial",
            "--out",
            str(out),
            "--config",
            str(ROOT / "config" / "germany.toml"),
        ],
        check=True,
    )
    stored = load_stored(out, "chronos-ft-bbb", SPLIT)
    assert set(stored["model"]) == {STORED_MODEL}
    assert sorted(set(stored["origin"])) == ["2021-11-22", "2021-11-23"]


def test_the_store_script_refuses_a_partial_label(tmp_path: Path) -> None:
    day = tmp_path / "cache" / "germany" / "bbb"
    day.mkdir(parents=True)
    _cache(("2021-11-22",)).to_csv(day / "2021-11-22.csv.gz", index=False)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "germany_ft_store.py"),
            str(tmp_path / "cache"),
            "bbb",
            "--no-push",
            "--out",
            str(tmp_path / "out"),
            "--config",
            str(ROOT / "config" / "germany.toml"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "79 development dates missing" in run.stderr


def test_the_load_command_writes_the_chosen_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal import cli
    from bdbvhubrenewal.forecasters import foundation

    store = tmp_path / "store"
    for tag in ("chronos-ft-aaa", "chronos-ft-bbb"):
        write_stored(store, tag, _cache(("2021-11-22",)), SPLIT, label=tag)
    monkeypatch.setattr(foundation, "finetuned_releases", lambda: RELEASES)
    monkeypatch.delenv(foundation.FINETUNED_ENV, raising=False)
    out = tmp_path / "parts" / "germany"
    note = tmp_path / "results" / "ft-checkpoint.json"
    assert (
        cli.main(
            [
                "germany-ft-load",
                "--config",
                str(ROOT / "config" / "germany.toml"),
                "--finetune",
                str(ROOT / "config" / "finetune.toml"),
                "--store",
                str(store),
                "--out",
                str(out),
                "--note",
                str(note),
            ]
        )
        == 0
    )
    meta = json.loads(note.read_text())
    assert meta["tag"] == "chronos-ft-bbb"
    assert meta["pinned"] == "chronos-ft-bbb"
    assert meta["model"] == STORED_MODEL
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["model"]) == {STORED_MODEL}
    assert (scores["origin"] < SPLIT).all()
    assert (out / "forecasts.csv.gz").exists()


def test_the_load_command_writes_nothing_without_a_stored_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal import cli
    from bdbvhubrenewal.forecasters import foundation

    monkeypatch.setattr(foundation, "finetuned_releases", lambda: RELEASES)
    monkeypatch.delenv(foundation.FINETUNED_ENV, raising=False)
    out = tmp_path / "parts" / "germany"
    note = tmp_path / "ft-checkpoint.json"
    args = [
        "germany-ft-load",
        "--config",
        str(ROOT / "config" / "germany.toml"),
        "--finetune",
        str(ROOT / "config" / "finetune.toml"),
        "--store",
        str(tmp_path / "missing"),
        "--out",
        str(out),
        "--note",
        str(note),
    ]
    assert cli.main(args) == 0
    assert not (out / "forecasts.csv.gz").exists()
    assert json.loads(note.read_text())["tag"] is None
