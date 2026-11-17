"""Manifest rows in scripts/sim_rounds.sh's format."""

import gzip
import hashlib
import re

import pytest
from flowlib import manifest


def _batch(tmp_path, seed=30421):
    sims = tmp_path / "sims"
    (sims / "logs").mkdir(parents=True)
    with gzip.open(sims / f"counts-{seed}.csv.gz", "wt") as f:
        f.write("x\n1\n")
    (sims / f"settings-{seed}.csv").write_text("outbreak\n30421-0001\n")
    (sims / "logs" / f"{seed}.log").write_text(
        "noise\npeak_rss_kb 123\nwall_s 4.5\npeak_rss_kb 456\nwall_s 7.25\n"
    )
    return sims


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_header_matches_the_driver():
    assert manifest.HEADER == (
        "round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,"
        "settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished"
    )


def test_a_row_records_hashes_memory_and_time(tmp_path):
    sims = _batch(tmp_path)
    row = manifest.row(
        sims,
        round_=4,
        seed=30421,
        kind="train",
        grid="r4",
        commit="abc",
        draws=10,
        batch=25,
        host="h",
        finished="2026-11-16T17:00:00Z",
    )
    fields = row.split(",")
    assert fields[:7] == ["4", "30421", "train", "r4", "abc", "10", "25"]
    assert fields[7] == _sha(sims / "counts-30421.csv.gz")
    assert fields[8] == _sha(sims / "settings-30421.csv")
    # No failures file: an empty field, as the driver writes.
    assert fields[9] == ""
    # The last attempt's memory and time.
    assert fields[10:12] == ["456", "7.25"]
    assert fields[12:] == ["h", "2026-11-16T17:00:00Z"]


def test_an_existing_row_is_kept(tmp_path):
    old = tmp_path / "manifest.csv"
    old.write_text(
        manifest.HEADER + "\n4,30421,train,r4,old,10,25,a,b,,1,2,x,t\n"
    )
    assert manifest.existing(old, 4, 30421) == (
        "4,30421,train,r4,old,10,25,a,b,,1,2,x,t"
    )
    assert manifest.existing(old, 4, 30422) is None
    assert manifest.existing(tmp_path / "none.csv", 4, 30421) is None


def test_rows_are_joined_in_seed_order_under_the_header(tmp_path):
    a = tmp_path / "30422.csv"
    b = tmp_path / "30421.csv"
    a.write_text("4,30422,train\n")
    b.write_text("4,30421,train\n")
    out = tmp_path / "manifest.csv"
    manifest.join([a, b], out)
    assert out.read_text() == (
        manifest.HEADER + "\n4,30421,train\n4,30422,train\n"
    )


def _row(sims):
    return manifest.row(
        sims,
        round_=4,
        seed=30421,
        kind="train",
        grid="r4",
        commit="abc",
        draws=10,
        batch=25,
        host="h",
        finished="t",
    )


def test_a_batch_without_counts_or_settings_has_no_row(tmp_path):
    for name in ("counts-30421.csv.gz", "settings-30421.csv"):
        sims = _batch(tmp_path / name)
        (sims / name).unlink()
        with pytest.raises(FileNotFoundError, match=re.escape(name)):
            _row(sims)


def test_a_batch_without_its_log_has_no_row(tmp_path):
    sims = _batch(tmp_path)
    (sims / "logs" / "30421.log").unlink()
    with pytest.raises(FileNotFoundError, match=re.escape("30421.log")):
        _row(sims)
