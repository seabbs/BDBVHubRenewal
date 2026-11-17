"""Two hosts share a round: the seed split and the push of a batch."""

import hashlib
import subprocess
from pathlib import Path

import pytest
from flowlib import hosts, manifest

ROOT = Path(__file__).parents[2]
CAP = ROOT / "workflow" / "scripts" / "mem_cap.sh"
TIMED = ROOT / "workflow" / "scripts" / "timed.sh"
SPLIT = {"laptop": ["31005", "31007", "31900-31939"]}


def test_seed_specs_are_seeds_and_ranges():
    assert hosts.seeds(["31005", 31007, "31900-31902"]) == {
        31005,
        31007,
        31900,
        31901,
        31902,
    }


def test_a_listed_host_simulates_only_its_seeds():
    assert hosts.simulates(31005, "laptop", SPLIT)
    assert hosts.simulates(31939, "laptop", SPLIT)
    assert not hosts.simulates(31006, "laptop", SPLIT)


def test_any_other_host_simulates_every_seed_not_listed():
    assert hosts.simulates(31006, "archie", SPLIT)
    assert not hosts.simulates(31005, "archie", SPLIT)
    assert hosts.simulates(31005, "archie", {})


def test_a_seed_listed_for_two_hosts_is_an_error():
    with pytest.raises(ValueError, match="31005"):
        hosts.check({"a": ["31005"], "b": ["31000-31010"]})


def _batch(sims: Path, seed: int) -> None:
    (sims / "logs").mkdir(parents=True)
    (sims / f"counts-{seed}.csv.gz").write_bytes(b"counts")
    (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    (sims / "logs" / f"{seed}.log").write_text("peak_rss_kb 10 wall_s 2\n")


def test_a_pushed_batch_lands_before_its_manifest_row(tmp_path):
    sims = tmp_path / "here" / "r10" / "sims"
    _batch(sims, 31005)
    there = tmp_path / "there" / "r10"
    row = "10,31005,block,r5,abc,1,21,x,y,,10,2,laptop,now"
    hosts.push(sims, 31005, there, row, remote="")
    assert (there / "sims" / "counts-31005.csv.gz").read_bytes() == b"counts"
    assert (there / "sims" / "settings-31005.csv").exists()
    assert (there / "sims" / "logs" / "31005.log").exists()
    lines = (there / "manifest.csv").read_text().splitlines()
    assert lines == [manifest.HEADER, row]
    # Pushed again, the row is not repeated.
    hosts.push(sims, 31005, there, row, remote="")
    assert (there / "manifest.csv").read_text().count("31005") == 1


def test_the_row_of_a_pushed_batch_hashes_its_files(tmp_path):
    sims = tmp_path / "sims"
    _batch(sims, 31005)
    row = manifest.row(
        sims,
        round_=10,
        seed=31005,
        kind="block",
        grid="r5",
        commit="abc",
        draws=1,
        batch=21,
        host="laptop",
        finished="now",
    )
    assert row.split(",")[7] == hashlib.sha256(b"counts").hexdigest()


def test_the_memory_cap_kills_a_job_over_it():
    out = subprocess.run(
        ["bash", str(CAP), "1K", "sleep", "30"],
        env={"PATH": "/usr/bin:/bin", "CAP_POLL": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode != 0
    assert "over the" in out.stderr


def test_the_memory_cap_passes_on_a_job_under_it():
    out = subprocess.run(
        ["bash", str(CAP), "1024M", "bash", "-c", "echo hi; exit 3"],
        env={"PATH": "/usr/bin:/bin", "CAP_POLL": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 3
    assert out.stdout == "hi\n"


def test_timing_writes_the_gnu_time_line_on_any_os():
    out = subprocess.run(
        ["bash", str(TIMED), "bash", "-c", "echo err >&2; echo out"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 0
    assert out.stdout == "out\n"
    assert "err" in out.stderr
    last = out.stderr.strip().splitlines()[-1].split()
    assert last[0] == "peak_rss_kb"
    assert int(last[1]) > 0
    assert last[2] == "wall_s"
    float(last[3])
