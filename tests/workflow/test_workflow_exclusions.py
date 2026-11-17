"""Released QC exclusions: fetched, or the rule fails."""

import json
import subprocess

import pytest
from flowlib import exclusions


def fake_gh(releases):
    """A `gh` that knows `releases`: tag -> asset names, or an error."""
    calls = []

    def run(args, **_):
        calls.append(args)
        tag = args[3]
        assets = releases.get(tag)
        if isinstance(assets, Exception):
            return subprocess.CompletedProcess(args, 1, "", str(assets))
        if args[2] == "view":
            if assets is None:
                return subprocess.CompletedProcess(
                    args, 1, "", "release not found"
                )
            body = json.dumps({"assets": [{"name": a} for a in assets]})
            return subprocess.CompletedProcess(args, 0, body, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    return run, calls


def test_a_released_exclusion_file_is_downloaded(tmp_path):
    run, calls = fake_gh(
        {"sims-r1": ["counts-30100.csv.gz", "excluded-r1.csv"]}
    )
    got = exclusions.fetch_released([1], "o/r", tmp_path, run=run)
    assert got == {1: "downloaded"}
    assert calls[-1][2] == "download"


def test_a_round_not_yet_reviewed_is_noted_not_failed(tmp_path):
    run, _ = fake_gh({"sims-r2": ["counts-30200.csv.gz"]})
    assert exclusions.fetch_released([2], "o/r", tmp_path, run=run) == {
        2: "not reviewed yet"
    }


def test_a_missing_release_fails(tmp_path):
    run, _ = fake_gh({})
    with pytest.raises(RuntimeError, match="sims-r3"):
        exclusions.fetch_released([3], "o/r", tmp_path, run=run)


def test_a_gh_error_fails(tmp_path):
    run, _ = fake_gh({"sims-r1": RuntimeError("HTTP 401: bad credentials")})
    with pytest.raises(RuntimeError, match="401"):
        exclusions.fetch_released([1], "o/r", tmp_path, run=run)
