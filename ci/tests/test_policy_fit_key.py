from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from submission_ci.fit_key import fit_key, source_hash
from submission_ci.policy import fit_timeout, load_policy, matrix
from submission_ci.window import minutes_left

POLICY_PATH = Path(__file__).parents[1] / "policy.toml"

# Measured on archie (jax-review-1, 2026-10-24 vintage, vectorised chains
# on 2 cores): main 130 minutes of sampling at 4 chains x 2000
# iterations. Cost follows iterations (trees at maximum depth), so
# 2 chains x 1000 take a quarter. Hosted runners are up to 1.5 times
# slower, plus about 15 minutes of setup.
SETUP, PREPARE = 15, 15
MAIN = 130 / 4 * 1.5 + SETUP
# The single-patch fallback at 2 x (500 + 500) on the same vintage: 21
# minutes of sampling for all six streams (a laptop, about 1.8 cores in
# use; submission agent, 2026-10-28).
SINGLE_PATCH = 21 * 1.5 + SETUP


def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("a = 1\n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "m.pyc").write_text("junk")
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "uv.lock").write_text("")
    return tmp_path


def test_source_hash_tracks_content_not_caches(tmp_path: Path) -> None:
    root = repo(tmp_path)
    inputs = ["src", "config", "pyproject.toml", "uv.lock"]
    h1 = source_hash(root, inputs)
    (root / "src" / "__pycache__" / "m.pyc").write_text("other")
    assert source_hash(root, inputs) == h1
    (root / "src" / "m.py").write_text("a = 2\n")
    assert source_hash(root, inputs) != h1


def test_missing_input_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        source_hash(tmp_path, ["src"])


def test_key_changes_with_data_and_settings() -> None:
    base = fit_key("s", "hub1", {"config": "main", "sampler": {"seed": 1}})
    assert base == fit_key(
        "s", "hub1", {"sampler": {"seed": 1}, "config": "main"}
    )
    assert base != fit_key(
        "s", "hub2", {"config": "main", "sampler": {"seed": 1}}
    )
    assert base != fit_key(
        "s", "hub1", {"config": "main", "sampler": {"seed": 2}}
    )
    assert base != fit_key(
        "t", "hub1", {"config": "main", "sampler": {"seed": 1}}
    )


def test_matrix_has_distinct_keys(tmp_path: Path) -> None:
    pol = load_policy(POLICY_PATH)
    keys = [e["key"] for e in matrix(pol, "primary", repo(tmp_path), "hub")]
    assert len(keys) == 3
    assert len(set(keys)) == len(keys)


def test_chain_runs_from_most_to_least_complex(tmp_path: Path) -> None:
    # Sam (2026-11-03): renewalbvd, renewalsp, renewal, then the random
    # walks, with no refit. Each fitted leg names the model it runs.
    pol = load_policy(POLICY_PATH)
    assert [(c.id, c.kind, c.model) for c in pol.chain()] == [
        ("bvd", "main", "renewalbvd"),
        ("renewalsp", "fallback", "renewalsp"),
        ("main", "fallback", "renewal"),
        ("randomwalknowcast", "randomwalk", "randomwalknowcast"),
        ("randomwalk", "randomwalk", ""),
    ]
    assert pol.for_model("renewalbvd") == pol
    # Both baselines need no fit and are built in `prepare`.
    assert [c.id for c in pol.stage("baseline")] == [
        "randomwalknowcast",
        "randomwalk",
    ]
    root = repo(tmp_path)
    primary = {e["id"]: e for e in matrix(pol, "primary", root, "hub")}
    assert {k: e["model"] for k, e in primary.items()} == {
        "bvd": "renewalbvd",
        "renewalsp": "renewalsp",
        "main": "renewal",
    }


def test_baseline_choices_keep_the_chain_from_that_baseline() -> None:
    pol = load_policy(POLICY_PATH)
    randomwalk = pol.for_model("randomwalk")
    assert [c.id for c in randomwalk.chain()] == ["randomwalk"]
    assert randomwalk.get("randomwalknowcast").kind == "shadow"
    now = pol.for_model("randomwalknowcast")
    assert [c.id for c in now.chain()] == ["randomwalknowcast", "randomwalk"]
    assert now.get("main").kind == "shadow"
    only = pol.for_model("randomwalknowcast", only=True)
    assert [c.id for c in only.chain()] == ["randomwalknowcast"]
    assert only.get("randomwalk").kind == "shadow"


def test_ci_sampler_settings(tmp_path: Path) -> None:
    # 2 chains of 500 warm-up, run in parallel. renewalbvd draws 1000 at
    # target acceptance 0.95 to clear its tail ESS and divergence warnings;
    # the others draw 500.
    pol = load_policy(POLICY_PATH)
    root = repo(tmp_path)
    samplers = {
        e["id"]: json.loads(e["sampler"])
        for e in matrix(pol, "primary", root, "hub")
    }
    assert set(samplers) == {"bvd", "renewalsp", "main"}
    for s in samplers.values():
        assert (s["num_chains"], s["num_warmup"]) == (2, 500)
        assert s["chain_method"] == "parallel"
    assert samplers["bvd"]["num_samples"] == 1000
    assert samplers["bvd"]["target_accept"] == 0.95
    assert (
        samplers["main"]["num_samples"]
        == samplers["renewalsp"]["num_samples"]
        == 500
    )


def test_policy_validation(tmp_path: Path) -> None:
    bad = tmp_path / "p.toml"
    cand = 'config = "x"\nstage = "primary"\nlabel = "x"\n'
    bad.write_text(f'[[candidates]]\nid = "x"\nkind = "fallback"\n{cand}')
    with pytest.raises(ValueError, match="first candidate"):
        load_policy(bad)
    bad.write_text(f'[[candidates]]\nid = "x"\nkind = "magic"\n{cand}')
    with pytest.raises(ValueError, match="unknown kind"):
        load_policy(bad)


def test_randomwalk_chosen_makes_the_renewal_candidates_shadows(
    tmp_path: Path,
) -> None:
    pol = load_policy(POLICY_PATH).for_model("randomwalk")
    assert [c.id for c in pol.chain()] == ["randomwalk"]
    root = repo(tmp_path)
    primary = [e["id"] for e in matrix(pol, "primary", root, "hub")]
    assert primary == ["bvd", "renewalsp", "main"]
    assert all(c.kind == "shadow" for c in pol.candidates if c.id in primary)


def test_timing_defaults() -> None:
    t = load_policy(POLICY_PATH).timing
    assert t.fit_cap_minutes == 350
    assert t.reserve_minutes >= 30


def test_minutes_left_to_the_window_close() -> None:
    # 2026-11-07's window closes Tuesday 3 November 23:00 UTC.
    now = dt.datetime(2026, 11, 3, 14, tzinfo=dt.UTC)
    assert minutes_left("2026-11-07", now) == 540
    assert minutes_left("", now) is None
    assert minutes_left("2026-11-07", now, ignore_window=True) is None


def test_fit_timeout_leaves_room_for_the_gate() -> None:
    pol = load_policy(POLICY_PATH)
    t = pol.timing
    assert fit_timeout(pol, None) == t.fit_cap_minutes
    assert fit_timeout(pol, 2000) == t.fit_cap_minutes
    assert fit_timeout(pol, 240) == 240 - t.reserve_minutes
    # Never below a floor: a fit that cannot finish fails and the chain
    # falls through to the fallback or randomwalk.
    assert fit_timeout(pol, 10) == 30


def test_matrix_carries_the_fit_timeout(tmp_path: Path) -> None:
    pol = load_policy(POLICY_PATH)
    root = repo(tmp_path)
    assert {e["timeout"] for e in matrix(pol, "primary", root, "hub")} == {
        "350"
    }
    late = matrix(pol, "primary", root, "hub", minutes_left=240)
    assert {e["timeout"] for e in late} == {
        str(240 - pol.timing.reserve_minutes)
    }


def test_whole_chain_fits_from_the_last_tuesday_start() -> None:
    # Every renewal candidate fits in parallel, and the single-patch
    # model is ready before the main one.
    assert SINGLE_PATCH < MAIN
    pol = load_policy(POLICY_PATH)
    window = 540  # the last Tuesday start, 14:00 UTC, to the close
    timeout = fit_timeout(pol, window - PREPARE)
    assert timeout >= MAIN
    # A fit that runs to its timeout still leaves the gate its time.
    assert window - PREPARE - timeout >= pol.timing.reserve_minutes
