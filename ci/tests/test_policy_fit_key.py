from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from submission_ci.fit_key import fit_key, source_hash
from submission_ci.policy import (
    fit_timeout,
    load_policy,
    matrix,
    retry_fits,
)
from submission_ci.window import minutes_left

POLICY_PATH = Path(__file__).parents[1] / "policy.toml"

# Measured on archie (jax-review-1, 2026-10-24 vintage, vectorised chains
# on 2 cores): main 130 and retry 134 minutes of sampling at 4 chains x
# 2000 iterations. Cost follows iterations (trees at maximum depth), so
# 2 chains x 1000 take a quarter. Hosted runners are up to 1.5 times
# slower, plus about 15 minutes of setup.
SETUP, PREPARE = 15, 15
MAIN = 130 / 4 * 1.5 + SETUP
RETRY = 134 / 4 * 1.5 + SETUP
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
    root = repo(tmp_path)
    entries = matrix(pol, "retry", root, "hub")
    assert [e["id"] for e in entries] == ["retry-1"]
    main = matrix(pol, "primary", root, "hub")
    keys = [e["key"] for e in [*entries, *main]]
    assert len(set(keys)) == len(keys)
    sampler = json.loads(entries[0]["sampler"])
    assert sampler["target_accept"] == 0.95
    assert sampler["init"] == "jitter"


def iterations(sampler: dict[str, int]) -> int:
    """NUTS iterations over all chains, which sets a fit's wall time when
    nearly every draw reaches the maximum tree depth."""
    return sampler["num_chains"] * (
        sampler["num_warmup"] + sampler["num_samples"]
    )


def test_retry_fits_the_time_budget(tmp_path: Path) -> None:
    # The last Tuesday start leaves about 540 minutes for main and the
    # retry in turn, and each job is capped at 350. A retry costing no
    # more than main, with no deeper trees, keeps both within the window.
    pol = load_policy(POLICY_PATH)
    root = repo(tmp_path)
    main = json.loads(matrix(pol, "primary", root, "hub")[0]["sampler"])
    retry = json.loads(matrix(pol, "retry", root, "hub")[0]["sampler"])
    assert iterations(retry) <= iterations(main)
    assert retry["max_tree_depth"] <= main["max_tree_depth"]
    assert retry["num_warmup"] >= main["num_warmup"]
    # No more chains than a hosted runner's 4 vCPUs.
    assert retry["num_chains"] <= 4


def test_chain_stays_within_the_three_models(tmp_path: Path) -> None:
    # Sam (2026-10-28): renewal, its refit, then the independent
    # single-patch fits (renewalsp) fitted beside main as the standby,
    # then persistence. renewalbvd is a shadow. No separate fallback
    # configuration.
    pol = load_policy(POLICY_PATH)
    assert [(c.id, c.kind) for c in pol.chain()] == [
        ("main", "main"),
        ("retry-1", "retry"),
        ("renewalsp", "fallback"),
        ("persistence", "persistence"),
    ]
    assert "fallback" not in {c.config for c in pol.candidates}
    root = repo(tmp_path)
    primary = {e["id"]: e for e in matrix(pol, "primary", root, "hub")}
    assert list(primary) == ["main", "renewalsp", "bvd"]
    # The model each leg runs: the chosen model unless the candidate
    # names its own.
    assert primary["main"]["model"] == ""
    assert primary["renewalsp"]["model"] == "renewalsp"
    assert primary["bvd"]["model"] == "renewalbvd"
    assert matrix(pol, "retry", root, "hub")[0]["model"] == ""


def test_ci_sampler_settings(tmp_path: Path) -> None:
    # Sam (2026-10-28): 2 chains of 500 warm-up and 500 draws for every
    # fitted candidate; the retry differs by its target acceptance and
    # jittered inits.
    pol = load_policy(POLICY_PATH)
    root = repo(tmp_path)
    fitted = [
        *matrix(pol, "primary", root, "hub"),
        *matrix(pol, "retry", root, "hub"),
    ]
    for e in fitted:
        s = json.loads(e["sampler"])
        assert (s["num_chains"], s["num_warmup"], s["num_samples"]) == (
            2,
            500,
            500,
        ), e["id"]
    retry = json.loads(matrix(pol, "retry", root, "hub")[0]["sampler"])
    assert retry["target_accept"] == 0.95 and retry["init"] == "jitter"


def test_policy_validation(tmp_path: Path) -> None:
    bad = tmp_path / "p.toml"
    bad.write_text('[[candidates]]\nid = "x"\nkind = "fallback"\n')
    with pytest.raises(ValueError, match="first candidate"):
        load_policy(bad)
    bad.write_text('[[candidates]]\nid = "x"\nkind = "magic"\n')
    with pytest.raises(ValueError, match="unknown kind"):
        load_policy(bad)


def test_retry_budget_caps_retries(tmp_path: Path) -> None:
    text = POLICY_PATH.read_text().replace(
        "retry_budget = 2", "retry_budget = 1"
    )
    p = tmp_path / "p.toml"
    p.write_text(text)
    assert [c.id for c in load_policy(p).stage("retry")] == ["retry-1"]


def test_shadow_candidates_are_fitted_but_never_in_the_chain(
    tmp_path: Path,
) -> None:
    pol = load_policy(POLICY_PATH)
    shadows = [c for c in pol.candidates if c.kind == "shadow"]
    assert [c.id for c in shadows] == ["bvd"]
    assert all(c.id != "bvd" for c in pol.chain())
    root = repo(tmp_path)
    primary = [e["id"] for e in matrix(pol, "primary", root, "hub")]
    assert "bvd" in primary
    assert primary[0] == "main"


def test_persistence_chosen_makes_the_renewal_candidates_shadows(
    tmp_path: Path,
) -> None:
    pol = load_policy(POLICY_PATH).for_model("persistence")
    assert [c.id for c in pol.chain()] == ["persistence"]
    root = repo(tmp_path)
    primary = [e["id"] for e in matrix(pol, "primary", root, "hub")]
    assert primary == ["main", "renewalsp", "bvd"]
    assert matrix(pol, "retry", root, "hub") == []
    assert all(c.kind == "shadow" for c in pol.candidates if c.id in primary)
    # Any other choice keeps the full chain.
    full = load_policy(POLICY_PATH).for_model("renewal")
    assert next(c.id for c in full.chain()) == "main"


def test_timing_defaults() -> None:
    t = load_policy(POLICY_PATH).timing
    assert t.fit_cap_minutes == 350
    assert t.reserve_minutes >= 30
    # The retry's measured 134 minutes at 4 chains x 2000 iterations,
    # scaled to 2 x 1000 (a quarter), 1.5 times slower on a hosted runner,
    # plus setup, and not so much more that a refit is skipped needlessly.
    assert RETRY + 10 <= t.retry_minutes <= 2 * RETRY


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
    # falls through to the fallback or persistence.
    assert fit_timeout(pol, 10) == 30


def test_retry_runs_only_with_time_for_it() -> None:
    pol = load_policy(POLICY_PATH)
    t = pol.timing
    assert retry_fits(pol, None)
    assert retry_fits(pol, t.retry_minutes + t.reserve_minutes)
    assert not retry_fits(pol, t.retry_minutes + t.reserve_minutes - 1)


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
    # The single-patch fallback runs beside main and is ready first, so
    # the standby never delays the chain.
    assert SINGLE_PATCH < MAIN
    # A main fit of normal length leaves time for the retry.
    pol = load_policy(POLICY_PATH)
    t = pol.timing
    window = 540  # the last Tuesday start, 14:00 UTC, to the close
    after_main = window - PREPARE - MAIN
    assert retry_fits(pol, after_main)
    assert fit_timeout(pol, after_main) >= RETRY
    # A main fit that runs to its job cap still leaves the gate its time,
    # and a retry runs then only if its timeout covers a normal refit.
    after_cap = window - PREPARE - fit_timeout(pol, window - PREPARE)
    assert after_cap >= t.reserve_minutes
    if retry_fits(pol, after_cap):
        assert fit_timeout(pol, after_cap) >= RETRY
