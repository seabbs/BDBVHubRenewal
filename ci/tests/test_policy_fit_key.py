from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_ci.fit_key import fit_key, source_hash
from submission_ci.policy import load_policy, matrix

POLICY_PATH = Path(__file__).parents[1] / "policy.toml"


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
    assert sampler["num_chains"] == 8 and sampler["init"] == "jitter"


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
    assert primary == ["main", "fallback", "bvd"]
    assert matrix(pol, "retry", root, "hub") == []
    assert all(c.kind == "shadow" for c in pol.candidates if c.id in primary)
    # Any other choice keeps the full chain.
    full = load_policy(POLICY_PATH).for_model("renewal")
    assert next(c.id for c in full.chain()) == "main"
