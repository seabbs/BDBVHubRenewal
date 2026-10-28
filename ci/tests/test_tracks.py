"""Which hub models a run submits: opted in, and ready."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_ci.tracks import load_tracks, main

HUB_TOML = Path(__file__).parents[2] / "config" / "hub.toml"


def write(path: Path, renewal: bool, baseline: bool) -> Path:
    path.write_text(
        "[submission]\n"
        'model_abbr = "renewal"\n'
        f"submit = {str(renewal).lower()}\n"
        "[models.persistencenow]\n"
        'model_abbr = "baseline"\n'
        f"submit = {str(baseline).lower()}\n"
    )
    return path


def test_repository_config_names_both_hub_models() -> None:
    tracks = {t.name: t for t in load_tracks(HUB_TOML, chosen="")}
    assert tracks["renewal"].model_id == "seabbs-renewal"
    base = tracks["baseline"]
    assert base.model_id == "seabbs-baseline"
    assert base.model == "persistencenow"
    # Sam opted seabbs-baseline in for the 2026-11-07 round (2026-10-28);
    # seabbs-renewal waits for its models. DRY_RUN still applies.
    assert base.submit
    assert base.runs
    assert not tracks["renewal"].submit


def test_opted_in_baseline_runs_without_a_model_choice(tmp_path: Path) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    tracks = {t.name: t for t in load_tracks(cfg, chosen="")}
    assert tracks["baseline"].runs
    assert not tracks["renewal"].runs
    assert "not opted in" in tracks["renewal"].reason


def test_renewal_needs_a_model_choice(tmp_path: Path) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=True, baseline=False)
    renewal = next(
        t for t in load_tracks(cfg, chosen="") if t.name == "renewal"
    )
    assert not renewal.runs
    assert "no model choice" in renewal.reason
    renewal = next(
        t for t in load_tracks(cfg, chosen="renewal") if t.name == "renewal"
    )
    assert renewal.runs
    assert renewal.model == "renewal"


def test_cli_sets_one_output_per_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert main(["--hub-config", str(cfg), "--chosen", ""]) == 0
    text = out.read_text()
    assert "renewal=false" in text
    assert "baseline=true" in text
    assert "baseline_id=seabbs-baseline" in text


def test_renewal_skips_without_opt_in_even_with_a_choice(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    tracks = {t.name: t for t in load_tracks(path, chosen="persistencenow")}
    assert not tracks["renewal"].runs
    assert tracks["renewal"].reason == "not opted in (`submit = false`)"
    assert tracks["baseline"].runs
