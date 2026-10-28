"""Which hub models a run submits: opted in, and ready."""

from __future__ import annotations

from pathlib import Path

import pytest
from submission_ci.tracks import load_tracks, main

HUB_TOML = Path(__file__).parents[2] / "config" / "hub.toml"


def write(path: Path, renewal: bool, baseline: bool) -> Path:
    path.write_text(
        "[submission]\n"
        'model_abbr = "seabbsrenewal"\n'
        f"submit = {str(renewal).lower()}\n"
        "[models.persistencenow]\n"
        'model_abbr = "seabbsbaseline"\n'
        f"submit = {str(baseline).lower()}\n"
    )
    return path


def test_repository_config_names_both_hub_models() -> None:
    tracks = {t.name: t for t in load_tracks(HUB_TOML, chosen="")}
    assert tracks["seabbsrenewal"].model_id == "epiforecasts-seabbsrenewal"
    base = tracks["seabbsbaseline"]
    assert base.model_id == "epiforecasts-seabbsbaseline"
    assert base.model == "persistencenow"
    # Nothing is submitted until Sam opts a model in.
    assert not any(t.submit for t in tracks.values())


def test_opted_in_baseline_runs_without_a_model_choice(tmp_path: Path) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    tracks = {t.name: t for t in load_tracks(cfg, chosen="")}
    assert tracks["seabbsbaseline"].runs
    assert not tracks["seabbsrenewal"].runs
    assert "not opted in" in tracks["seabbsrenewal"].reason


def test_renewal_needs_a_model_choice(tmp_path: Path) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=True, baseline=False)
    renewal = next(
        t for t in load_tracks(cfg, chosen="") if t.name == "seabbsrenewal"
    )
    assert not renewal.runs
    assert "no model choice" in renewal.reason
    renewal = next(
        t
        for t in load_tracks(cfg, chosen="renewal")
        if t.name == "seabbsrenewal"
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
    assert "baseline_id=epiforecasts-seabbsbaseline" in text
