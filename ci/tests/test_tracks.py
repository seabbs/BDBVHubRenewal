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
        'model = "renewal"\n'
        f"submit = {str(renewal).lower()}\n"
        "[models.randomwalknowcast]\n"
        'model_abbr = "rwnowcast"\n'
        f"submit = {str(baseline).lower()}\n"
    )
    return path


def test_repository_config_names_both_hub_models() -> None:
    tracks = {t.name: t for t in load_tracks(HUB_TOML)}
    assert tracks["renewal"].model_id == "seabbs-renewal"
    base = tracks["rwnowcast"]
    assert base.model_id == "seabbs-rwnowcast"
    assert base.model == "randomwalknowcast"
    # Both hub models are opted in, and seabbs-renewal's chain starts at
    # renewalbvd. DRY_RUN still applies.
    assert base.submit
    assert base.runs
    assert tracks["renewal"].submit
    assert tracks["renewal"].model == "renewalbvd"


def test_opted_in_baseline_runs_alone(tmp_path: Path) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    tracks = {t.name: t for t in load_tracks(cfg)}
    assert tracks["rwnowcast"].runs
    assert not tracks["renewal"].runs
    assert "not opted in" in tracks["renewal"].reason


def test_cli_sets_one_output_per_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert main(["--hub-config", str(cfg)]) == 0
    text = out.read_text()
    assert "renewal=false" in text
    assert "baseline=true" in text
    assert "baseline_id=seabbs-rwnowcast" in text


def test_renewal_skips_without_opt_in(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "hub.toml", renewal=False, baseline=True)
    tracks = {t.name: t for t in load_tracks(path)}
    assert not tracks["renewal"].runs
    assert tracks["renewal"].reason == "not opted in (`submit = false`)"
    assert tracks["rwnowcast"].runs
