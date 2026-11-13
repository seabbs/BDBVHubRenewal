"""The submission workflow's wiring of the hub models it may submit."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_only_the_chronos_job_submits_chronos_and_only_when_on() -> None:
    workflow = ROOT / ".github" / "workflows" / "submit.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    prepare = jobs["prepare"]["outputs"]
    assert prepare["chronos"] == "${{ steps.tracks.outputs.chronos }}"
    assert prepare["chronos_model"] == (
        "${{ steps.tracks.outputs.chronos_model }}"
    )
    job = jobs["chronos"]
    assert "needs.prepare.outputs.chronos == 'true'" in job["if"]
    track = next(s for s in job["steps"] if s.get("id") == "track")
    assert track["run"] == "ci/steps/track.sh"
    assert track["env"]["TRACK_MODEL"] == (
        "${{ needs.prepare.outputs.chronos_model }}"
    )
    # Every other job that runs track.sh is the baseline's.
    runs = [
        name
        for name, j in jobs.items()
        for s in j.get("steps", [])
        if s.get("run") == "ci/steps/track.sh"
    ]
    assert sorted(runs) == ["baseline", "chronos"]
    assert "chronos" in jobs["record"]["needs"]
    assert "chronos" in jobs["outcome"]["needs"]
