"""The workflows keep Actions caches and artefacts small."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

GITHUB = Path(__file__).parents[1] / ".github"
WORKFLOWS = sorted((GITHUB / "workflows").glob("*.yml"))
ACTIONS = sorted((GITHUB / "actions").glob("*/action.yml"))


def _steps(path: Path) -> Iterator[dict]:
    doc = yaml.safe_load(path.read_text())
    if "runs" in doc:
        yield from doc["runs"].get("steps", [])
        return
    for job in doc.get("jobs", {}).values():
        yield from job.get("steps", [])


def _uses(path: Path, action: str) -> Iterator[dict]:
    for step in _steps(path):
        if step.get("uses", "").startswith(action):
            yield step


def test_every_artefact_expires_within_a_week_but_a_live_submission() -> None:
    for path in WORKFLOWS:
        for step in _uses(path, "actions/upload-artifact"):
            name = f"{path.name}: {step['with']['name']}"
            days = step["with"].get("retention-days")
            assert days is not None, f"{name} has no retention-days"
            if isinstance(days, int):
                assert days <= 7, name
            else:
                # The hub submission's record: kept long for a live
                # round, briefly for a dry run.
                assert "dry_run" in days, name


def test_no_cache_key_is_made_per_run() -> None:
    for path in WORKFLOWS + ACTIONS:
        for action in ("actions/cache", "actions/cache/restore"):
            for step in _uses(path, action):
                key = str(step["with"].get("key", ""))
                assert "run_id" not in key, f"{path.name}: {key}"
                assert "run_number" not in key, f"{path.name}: {key}"
                assert "run_attempt" not in key, f"{path.name}: {key}"


def test_pull_requests_restore_the_jax_cache_without_saving_it() -> None:
    (env,) = [p for p in ACTIONS if p.parent.name == "setup-forecast-env"]
    jax = [
        s
        for s in _steps(env)
        if s.get("with", {}).get("path") == "~/.cache/jax"
    ]
    saving = [s for s in jax if s["uses"].split("@")[0] == "actions/cache"]
    assert len(saving) == 1
    assert "github.event_name != 'pull_request'" in saving[0]["if"]


def test_a_closed_pull_request_s_caches_are_deleted() -> None:
    path = GITHUB / "workflows" / "pr-closed.yml"
    jobs = yaml.safe_load(path.read_text())["jobs"]
    deletes = [
        job
        for job in jobs.values()
        if any("gh cache delete" in s.get("run", "") for s in job["steps"])
    ]
    assert len(deletes) == 1
    (job,) = deletes
    assert job["permissions"]["actions"] == "write"
    steps = yaml.safe_dump(job["steps"])
    assert "refs/pull/${{ github.event.number }}/merge" in steps
