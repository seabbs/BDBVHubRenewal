"""The root Taskfile includes one Taskfile per project, by namespace."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

pytestmark = pytest.mark.skipif(not shutil.which("task"), reason="needs task")

# Each task before the projects had Taskfiles, and its task now.
MOVED = {
    "setup": "setup",
    "test": "test",
    "test-slow": "test-slow",
    "test-all": "test-all",
    "lint": "lint",
    "format": "format",
    "docs": "docs",
    "serve": "serve",
    "getting-started": "getting-started",
    "paper": "paper",
    "paper:targets": "paper:targets",
    "paper:render": "paper",
    "data": "nowcast-utils:data",
    "data-bvd": "nowcast-utils:data-bvd",
    "backtest": "nowcast-eval:backtest",
    "backtest-lopo": "nowcast-eval:backtest-lopo",
    "backtest-scenarios": "nowcast-eval:backtest-scenarios",
    "backtest-zones": "nowcast-eval:backtest-zones",
    "results": "nowcast-eval:results",
    "score-history": "nowcast-eval:score-history",
    "germany": "nowcast-eval:germany",
    "compare-hub": "nowcast-eval:compare-hub",
    "plots": "nowcast-eval:plots",
    "simulate": "nowcast-foundation:simulate",
    "push": "nowcast-foundation:push",
    "release": "nowcast-foundation:release",
    "finetune": "nowcast-foundation:finetune",
    "publish": "nowcast-foundation:publish",
    "score": "nowcast-foundation:score",
    "land": "nowcast-foundation:land",
    "dashboard": "nowcast-foundation:dashboard",
    "dashboard:static": "nowcast-foundation:dashboard-static",
    "all": "nowcast-workflow:all",
    "tick": "nowcast-workflow:tick",
    "run": "nowcast-workflow:run",
    "schedule": "nowcast-workflow:schedule",
    "monitor": "nowcast-workflow:monitor",
    "dag": "nowcast-workflow:dag",
    "progress": "nowcast-workflow:progress",
    "report": "nowcast-workflow:report",
    "workflow:check": "nowcast-workflow:check",
    "nowcast": "bdbv-hub:nowcast",
    "metadata": "bdbv-hub:metadata",
    "validate-hub": "bdbv-hub:validate",
    "submission-preflight": "bdbv-hub:preflight",
    "ci-test": "bdbv-hub:ci-test",
    "ci-dry": "bdbv-hub:ci-dry",
    "ci-fit": "bdbv-hub:ci-fit",
    "fit-store": "bdbv-hub:fit-store",
    "fit-store-plan": "bdbv-hub:fit-store-plan",
}

# The named dry runs.
DRY = ["dry:tick", "dry:paper", "dry:submit", "dry:fit-store"]

# Each project's Taskfile and its short name.
PROJECTS = {
    "nowcast-utils": "utils",
    "nowcast-eval": "eval",
    "nowcast-foundation": "foundation",
    "nowcast-workflow": "workflow",
    "bdbv-hub": "hub",
}


def _tasks() -> dict[str, dict]:
    out = subprocess.run(
        ["task", "--list-all", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {t["name"]: t for t in json.loads(out)["tasks"]}


def test_every_task_has_a_home():
    tasks = _tasks()
    missing = sorted(new for new in MOVED.values() if new not in tasks)
    assert missing == []


def test_the_dry_runs_are_named():
    tasks = _tasks()
    assert [name for name in DRY if name not in tasks] == []


def test_the_root_keeps_only_top_level_tasks():
    tasks = _tasks()
    top = {name for name in tasks if ":" not in name}
    assert top == {new for new in MOVED.values() if ":" not in new}


@pytest.mark.parametrize("project", sorted(PROJECTS))
def test_each_project_has_a_taskfile_by_namespace(project):
    tasks = _tasks()
    assert (ROOT / "taskfiles" / f"{project}.yml").is_file()
    mine = [t for name, t in tasks.items() if name.startswith(f"{project}:")]
    assert mine
    for task in mine:
        assert task["location"]["taskfile"].endswith(f"{project}.yml")


def test_a_project_answers_to_its_short_name():
    for project, short in PROJECTS.items():
        out = subprocess.run(
            ["task", "--summary", f"{short}:" + _first(project)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr


def _first(project: str) -> str:
    names = sorted(n for n in _tasks() if n.startswith(f"{project}:"))
    return names[0].split(":", 1)[1]


def test_contributing_has_the_old_to_new_table():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    for old, new in MOVED.items():
        if old != new:
            assert f"| `{old}` | `{new}` |" in text, old
