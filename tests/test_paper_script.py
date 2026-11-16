"""`scripts/paper.sh`: the one command that runs and renders the paper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper.sh"

# Stand-ins that log each call to $CALLS.
STUB = """#!/bin/sh
echo "$(basename "$0") $*" >> "$CALLS"
exit 0
"""

OUTPUTS = (
    "main.html",
    "main.pdf",
    "si.html",
    "si.pdf",
    "main_files/a.js",
    "si_files/b.js",
    "figures/f.svg",
    "figures/f.pdf",
    "figures/shot.png",
    "build/figures/g.png",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tree with the script, stand-in tools and rendered outputs."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "paper.sh")
    for name in OUTPUTS:
        path = tmp_path / "paper" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("uv", "git", "quarto"):
        (bin_dir / tool).write_text(STUB)
        (bin_dir / tool).chmod(0o755)
    # uv also logs the token its steps get; gh hands out one.
    (bin_dir / "uv").write_text(
        STUB.replace("exit 0", 'echo "token=$GITHUB_TOKEN" >> "$CALLS"\nexit 0')
    )
    (bin_dir / "gh").write_text("#!/bin/sh\necho gh-token\n")
    (bin_dir / "gh").chmod(0o755)
    return tmp_path


def _run(
    repo: Path, *args: str, cores: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{repo / 'bin'}:{os.environ['PATH']}",
        "CALLS": str(repo / "calls.log"),
    }
    for name in ("CORES", "GITHUB_TOKEN", "GH_TOKEN"):
        env.pop(name, None)
    if cores is not None:
        env["CORES"] = cores
    return subprocess.run(
        ["bash", str(repo / "scripts" / "paper.sh"), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _calls(repo: Path) -> list[str]:
    return (repo / "calls.log").read_text().splitlines()


def test_development_run_fetches_the_store_then_renders(repo: Path) -> None:
    out = _run(repo)
    assert out.returncode == 0, out.stderr
    calls = _calls(repo)
    fetch = next(i for i, c in enumerate(calls) if c.startswith("git fetch"))
    snake = next(i for i, c in enumerate(calls) if "snakemake" in c)
    assert "refs/heads/fit-store:refs/remotes/origin/fit-store" in calls[fetch]
    assert fetch < snake
    assert calls[snake] == (
        "uv run --group paper --group chronos snakemake -s workflow/Snakefile "
        "render "
        "--config render=development --cores 4 --resources heavy=1 "
        "--rerun-triggers mtime"
    )


def test_paper_target_and_cores_are_passed(repo: Path) -> None:
    out = _run(repo, "paper", cores="2")
    assert out.returncode == 0, out.stderr
    (snake,) = [c for c in _calls(repo) if "snakemake" in c]
    assert "--config render=paper --cores 2 " in snake


def test_an_unknown_target_fails(repo: Path) -> None:
    out = _run(repo, "final")
    assert out.returncode != 0
    assert "development or paper" in out.stderr


def test_the_rendered_paper_is_copied_into_the_docs(repo: Path) -> None:
    assert _run(repo).returncode == 0
    docs = repo / "docs" / "paper"
    for name in ("main.html", "si.pdf", "main_files/a.js", "si_files/b.js"):
        assert (docs / name).is_file()
    assert (docs / "figures" / "f.svg").is_file()
    assert (docs / "figures" / "f.pdf").is_file()
    # Image inputs, such as a screenshot, are figures too.
    assert (docs / "figures" / "shot.png").is_file()
    assert (docs / "build" / "figures" / "g.png").is_file()


def test_pages_and_the_taskfile_run_the_script() -> None:
    pages = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
    steps = [s for job in pages["jobs"].values() for s in job.get("steps", [])]
    (paper,) = [
        s
        for s in steps
        if s.get("name") == "Run the paper workflow and render the paper"
    ]
    assert 'scripts/paper.sh "$TARGET"' in paper["run"]
    # The script path is from the repository root.
    assert "working-directory" not in paper
    assert "snakemake" not in paper["run"]
    tasks = yaml.safe_load((ROOT / "Taskfile.yml").read_text())["tasks"]
    assert any("scripts/paper.sh" in str(c) for c in tasks["paper"]["cmds"])


def test_the_steps_get_a_github_token(repo: Path) -> None:
    # CI passes one; locally it is gh's, so the hub data downloads are not
    # rate limited.
    assert _run(repo).returncode == 0
    assert "token=gh-token" in _calls(repo)
