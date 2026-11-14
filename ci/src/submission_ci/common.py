"""Shared helpers: commands, GitHub Actions outputs and statuses.

Every module that calls `gh` or `git` takes a `Runner`, so tests swap
in a fake that records the calls and returns scripted output.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Status levels, in order of severity. `skip` means a check did not
# apply (for example convergence for a model with no sampler). It gates
# like `pass`. `error` means a check could not reach a verdict, and
# gates like `fail`: an unjudged run is not a passing one.
STATUS_ORDER = {"pass": 0, "skip": 0, "warn": 1, "fail": 2, "error": 3}

STATUS_LABEL = {
    "pass": "PASS",
    "skip": "N/A",
    "warn": "WARN",
    "fail": "FAIL",
    "error": "ERROR",
}


def worst(statuses: Iterable[str]) -> str:
    """Return the most severe status, or `pass` when there are none."""
    out = "pass"
    for s in statuses:
        if STATUS_ORDER[s] > STATUS_ORDER[out]:
            out = s
    return out


def blocks(status: str) -> bool:
    """Whether a status stops the submission."""
    return STATUS_ORDER[status] >= STATUS_ORDER["fail"]


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one external command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandError(RuntimeError):
    """A command exited non-zero."""

    def __init__(self, args: Sequence[str], result: CommandResult) -> None:
        super().__init__(
            f"`{' '.join(args)}` exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )
        self.result = result


class Runner(Protocol):
    """Run a command and return its result."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run `args`."""
        ...


def subprocess_runner(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> CommandResult:
    """Run a command with `subprocess`, capturing its output."""
    proc = subprocess.run(
        list(args), cwd=cwd, capture_output=True, text=True, check=False
    )
    result = CommandResult(proc.returncode, proc.stdout, proc.stderr)
    if check and proc.returncode != 0:
        raise CommandError(args, result)
    return result


def set_outputs(values: Mapping[str, object]) -> None:
    """Write step outputs to `$GITHUB_OUTPUT` when it is set."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as fh:
        for key, value in values.items():
            text = str(value).lower() if isinstance(value, bool) else value
            text = "" if text is None else str(text)
            if "\n" in text:
                fh.write(f"{key}<<__EOF__\n{text}\n__EOF__\n")
            else:
                fh.write(f"{key}={text}\n")


def append_summary(markdown: str) -> None:
    """Append markdown to the job summary when running in Actions."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(markdown.rstrip() + "\n\n")


def run_url() -> str:
    """Link to the current Actions run, or an empty string locally."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run = os.environ.get("GITHUB_RUN_ID", "")
    if not (repo and run):
        return ""
    return f"{server}/{repo}/actions/runs/{run}"


def read_json(path: Path | str) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, data: Any) -> None:
    """Write a JSON file, creating its directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_text(path: Path | str, text: str) -> None:
    """Write a text file, creating its directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a number for a report, with `n/a` for missing values."""
    if (
        value is None
        or value != value
        or value
        in (
            float("inf"),
            float("-inf"),
        )
    ):
        return "n/a"
    if float(value).is_integer() and abs(value) >= 10:
        return str(int(value))
    return f"{value:.{digits}g}"


def fmt_rhat(value: float | None) -> str:
    """Format an R-hat to three decimals, so 1.004 is not shown as 1."""
    if value is None or value != value:
        return "n/a"
    return f"{value:.3f}"


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render a GitHub markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)
