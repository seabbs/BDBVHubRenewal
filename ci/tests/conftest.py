"""Shared fixtures: a fake command runner and sample-file builders."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from submission_ci.common import CommandError, CommandResult, subprocess_runner

FIXTURES = Path(__file__).parents[1] / "fixtures"
HUB = FIXTURES / "hub"
REF = "2026-11-07"
AS_OF = "2026-10-31"
PROVINCES = {"CD-IT": 200.0, "CD-NK": 80.0, "CD-HU": 15.0}

Handler = Callable[[Sequence[str]], CommandResult]


@dataclass
class FakeRunner:
    """Records commands; answers `gh` from handlers, runs `git` for real."""

    handlers: list[tuple[tuple[str, ...], Handler]] = field(
        default_factory=list
    )
    calls: list[list[str]] = field(default_factory=list)
    real_git: bool = True

    def on(
        self, prefix: Sequence[str], out: str | Handler, code: int = 0
    ) -> None:
        """Answer commands starting with `prefix`."""
        if callable(out):
            self.handlers.append((tuple(prefix), out))
        else:
            self.handlers.append(
                (tuple(prefix), lambda _a: CommandResult(code, out, ""))
            )

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run or answer one command."""
        self.calls.append(list(args))
        for prefix, handler in reversed(self.handlers):
            if tuple(args[: len(prefix)]) == prefix:
                result = handler(args)
                if check and result.returncode != 0:
                    raise CommandError(args, result)
                return result
        if args[0] == "git" and self.real_git:
            return subprocess_runner(args, cwd=cwd, check=check)
        raise AssertionError(f"unexpected command: {args}")

    def gh_calls(self) -> list[list[str]]:
        """The `gh` commands that were made."""
        return [c for c in self.calls if c[0] == "gh"]


def weeks(ref: str = REF, back: int = 7, ahead: int = 3) -> list[str]:
    """Saturdays from `back` weeks before to `ahead` weeks after `ref`."""
    import datetime as dt

    r = dt.date.fromisoformat(ref)
    return [
        (r + dt.timedelta(days=7 * k)).isoformat()
        for k in range(-back, ahead + 1)
    ]


def make_samples(
    *,
    ref: str = REF,
    levels: dict[str, float] | None = None,
    n: int = 100,
    growth: float = 1.0,
    seed: int = 1,
    national: bool = True,
    mutate: Callable[[list[list[Any]]], None] | None = None,
) -> list[list[Any]]:
    """Rows of a plausible submission, before an optional mutation."""
    rng = random.Random(seed)
    levels = levels or PROVINCES
    ws = weeks(ref)
    rows: list[list[Any]] = []
    totals: dict[tuple[int, str], float] = {}
    for loc, level in levels.items():
        for s in range(1, n + 1):
            noise = math.exp(rng.gauss(0, 0.1))
            for k, w in enumerate(ws):
                g = growth ** max(0, k - 7)
                v = float(round(level * noise * g))
                rows.append(
                    [ref, "wk inc confirmed cases", w, loc, "sample", s, v]
                )
                totals[(s, w)] = totals.get((s, w), 0.0) + v
    if national:
        for (s, w), v in totals.items():
            rows.append(
                [ref, "wk inc confirmed cases", w, "CD", "sample", s, v]
            )
    if mutate:
        mutate(rows)
    return rows


def write_samples(
    path: Path, rows: list[list[Any]], header: Sequence[str] | None = None
) -> Path:
    """Write rows to a hub-format CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            header
            or [
                "reference_date",
                "target",
                "target_end_date",
                "location",
                "output_type",
                "output_type_id",
                "value",
            ]
        )
        w.writerows(rows)
    return path


def diagnostics(
    rhat: float = 1.004, ess: float = 900.0, div: int = 0, sampler: str = "nuts"
) -> dict[str, Any]:
    """A `diagnostics.json` payload."""
    return {
        "schema_version": 1,
        "sampler": sampler,
        "num_chains": 4,
        "num_samples": 1000,
        "parameters": [
            {
                "name": "log_rt",
                "index": i,
                "rhat": rhat - 0.001 * i,
                "ess_bulk": ess + 10 * i,
                "ess_tail": ess + 20 * i,
            }
            for i in range(3)
        ]
        + [
            {
                "name": "fixed",
                "index": 0,
                "rhat": None,
                "ess_bulk": None,
                "ess_tail": None,
            }
        ],
        "chains": [
            {
                "chain": c,
                "draws": 1000,
                "divergences": div // 4,
                "step_size": 0.1,
                "tree_depth_hits": 0,
                "mean_accept": 0.9,
                "ebfmi": 0.8,
            }
            for c in range(4)
        ],
    }


def write_json(path: Path, data: Any) -> Path:
    """Write JSON, creating the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def runner() -> FakeRunner:
    """A fresh fake runner."""
    return FakeRunner()
