"""The fallback chain and the thresholds, read from `policy.toml`.

The chain is an ordered list of candidates, and the first that clears
every check is submitted. It runs through the renewal models from the
most to the least complex, then `randomwalknowcast` (random walk on a
nowcast) and `randomwalk` (the random walk baseline, built from the data
alone). There are no refits.

Stage `primary` candidates are fitted models, read from the fit store
(`bdbvhubrenewal fit-store unpack`). Stage `baseline` candidates need no
fit.

The wait for the store's fits, and a fit made in the submission when the
store cannot hold the round, stop `reserve_minutes` before the window
closes for the gate and the pull request, so the chain can always reach
the fallback or random walk in time.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .common import set_outputs
from .window import minutes_left, parse_now

KINDS = ("main", "fallback", "randomwalk", "shadow")
"""A `shadow` candidate is fitted and checked with its stage but is never
part of the chain, so it is never submitted. It lets a variant run weekly
before it is chosen."""
STAGES = ("primary", "baseline")


@dataclass(frozen=True)
class Candidate:
    """One step of the fallback chain."""

    id: str
    kind: str
    config: str
    stage: str
    label: str
    sampler: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    """The model the leg runs; empty for the chosen model."""


@dataclass(frozen=True)
class Timing:
    """Minutes the chain plans with (`[timing]` in `policy.toml`)."""

    fit_cap_minutes: int
    reserve_minutes: int
    min_fit_minutes: int


@dataclass(frozen=True)
class Policy:
    """The parsed policy file."""

    candidates: list[Candidate]
    convergence: dict[str, dict[str, float]]
    checks: dict[str, Any]
    ppc: dict[str, Any]
    timing: Timing

    def chain(self) -> list[Candidate]:
        """Candidates in submission order."""
        return [c for c in self.candidates if c.kind != "shadow"]

    def stage(self, name: str) -> list[Candidate]:
        """Candidates fitted in one stage, shadows included."""
        shadows = [c for c in self.candidates if c.kind == "shadow"]
        return [c for c in [*self.chain(), *shadows] if c.stage == name]

    def for_model(self, model: str, only: bool = False) -> Policy:
        """The chain from the candidate of `model` on.

        Candidates before it are still fitted and checked, as shadows.
        A model with no candidate keeps the
        policy as it is.

        With `only`, the chain is that candidate alone, for a hub model
        that must never be submitted as anything else. The others are
        still built as shadows, as references for the checks, and are
        never selected.
        """
        chain = self.chain()
        names = [c.model or c.id for c in chain]
        if model not in names:
            if only:
                raise ValueError(f"{model!r} has no candidate")
            return self
        start = names.index(model)
        kept = {c.id for c in chain[start : start + 1 if only else None]}
        cands = [
            c if c.id in kept else replace(c, kind="shadow")
            for c in self.candidates
        ]
        return replace(self, candidates=cands)

    def get(self, cid: str) -> Candidate:
        """Look up a candidate by id."""
        for c in self.candidates:
            if c.id == cid:
                return c
        raise KeyError(cid)


def load_policy(path: Path | str) -> Policy:
    """Read and check `policy.toml`."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cands = []
    for c in raw["candidates"]:
        cand = Candidate(
            id=c["id"],
            kind=c["kind"],
            config=c["config"],
            stage=c["stage"],
            label=c["label"],
            sampler=dict(c.get("sampler", {})),
            model=c.get("model", ""),
        )
        if cand.kind not in KINDS:
            raise ValueError(f"candidate {cand.id}: unknown kind {cand.kind}")
        if cand.stage not in STAGES:
            raise ValueError(f"candidate {cand.id}: unknown stage {cand.stage}")
        cands.append(cand)
    ids = [c.id for c in cands]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    if not cands or cands[0].kind != "main":
        raise ValueError("the first candidate must be the main model")
    return Policy(
        candidates=cands,
        convergence=raw["convergence"],
        checks=raw["checks"],
        ppc=raw["ppc"],
        timing=Timing(**{k: int(v) for k, v in raw["timing"].items()}),
    )


def fit_timeout(policy: Policy, minutes_left: float | None) -> int:
    """How long to wait for, or make, a fit: the cap, less the gate's time.

    `minutes_left` is the time to the window's close (None for no
    limit). The timeout never drops below `min_fit_minutes`; a fit that
    cannot finish then fails and the chain moves on.
    """
    t = policy.timing
    if minutes_left is None:
        return t.fit_cap_minutes
    room = int(minutes_left) - t.reserve_minutes
    return max(t.min_fit_minutes, min(t.fit_cap_minutes, room))


def matrix(
    policy: Policy,
    stage: str,
    minutes_left: float | None = None,
) -> list[dict[str, str]]:
    """Matrix `include` entries for one stage, each with its timeout in
    minutes (`fit_timeout`)."""
    timeout = str(fit_timeout(policy, minutes_left))
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "config": c.config,
            "label": c.label,
            "model": c.model,
            "sampler": json.dumps(c.sampler, sort_keys=True),
            "timeout": timeout,
        }
        for c in policy.stage(stage)
    ]


def add_deadline_args(p: argparse.ArgumentParser) -> None:
    """The round, so fits can be timed against its window."""
    p.add_argument("--reference-date", default="")
    p.add_argument("--ignore-window", action="store_true")
    p.add_argument("--now", default=None, help="ISO time (for tests)")


def deadline_minutes(args: argparse.Namespace) -> float | None:
    """Minutes to the window's close from parsed `add_deadline_args`."""
    return minutes_left(
        args.reference_date,
        parse_now(args.now),
        ignore_window=args.ignore_window,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the matrix for a stage and set it as a step output."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument(
        "--model",
        default="",
        help="The chosen model; random walk makes a randomwalk-only chain",
    )
    p.add_argument(
        "--only-model",
        action="store_true",
        help="A chain of --model alone, with the others as shadows",
    )
    add_deadline_args(p)
    args = p.parse_args(argv)
    entries = matrix(
        load_policy(args.policy).for_model(args.model, args.only_model),
        args.stage,
        deadline_minutes(args),
    )
    if args.only is not None:
        entries = [e for e in entries if e["id"] in args.only]
    text = json.dumps({"include": entries})
    set_outputs({"matrix": text, "count": len(entries)})
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
