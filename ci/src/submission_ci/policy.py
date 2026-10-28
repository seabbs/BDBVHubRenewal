"""The fallback chain and the thresholds, read from `policy.toml`.

The chain is an ordered list of candidates. The first candidate that
clears every check is submitted:

1. `main`: the main model with the default sampler settings.
2. `retry-1`: one refit of the main model with a longer warm-up, smaller
   steps and jittered inits, only when `main` is past the convergence
   retry thresholds or was ruled out (`retry_budget`, 1 by default), and
   only when there is time for it before the window closes (`[timing]`).
3. `fallback`: the same model code with `--config fallback` (no spatial
   coupling, delays and missingness fixed from the data).
4. `persistence`: the persistence baseline, built from the data alone.

Stage `primary` candidates fit in parallel straight away. Stage `retry`
candidates fit only when the primary `main` fails. Stage `baseline`
candidates need no fit.

Every fit job's timeout leaves `reserve_minutes` before the window
closes for the gate and the pull request, so the chain can always reach
the fallback or persistence in time.
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
from .fit_key import fit_key, source_hash
from .window import minutes_left, parse_now

KINDS = ("main", "retry", "fallback", "persistence", "shadow")
"""A `shadow` candidate is fitted and checked with its stage but is never
part of the chain, so it is never submitted. It lets a variant run weekly
before it is chosen."""
STAGES = ("primary", "retry", "baseline")


@dataclass(frozen=True)
class Candidate:
    """One step of the fallback chain."""

    id: str
    kind: str
    config: str
    stage: str
    label: str
    sampler: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> dict[str, Any]:
        """What the fit depends on besides source and data."""
        return {"config": self.config, "sampler": self.sampler}


@dataclass(frozen=True)
class Timing:
    """Minutes the chain plans with (`[timing]` in `policy.toml`)."""

    fit_cap_minutes: int = 350
    reserve_minutes: int = 45
    retry_minutes: int = 240
    min_fit_minutes: int = 30


@dataclass(frozen=True)
class Policy:
    """The parsed policy file."""

    candidates: list[Candidate]
    retry_budget: int
    cache_inputs: list[str]
    convergence: dict[str, dict[str, float]]
    checks: dict[str, Any]
    ppc: dict[str, Any] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)

    def chain(self) -> list[Candidate]:
        """Candidates in submission order, with retries cut to budget."""
        out: list[Candidate] = []
        retries = 0
        for c in self.candidates:
            if c.kind == "shadow":
                continue
            if c.kind == "retry":
                retries += 1
                if retries > self.retry_budget:
                    continue
            out.append(c)
        return out

    def stage(self, name: str) -> list[Candidate]:
        """Candidates fitted in one stage, shadows included."""
        shadows = [c for c in self.candidates if c.kind == "shadow"]
        return [c for c in [*self.chain(), *shadows] if c.stage == name]

    def for_model(self, model: str) -> Policy:
        """The policy for the model chosen in the model choice record.

        With persistence chosen, the chain is persistence alone. The
        renewal candidates (main, fallback and any shadows) are still
        fitted and checked as shadows, and the retry is dropped.
        Any other choice keeps the policy as it is.
        """
        if model != "persistence":
            return self
        cands = [
            replace(c, kind="shadow")
            if c.kind in ("main", "fallback", "shadow")
            else c
            for c in self.candidates
            if c.kind != "retry"
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
    for c in raw.get("candidates", []):
        cand = Candidate(
            id=c["id"],
            kind=c["kind"],
            config=c.get("config", c["kind"]),
            stage=c.get("stage", "primary"),
            label=c.get("label", c["id"]),
            sampler=dict(c.get("sampler", {})),
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
    pol = raw.get("policy", {})
    return Policy(
        candidates=cands,
        retry_budget=int(pol.get("retry_budget", 1)),
        cache_inputs=list(raw.get("cache", {}).get("inputs", [])),
        convergence=raw.get("convergence", {}),
        checks=raw.get("checks", {}),
        ppc=raw.get("ppc", {}),
        timing=Timing(**{k: int(v) for k, v in raw.get("timing", {}).items()}),
    )


def fit_timeout(policy: Policy, minutes_left: float | None) -> int:
    """A fit job's timeout: the cap, less whatever the gate needs.

    `minutes_left` is the time to the window's close (None for no
    limit). The timeout never drops below `min_fit_minutes`; a fit that
    cannot finish then fails and the chain moves on.
    """
    t = policy.timing
    if minutes_left is None:
        return t.fit_cap_minutes
    room = int(minutes_left) - t.reserve_minutes
    return max(t.min_fit_minutes, min(t.fit_cap_minutes, room))


def retry_fits(policy: Policy, minutes_left: float | None) -> bool:
    """Whether a refit can finish and still leave the gate its time."""
    if minutes_left is None:
        return True
    t = policy.timing
    return minutes_left >= t.retry_minutes + t.reserve_minutes


def matrix(
    policy: Policy,
    stage: str,
    root: Path,
    hub_ref: str,
    minutes_left: float | None = None,
) -> list[dict[str, str]]:
    """Matrix `include` entries for one stage, each with its cache key
    and its job timeout in minutes."""
    src = source_hash(root, policy.cache_inputs) if policy.cache_inputs else ""
    timeout = str(fit_timeout(policy, minutes_left))
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "config": c.config,
            "label": c.label,
            "sampler": json.dumps(c.sampler, sort_keys=True),
            "key": fit_key(src, hub_ref, c.spec()),
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
    p.add_argument("--root", type=Path, default=Path())
    p.add_argument("--hub-ref", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument(
        "--model",
        default="",
        help="The chosen model; persistence makes a persistence-only chain",
    )
    add_deadline_args(p)
    args = p.parse_args(argv)
    entries = matrix(
        load_policy(args.policy).for_model(args.model),
        args.stage,
        args.root,
        args.hub_ref,
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
