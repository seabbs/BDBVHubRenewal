"""The fallback chain and the thresholds, read from `policy.toml`.

The chain is an ordered list of candidates. The first candidate that
clears every check is submitted:

1. `main`: the main model with the default sampler settings.
2. `retry-1`: one refit of the main model with a longer warm-up, more
   chains and jittered inits, only when `main` is past the convergence
   retry thresholds or was ruled out (`retry_budget`, 1 by default).
3. `fallback`: the same model code with `--config fallback` (no spatial
   coupling, delays and missingness fixed from the data).
4. `persistence`: the persistence baseline, built from the data alone.

Stage `primary` candidates fit in parallel straight away. Stage `retry`
candidates fit only when the primary `main` fails. Stage `baseline`
candidates need no fit.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import set_outputs
from .fit_key import fit_key, source_hash

KINDS = ("main", "retry", "fallback", "persistence")
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
class Policy:
    """The parsed policy file."""

    candidates: list[Candidate]
    retry_budget: int
    cache_inputs: list[str]
    convergence: dict[str, dict[str, float]]
    checks: dict[str, Any]
    ppc: dict[str, Any] = field(default_factory=dict)

    def chain(self) -> list[Candidate]:
        """Candidates in submission order, with retries cut to budget."""
        out: list[Candidate] = []
        retries = 0
        for c in self.candidates:
            if c.kind == "retry":
                retries += 1
                if retries > self.retry_budget:
                    continue
            out.append(c)
        return out

    def stage(self, name: str) -> list[Candidate]:
        """Candidates fitted in one stage."""
        return [c for c in self.chain() if c.stage == name]

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
    )


def matrix(
    policy: Policy, stage: str, root: Path, hub_ref: str
) -> list[dict[str, str]]:
    """Matrix `include` entries for one stage, each with its cache key."""
    src = source_hash(root, policy.cache_inputs) if policy.cache_inputs else ""
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "config": c.config,
            "label": c.label,
            "sampler": json.dumps(c.sampler, sort_keys=True),
            "key": fit_key(src, hub_ref, c.spec()),
        }
        for c in policy.stage(stage)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Print the matrix for a stage and set it as a step output."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--root", type=Path, default=Path())
    p.add_argument("--hub-ref", required=True)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args(argv)
    entries = matrix(
        load_policy(args.policy), args.stage, args.root, args.hub_ref
    )
    if args.only is not None:
        entries = [e for e in entries if e["id"] in args.only]
    text = json.dumps({"include": entries})
    set_outputs({"matrix": text, "count": len(entries)})
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
