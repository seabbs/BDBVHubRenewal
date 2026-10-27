"""Walk the fallback chain and pick the candidate to submit.

Each candidate directory (`<candidates>/<id>/`) holds what its matrix
job produced: `fit-summary.json` (none for persistence), `checks.json`,
`run.json` and the hub file under `submission/`. A candidate is
eligible when neither its convergence gate nor its forecast checks
fail. The first eligible candidate in chain order is then validated with
the hub's own rules (the `--validate-cmd` hook, hubValidations in CI).
If that fails too, the chain moves on. If nothing is left the round is
skipped.

`triage` runs after the primary stage and says whether retries of the
main model are needed. `select` writes `provenance.json` and
`provenance.md` and copies the chosen files to `<out>/selected/`.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    STATUS_LABEL,
    append_summary,
    blocks,
    markdown_table,
    read_json,
    set_outputs,
    subprocess_runner,
    worst,
    write_json,
    write_text,
)
from .policy import Candidate, Policy, load_policy

Validator = Callable[[Path, Path], str]  # (submission, log) -> status


@dataclass
class Attempt:
    """What happened to one candidate."""

    id: str
    kind: str
    label: str
    fit: str = "n/a"
    checks: str = "n/a"
    validation: str = "not run"
    status: str = "not run"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _submission(cdir: Path) -> Path | None:
    files = sorted((cdir / "submission").glob("**/*.csv"))
    return files[0] if files else None


def assess(c: Candidate, cdir: Path) -> Attempt:
    """Combine a candidate's fit gate and forecast checks."""
    a = Attempt(c.id, c.kind, c.label)
    if not cdir.is_dir():
        a.status = "missing"
        a.reasons.append("no output (the job did not run or did not finish)")
        return a
    statuses = []
    fit_path = cdir / "fit-summary.json"
    if c.kind == "persistence" and not fit_path.exists():
        a.fit = "skip"
    elif fit_path.exists():
        fit = read_json(fit_path)
        a.fit = fit["status"]
        a.reasons += [f"convergence: {m}" for m in fit.get("failures", [])]
        a.warnings += [f"convergence: {m}" for m in fit.get("warnings", [])]
    else:
        a.fit = "error"
        a.reasons.append("convergence: no diagnostics were written")
    statuses.append(a.fit)
    checks_path = cdir / "checks.json"
    if checks_path.exists():
        checks = read_json(checks_path)
        a.checks = checks["status"]
        for ch in checks["checks"]:
            if blocks(ch["status"]):
                a.reasons.append(f"{ch['name']}: {ch['summary']}")
            elif ch["status"] == "warn":
                a.warnings.append(f"{ch['name']}: {ch['summary']}")
    else:
        a.checks = "error"
        a.reasons.append("forecast checks: no result was written")
    statuses.append(a.checks)
    if _submission(cdir) is None:
        statuses.append("error")
        a.reasons.append("no submission file was written")
    a.status = worst(statuses)
    return a


def triage(policy: Policy, root: Path) -> bool:
    """Whether the main model needs retries."""
    main = policy.chain()[0]
    return blocks_status(assess(main, root / main.id).status)


def blocks_status(status: str) -> bool:
    """Whether an attempt status rules the candidate out."""
    return status in ("missing", "not run") or blocks(status)


def command_validator(template: str) -> Validator:
    """Validator running a shell command with `{submission}` filled in.

    The command's output goes to the log; exit 0 is a pass.
    """

    def run(submission: Path, log: Path) -> str:
        cmd = shlex.split(
            template.format(submission=shlex.quote(str(submission)))
        )
        res = subprocess_runner(cmd, check=False)
        write_text(log, res.stdout + res.stderr)
        return "pass" if res.returncode == 0 else "fail"

    return run


def select(
    policy: Policy,
    root: Path,
    validate: Validator | None,
    retried: bool,
) -> tuple[Attempt | None, list[Attempt]]:
    """Return the chosen attempt (or None) and every attempt."""
    attempts: list[Attempt] = []
    chosen: Attempt | None = None
    for c in policy.chain():
        if chosen is not None:
            attempts.append(Attempt(c.id, c.kind, c.label, status="not needed"))
            continue
        if c.kind == "retry" and not retried:
            attempts.append(Attempt(c.id, c.kind, c.label, status="not needed"))
            continue
        a = assess(c, root / c.id)
        attempts.append(a)
        if blocks_status(a.status):
            continue
        sub = _submission(root / c.id)
        assert sub is not None  # assess() checked it
        if validate is None:
            a.validation = "skip"
        else:
            a.validation = validate(sub, root / c.id / "hub-validation.log")
            if a.validation != "pass":
                a.reasons.append(
                    "hub validation failed (see hub-validation.log)"
                )
                a.status = "fail"
                continue
        chosen = a
    return chosen, attempts


def _reason(chosen: Attempt | None, attempts: Sequence[Attempt]) -> str:
    """One sentence on why this candidate produced the submission."""
    before = [a for a in attempts if a.reasons and a is not chosen]
    why = "; ".join(f"{a.label} was ruled out ({a.reasons[0]})" for a in before)
    if chosen is None:
        return "Every candidate was ruled out: " + (why or "none ran") + "."
    if not before:
        if chosen.warnings:
            return f"The {chosen.label} cleared every check, with warnings."
        return f"The {chosen.label} cleared every check."
    return f"{why[0].upper()}{why[1:]}. The {chosen.label} cleared the checks."


def provenance_markdown(prov: dict[str, Any]) -> str:
    """The selection as a job summary section."""
    head = (
        f"## Produced by: {prov['produced_by']}\n\n{prov['reason']}\n"
        if prov["selected"]
        else f"## Round skipped\n\n{prov['reason']}\n"
    )
    rows = [
        [
            f"`{a['id']}`",
            a["label"],
            STATUS_LABEL.get(a["fit"], a["fit"]),
            STATUS_LABEL.get(a["checks"], a["checks"]),
            STATUS_LABEL.get(a["validation"], a["validation"]),
            STATUS_LABEL.get(a["status"], a["status"]),
        ]
        for a in prov["attempts"]
    ]
    table = markdown_table(
        [
            "step",
            "candidate",
            "convergence",
            "forecast checks",
            "hub validation",
            "outcome",
        ],
        rows,
    )
    return f"{head}\n{table}\n"


def run_select(
    policy: Policy,
    root: Path,
    out: Path,
    validate: Validator | None,
    retried: bool,
) -> dict[str, Any]:
    """Select, write provenance and copy the chosen files."""
    chosen, attempts = select(policy, root, validate, retried)
    prov: dict[str, Any] = {
        "selected": chosen.id if chosen else None,
        "kind": chosen.kind if chosen else None,
        "produced_by": chosen.label if chosen else "nothing (round skipped)",
        "status": (
            ("warn" if chosen.warnings or chosen.status == "warn" else "pass")
            if chosen
            else "skip"
        ),
        "reason": _reason(chosen, attempts),
        "warnings": chosen.warnings if chosen else [],
        "attempts": [asdict(a) for a in attempts],
    }
    if chosen:
        cdir = root / chosen.id
        run_json = cdir / "run.json"
        if run_json.exists():
            prov["run"] = read_json(run_json)
        dest = out / "selected"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cdir, dest)
    write_json(out / "provenance.json", prov)
    md = provenance_markdown(prov)
    write_text(out / "provenance.md", md)
    append_summary(md)
    return prov


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point.

    `select` exits 0 when a candidate was chosen and 3 when the round is
    skipped, so the workflow can fail loudly on a skip.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("triage", "select"):
        s = sub.add_parser(name)
        s.add_argument("--policy", type=Path, required=True)
        s.add_argument("--candidates", type=Path, required=True)
    sel = sub.choices["select"]
    sel.add_argument("--out", type=Path, required=True)
    sel.add_argument("--validate-cmd", default="")
    sel.add_argument("--retried", action="store_true")
    args = p.parse_args(argv)
    policy = load_policy(args.policy)
    if args.cmd == "triage":
        need = triage(policy, args.candidates)
        set_outputs({"need_retry": need})
        print(f"need_retry={str(need).lower()}")
        return 0
    validate = (
        command_validator(args.validate_cmd) if args.validate_cmd else None
    )
    prov = run_select(policy, args.candidates, args.out, validate, args.retried)
    set_outputs(
        {
            "selected": prov["selected"] or "",
            "kind": prov["kind"] or "",
            "status": prov["status"],
            "produced_by": prov["produced_by"],
            "skipped": prov["selected"] is None,
        }
    )
    print(f"selected={prov['selected']} ({prov['reason']})")
    return 0 if prov["selected"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
