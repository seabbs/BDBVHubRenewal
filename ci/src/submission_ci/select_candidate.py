"""Walk the fallback chain and pick the candidate to submit.

Each candidate directory (`<candidates>/<id>/`) holds what its matrix
job produced: `fit-summary.json` and `ppc-summary.json` (none for
persistence), `checks.json`, `run.json` and the hub file under
`submission/`. Problems are handled rather than failed on, so the chain
rarely moves past the main model. A candidate is ruled out only when
its fit is clearly broken, or its file would be invalid on the hub
(forecast format checks, then the hub's own rules through the
`--validate-cmd` hook, hubValidations in CI). Posterior predictive
checks and plausibility checks are reported, never gated.

A main model past the convergence retry thresholds is refitted once
(`retry-1`). The refit is preferred if it is not clearly broken;
otherwise the original is kept if it is not. Either way the PR says so
under `Handled:`. If nothing is left the round is skipped.

`triage` runs after the primary stage and says whether the refit is
needed. `select` writes `provenance.json` and `provenance.md` and copies
the chosen files to `<out>/selected/`.
"""

from __future__ import annotations

import argparse
import contextlib
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
from .policy import (
    Candidate,
    Policy,
    add_deadline_args,
    deadline_minutes,
    load_policy,
    retry_fits,
)

Validator = Callable[[Path, Path], str]  # (submission, log) -> status


@dataclass
class Attempt:
    """What happened to one candidate."""

    id: str
    kind: str
    label: str
    fit: str = "n/a"
    ppc: str = "n/a"
    checks: str = "n/a"
    validation: str = "not run"
    status: str = "not run"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    handled: list[str] = field(default_factory=list)
    stand_in: str = ""
    retry: bool = False

    @property
    def name(self) -> str:
        """The label, marked when the candidate runs a stand-in model."""
        return f"{self.label} (stand-in)" if self.stand_in else self.label


def _submission(cdir: Path) -> Path | None:
    files = sorted((cdir / "submission").glob("**/*.csv"))
    return files[0] if files else None


def stand_in(cdir: Path) -> str:
    """What a candidate stands in with, from `run.json` or diagnostics.

    The main CLI writes `"stand_in": "<text>"` when a config runs a
    stand-in model, such as persistence until the renewal model is
    integrated. Empty when the candidate is the real thing.
    """
    for name in ("run.json", "diagnostics.json"):
        path = cdir / name
        try:
            value = read_json(path).get("stand_in") if path.exists() else None
        except (ValueError, AttributeError):
            value = None
        if value:
            return str(value)
    return ""


def assess(c: Candidate, cdir: Path) -> Attempt:
    """Combine a candidate's fit gate and forecast checks."""
    a = Attempt(c.id, c.kind, c.label)
    if not cdir.is_dir():
        a.status = "missing"
        a.reasons.append("no output (the job did not run or did not finish)")
        return a
    a.stand_in = stand_in(cdir)
    statuses = []
    fit_path = cdir / "fit-summary.json"
    if c.kind == "persistence" and not fit_path.exists():
        a.fit = "skip"
    elif fit_path.exists():
        fit = read_json(fit_path)
        a.fit = fit["status"]
        a.retry = bool(fit.get("retry"))
        a.reasons += [f"convergence: {m}" for m in fit.get("failures", [])]
        a.warnings += [f"convergence: {m}" for m in fit.get("warnings", [])]
    else:
        a.fit = "error"
        a.reasons.append("convergence: no diagnostics were written")
    statuses.append(a.fit)
    ppc_path = cdir / "ppc-summary.json"
    if ppc_path.exists():
        ppc = read_json(ppc_path)
        a.ppc = ppc["status"]
        a.warnings += [
            f"ppc: {m}"
            for m in [*ppc.get("failures", []), *ppc.get("warnings", [])]
        ]
    elif c.kind == "persistence" or a.fit == "skip":
        a.ppc = "skip"
    else:
        a.ppc = "warn"
        a.warnings.append("ppc: no posterior predictive check was written")
    # Reported only: the predictive checks never rule a candidate out.
    statuses.append("warn" if blocks(a.ppc) else a.ppc)
    checks_path = cdir / "checks.json"
    if checks_path.exists():
        checks = read_json(checks_path)
        a.checks = checks["status"]
        for ch in checks["checks"]:
            if blocks(ch["status"]):
                a.reasons.append(f"{ch['name']}: {ch['summary']}")
            elif ch["status"] == "warn":
                a.warnings.append(f"{ch['name']}: {ch['summary']}")
        a.handled += [str(h) for h in checks.get("handled", [])]
    else:
        a.checks = "error"
        a.reasons.append("forecast checks: no result was written")
    statuses.append(a.checks)
    notes = cdir / "submission" / "handled.json"
    if notes.exists():
        with contextlib.suppress(ValueError):
            a.handled = [str(h) for h in read_json(notes)] + a.handled
    if _submission(cdir) is None:
        statuses.append("error")
        a.reasons.append("no submission file was written")
    a.status = worst(statuses)
    return a


def triage(
    policy: Policy, root: Path, minutes_left: float | None = None
) -> bool:
    """Whether the main model gets its one refit.

    It does when its fit is past the retry thresholds, or when it was
    ruled out, and there is still time to refit before the window closes
    (`minutes_left`, None for no limit). Without that time the chain
    moves on to the single-patch fallback, fitted beside the main model.
    """
    main = policy.chain()[0]
    a = assess(main, root / main.id)
    need = a.retry or blocks_status(a.status)
    if need and not retry_fits(policy, minutes_left):
        assert minutes_left is not None
        t = policy.timing
        append_summary(
            f"Not refitted: {int(minutes_left)} minutes left before the "
            f"window closes, and a refit needs about {t.retry_minutes} "
            f"plus {t.reserve_minutes} for the gate."
        )
        return False
    return need


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


def _validate(a: Attempt, cdir: Path, validate: Validator | None) -> bool:
    """Run the hub's checks on an eligible attempt; True if it passes."""
    sub = _submission(cdir)
    assert sub is not None  # assess() checked it
    if validate is None:
        a.validation = "skip"
        return True
    a.validation = validate(sub, cdir / "hub-validation.log")
    if a.validation != "pass":
        a.reasons.append("hub validation failed (see hub-validation.log)")
        a.status = "fail"
        return False
    return True


def _fit_note(a: Attempt) -> str:
    return next(
        (w.removeprefix("convergence: ") for w in a.warnings
         if "retry threshold" in w),
        "convergence past the retry thresholds",
    )  # fmt: skip


def select(
    policy: Policy,
    root: Path,
    validate: Validator | None,
    retried: bool,
) -> tuple[Attempt | None, list[Attempt]]:
    """Return the chosen attempt (or None) and every attempt."""
    attempts: list[Attempt] = []
    chosen: Attempt | None = None
    deferred: Attempt | None = None  # a main fit waiting on its refit
    for c in policy.chain():
        if chosen is not None:
            attempts.append(Attempt(c.id, c.kind, c.label, status="not needed"))
            continue
        if c.kind == "retry" and not retried:
            attempts.append(Attempt(c.id, c.kind, c.label, status="not needed"))
            continue
        a = assess(c, root / c.id)
        attempts.append(a)
        if c.kind == "retry" and deferred is not None:
            why = _fit_note(deferred)
            if not blocks_status(a.status) and _validate(
                a, root / c.id, validate
            ):
                a.handled.insert(0, f"main model refitted once ({why})")
                chosen = a
                continue
            if _validate(deferred, root / deferred.id, validate):
                deferred.handled.insert(
                    0,
                    f"main model kept after one refit did not help ({why})",
                )
                chosen = deferred
                deferred.status = "warn"
            deferred = None
            continue
        if blocks_status(a.status):
            continue
        if c.kind == "main" and a.retry and retried:
            deferred = a
            continue
        if not _validate(a, root / c.id, validate):
            continue
        if a.retry:
            a.handled.insert(0, f"accepted without a refit ({_fit_note(a)})")
        chosen = a
    if (
        chosen is None
        and deferred is not None
        and _validate(deferred, root / deferred.id, validate)
    ):
        deferred.handled.insert(
            0, f"main model kept; its refit did not run ({_fit_note(deferred)})"
        )
        chosen = deferred
    return chosen, attempts


def _reason(chosen: Attempt | None, attempts: Sequence[Attempt]) -> str:
    """One sentence on why this candidate produced the submission."""
    before = [a for a in attempts if a.reasons and a is not chosen]
    why = "; ".join(f"{a.name} was ruled out ({a.reasons[0]})" for a in before)
    if chosen is None:
        return "Every candidate was ruled out: " + (why or "none ran") + "."
    if not before:
        if chosen.warnings:
            return f"The {chosen.name} cleared every check, with warnings."
        return f"The {chosen.name} cleared every check."
    return f"{why[0].upper()}{why[1:]}. The {chosen.name} cleared the checks."


def stand_in_line(text: str) -> str:
    """The sentence that states a stand-in, in bold, ending in a stop."""
    text = text.strip()
    return f"**Stand-in:** {text}{'' if text.endswith('.') else '.'}"


def provenance_markdown(prov: dict[str, Any]) -> str:
    """The selection as a job summary section."""
    head = (
        f"## Produced by: {prov['produced_by']}\n\n{prov['reason']}\n"
        if prov["selected"]
        else f"## Round skipped\n\n{prov['reason']}\n"
    )
    if prov.get("stand_in"):
        head += f"\n{stand_in_line(prov['stand_in'])}\n"
    rows = [
        [
            f"`{a['id']}`",
            a["label"] + (" (stand-in)" if a.get("stand_in") else ""),
            STATUS_LABEL.get(a["fit"], a["fit"]),
            STATUS_LABEL.get(a.get("ppc", "n/a"), a.get("ppc", "n/a")),
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
            "predictive checks",
            "forecast checks",
            "hub validation",
            "outcome",
        ],
        rows,
    )
    return f"{head}\n{table}\n"


def _shadows(policy: Policy, root: Path) -> list[dict[str, Any]]:
    """Fit and PPC summaries of the shadow candidates, never submitted."""
    out = []
    for c in policy.candidates:
        if c.kind != "shadow":
            continue
        cdir = root / c.id
        fit, ppc = cdir / "fit-summary.json", cdir / "ppc-summary.json"
        out.append(
            {
                "id": c.id,
                "label": c.label,
                "fit": read_json(fit) if fit.exists() else None,
                "ppc": read_json(ppc) if ppc.exists() else None,
            }
        )
    return out


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
        "produced_by": chosen.name if chosen else "nothing (round skipped)",
        "stand_in": chosen.stand_in if chosen else "",
        "status": (
            ("warn" if chosen.warnings or chosen.status == "warn" else "pass")
            if chosen
            else "skip"
        ),
        "reason": _reason(chosen, attempts),
        "warnings": chosen.warnings if chosen else [],
        "handled": chosen.handled if chosen else [],
        "attempts": [asdict(a) for a in attempts],
        "shadows": _shadows(policy, root),
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
        s.add_argument("--model", default="", help="The chosen model")
    add_deadline_args(sub.choices["triage"])
    sel = sub.choices["select"]
    sel.add_argument("--out", type=Path, required=True)
    sel.add_argument("--validate-cmd", default="")
    sel.add_argument("--retried", action="store_true")
    sel.add_argument(
        "--only",
        default="",
        help="Submit this one candidate alone (a hub model of its own)",
    )
    args = p.parse_args(argv)
    policy = load_policy(args.policy).for_model(args.model)
    if getattr(args, "only", ""):
        policy = policy.only(args.only)
    if args.cmd == "triage":
        need = triage(policy, args.candidates, deadline_minutes(args))
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
