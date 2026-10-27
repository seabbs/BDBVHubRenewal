"""Check every prerequisite of a live submission that can be read.

Reads the repository's settings with `gh` and prints what is missing and
how to fix it. Nothing is changed. It needs `gh` logged in as someone who
can read the repository's settings (Sam, as admin): environments,
secrets and workflow permissions are admin-only reads. A setting that
cannot be read is reported as `unknown`, not as missing.

Two prerequisites cannot be checked from here: the token's scopes (the
secret's value is never readable) and the hub organisers' approval of
the bot's first pull request. `render` lists them at the end.

The model and hub id follow the same rule as `submit.yml`: the
repository variables `HUB_MODEL` and `HUB_MODEL_ID` when set, otherwise
the workflow's defaults.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import Runner, subprocess_runner

ENVIRONMENT = "hub-submission"
SECRET = "HUB_SUBMIT_TOKEN"
WORKFLOW = "submit.yml"
# The token is created with a 90-day expiry (docs/submission/setup.md).
TOKEN_DAYS = 90
TOKEN_WARN_DAYS = 80
# The schema allows 16 characters; the hub READMEs ask for fewer than 16.
ABBR = re.compile(r"^[a-zA-Z0-9_+]{1,15}$")
BOT = "seabbs-bot"

LABEL = {
    "ok": "OK",
    "warn": "WARN",
    "missing": "MISSING",
    "unknown": "UNKNOWN",
}

MANUAL = (
    "The token has the classic scopes `public_repo` and `workflow` "
    "(GitHub never shows a secret's value).",
    "The hub organisers know weekly bot pull requests will follow, and "
    "approve the first run of the hub's validation workflow on it.",
)


@dataclass(frozen=True)
class Check:
    """The verdict on one prerequisite."""

    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class Settings:
    """What to check, with the workflow's defaults."""

    repo: str
    hub: str
    fork_default: str
    model_default: str
    model_id_default: str
    metadata_dir: Path
    model_choice: Path
    workflow_file: Path
    now: dt.datetime


def _api(runner: Runner, path: str) -> tuple[bool, Any]:
    """`gh api <path>`: (True, json) or (False, error message)."""
    res = runner(["gh", "api", path], check=False)
    try:
        data = json.loads(res.stdout or "null")
    except json.JSONDecodeError:
        data = None
    if res.returncode != 0:
        message = data.get("message", "") if isinstance(data, dict) else ""
        return False, message or res.stderr.strip() or "request failed"
    return True, data


def _not_found(message: object) -> bool:
    return "Not Found" in str(message)


def _unknown(name: str, message: object) -> Check:
    return Check(
        name,
        "unknown",
        f"could not read the setting ({message})",
        "Run the preflight as a repository admin (`gh auth login` as seabbs).",
    )


def check_gh(runner: Runner) -> Check:
    """Whether `gh` is logged in."""
    res = runner(["gh", "auth", "status"], check=False)
    if res.returncode == 0:
        return Check("gh login", "ok", "gh is logged in")
    return Check(
        "gh login",
        "missing",
        "gh is not logged in, so no setting can be read",
        "Run `gh auth login` as seabbs.",
    )


def check_hub(runner: Runner, hub: str) -> Check:
    """Whether the hub repository can be reached."""
    res = runner(
        ["git", "ls-remote", f"https://github.com/{hub}.git", "HEAD"],
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
        sha = res.stdout.split()[0][:10]
        return Check("hub reachable", "ok", f"{hub} main is at {sha}")
    return Check(
        "hub reachable",
        "missing",
        f"cannot reach https://github.com/{hub}",
        "Check the network and that the hub repository still exists.",
    )


def check_fork(runner: Runner, fork: str, hub: str) -> list[Check]:
    """The bot's fork exists, is a fork of the hub, and has Actions off."""
    fix = (
        f"As {BOT}: `gh repo fork {hub} --clone=false`, or set the "
        "repository variable HUB_FORK to the fork."
    )
    ok, data = _api(runner, f"repos/{fork}")
    if not ok:
        return [Check("fork", "missing", f"{fork} not found ({data})", fix)]
    parent = (data.get("parent") or {}).get("full_name", "")
    if not data.get("fork") or parent != hub:
        return [
            Check(
                "fork",
                "missing",
                f"{fork} is not a fork of {hub} (parent: {parent or 'none'})",
                fix,
            )
        ]
    checks = [Check("fork", "ok", f"{fork} is a fork of {hub}")]
    ok, data = _api(runner, f"repos/{fork}/actions/permissions")
    actions_fix = (
        f"As {BOT}: in {fork}, Settings → Actions → General → Disable "
        "actions, so the hub's workflows never run in the fork."
    )
    if not ok:
        checks.append(
            Check(
                "fork actions",
                "unknown",
                f"needs admin on the fork to read ({data})",
                actions_fix,
            )
        )
    elif data.get("enabled"):
        checks.append(
            Check("fork actions", "warn", "Actions are on", actions_fix)
        )
    else:
        checks.append(Check("fork actions", "ok", "Actions are off"))
    return checks


def check_environment(runner: Runner, repo: str) -> Check:
    """`hub-submission` exists, deploys from `main` only, no reviewers."""
    fix = (
        f"In {repo}: Settings → Environments → New environment "
        f"`{ENVIRONMENT}`; Deployment branches: Selected branches, add "
        "`main`; no required reviewers."
    )
    ok, data = _api(runner, f"repos/{repo}/environments/{ENVIRONMENT}")
    if not ok:
        if _not_found(data):
            return Check(
                "environment", "missing", f"no `{ENVIRONMENT}` environment", fix
            )
        return _unknown("environment", data)
    policy = data.get("deployment_branch_policy") or {}
    if not policy.get("custom_branch_policies"):
        return Check(
            "environment",
            "missing",
            "any branch can deploy to it, so any branch could read the token",
            fix,
        )
    ok, branches = _api(
        runner,
        f"repos/{repo}/environments/{ENVIRONMENT}/deployment-branch-policies",
    )
    if not ok:
        return _unknown("environment", branches)
    names = sorted(
        b.get("name", "") for b in branches.get("branch_policies", [])
    )
    if names != ["main"]:
        return Check(
            "environment",
            "missing",
            f"deployment branches are {names or 'none'}, not only `main`",
            fix,
        )
    rules = {r.get("type") for r in data.get("protection_rules", [])}
    if "required_reviewers" in rules or "wait_timer" in rules:
        return Check(
            "environment",
            "warn",
            "has required reviewers or a wait timer, so scheduled runs "
            "stall until someone approves",
            "Remove them: the gate is the approval.",
        )
    return Check("environment", "ok", "deploys from `main` only")


def check_secret(runner: Runner, repo: str, now: dt.datetime) -> list[Check]:
    """The token is an environment secret, not a repository secret."""
    fix = (
        f"Create the token (see Setup), then in {repo}: Settings → "
        f"Environments → `{ENVIRONMENT}` → Add environment secret `{SECRET}`."
    )
    ok, data = _api(runner, f"repos/{repo}/environments/{ENVIRONMENT}/secrets")
    checks: list[Check] = []
    if not ok and not _not_found(data):
        checks.append(_unknown("token secret", data))
    else:
        found = [
            s
            for s in ((data or {}).get("secrets", []) if ok else [])
            if s.get("name") == SECRET
        ]
        if not found:
            checks.append(
                Check(
                    "token secret",
                    "missing",
                    f"no environment secret `{SECRET}` in `{ENVIRONMENT}`",
                    fix,
                )
            )
        else:
            updated = dt.datetime.fromisoformat(
                str(found[0]["updated_at"]).replace("Z", "+00:00")
            )
            age = (now - updated).days
            detail = (
                f"`{SECRET}` set {updated.date()} ({age} days ago; a "
                f"{TOKEN_DAYS}-day token expires about "
                f"{(updated + dt.timedelta(days=TOKEN_DAYS)).date()})"
            )
            if age >= TOKEN_WARN_DAYS:
                checks.append(
                    Check(
                        "token secret",
                        "warn",
                        detail,
                        "Regenerate the token and update the secret.",
                    )
                )
            else:
                checks.append(Check("token secret", "ok", detail))
    ok, data = _api(runner, f"repos/{repo}/actions/secrets")
    if not ok:
        checks.append(_unknown("token scope", data))
    elif any(s.get("name") == SECRET for s in data.get("secrets", [])):
        checks.append(
            Check(
                "token scope",
                "warn",
                f"`{SECRET}` is also a repository secret, readable by "
                "every workflow",
                f"Delete the repository secret `{SECRET}`; keep the "
                "environment one.",
            )
        )
    else:
        checks.append(
            Check("token scope", "ok", "not a repository-wide secret")
        )
    return checks


def check_workflow_permissions(runner: Runner, repo: str) -> Check:
    """The default `GITHUB_TOKEN` is read-only."""
    ok, data = _api(runner, f"repos/{repo}/actions/permissions/workflow")
    if not ok:
        return _unknown("workflow permissions", data)
    perm = data.get("default_workflow_permissions")
    if perm == "read":
        return Check("workflow permissions", "ok", "default token is read-only")
    return Check(
        "workflow permissions",
        "missing",
        f"default token permission is `{perm}`",
        f"In {repo}: Settings → Actions → General → Workflow permissions → "
        "Read repository contents and packages permissions.",
    )


def check_workflow(runner: Runner, repo: str) -> Check:
    """The submission workflow is enabled."""
    ok, data = _api(runner, f"repos/{repo}/actions/workflows/{WORKFLOW}")
    if not ok:
        return Check(
            "workflow",
            "missing",
            f"`{WORKFLOW}` not found on the default branch ({data})",
            "Merge the submission workflow to `main`.",
        )
    state = data.get("state")
    if state == "active":
        return Check("workflow", "ok", f"`{WORKFLOW}` is active")
    return Check(
        "workflow",
        "missing",
        f"`{WORKFLOW}` is `{state}`",
        f"`gh workflow enable {WORKFLOW} -R {repo}` (GitHub disables "
        "schedules after 60 days without activity).",
    )


def read_variables(runner: Runner, repo: str) -> tuple[dict[str, str], Check]:
    """The repository's Actions variables, and the dry-run mode."""
    ok, data = _api(runner, f"repos/{repo}/actions/variables")
    if not ok:
        return {}, _unknown("mode", data)
    values = {
        str(v["name"]): str(v.get("value", ""))
        for v in data.get("variables", [])
    }
    if values.get("DRY_RUN") == "false":
        return values, Check(
            "mode",
            "ok",
            "live: scheduled runs open hub pull requests",
            "Set DRY_RUN to `true` to pause.",
        )
    return values, Check(
        "mode",
        "warn",
        f"dry run (DRY_RUN is {values.get('DRY_RUN', 'unset')})",
        "After a supervised live run, set the repository variable "
        "DRY_RUN to `false`.",
    )


def check_locations(values: dict[str, str]) -> Check:
    """`EXPECTED_LOCATIONS` turns on the coverage check."""
    locs = values.get("EXPECTED_LOCATIONS", "")
    if locs:
        return Check("expected locations", "ok", locs)
    return Check(
        "expected locations",
        "warn",
        "unset, so a file missing a province still passes the checks",
        "Set the repository variable EXPECTED_LOCATIONS, for example "
        "`CD,CD-IT,CD-NK,CD-HU,CD-TO`.",
    )


def _top_level(text: str) -> dict[str, str]:
    """Top-level `key: value` scalars of a YAML file (no dependencies)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]+):\s*(.*?)\s*$", line)
        if m and m.group(2):
            out[m.group(1)] = m.group(2).strip("'\"")
    return out


def check_metadata(metadata_dir: Path, model_id: str) -> Check:
    """`<model_id>.yml` exists and its abbreviations make that id."""
    path = metadata_dir / f"{model_id}.yml"
    fix = "Write it with `task metadata` and check it with `task validate-hub`."
    if not path.exists():
        return Check("metadata", "missing", f"no {path}", fix)
    meta = _top_level(path.read_text(encoding="utf-8"))
    team, model = meta.get("team_abbr", ""), meta.get("model_abbr", "")
    for part in (team, model):
        if not ABBR.match(part):
            return Check(
                "metadata",
                "missing",
                f"{path}: `{part}` must match ^[a-zA-Z0-9_+]+$ with fewer "
                "than 16 characters",
                fix,
            )
    if f"{team}-{model}" != model_id:
        return Check(
            "metadata",
            "missing",
            f"{path} describes `{team}-{model}`, not `{model_id}`",
            fix,
        )
    return Check("metadata", "ok", f"{path} names `{model_id}`")


def latest_choice(path: Path) -> str | None:
    """The model of the newest entry in `model-choice.yml`."""
    if not path.exists():
        return None
    entries: list[tuple[str, str]] = []
    date = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^\s*-\s*date:\s*(\S+)", line):
            date = m.group(1)
        elif (m := re.match(r"^\s+model:\s*(\S+)", line)) and date:
            entries.append((date, m.group(1)))
    return max(entries)[1] if entries else None


def check_model_choice(path: Path, model: str) -> Check:
    """The model the workflow submits is the one last chosen."""
    chosen = latest_choice(path)
    if chosen is None:
        return Check(
            "model choice",
            "missing",
            f"no choice recorded in {path}",
            "Commit the choice to results/model-choice.yml before held-out "
            "scoring.",
        )
    if chosen == model:
        return Check("model choice", "ok", f"`{model}`, as last chosen")
    return Check(
        "model choice",
        "warn",
        f"the workflow submits `{model}` but {path.name} last chose `{chosen}`",
        "Review the held-out results, then add a dated entry to "
        "results/model-choice.yml, or set HUB_MODEL to the chosen model.",
    )


def check_registered(runner: Runner, hub: str, model_id: str) -> Check:
    """Whether the hub already has our metadata."""
    ok, data = _api(
        runner, f"repos/{hub}/contents/model-metadata/{model_id}.yml"
    )
    if ok:
        return Check("hub model id", "ok", f"`{model_id}` is registered")
    if _not_found(data):
        return Check(
            "hub model id",
            "ok",
            f"`{model_id}` is not registered yet; the first pull request "
            "adds its metadata",
        )
    return _unknown("hub model id", data)


def check_cron_owner(runner: Runner, workflow_file: Path) -> Check:
    """Who last edited the schedule, and so gets failure emails."""
    res = runner(
        [
            "git",
            "log",
            "-n",
            "1",
            "--format=%an%x09%ae",
            "-G",
            "cron:",
            "--",
            str(workflow_file),
        ],
        check=False,
    )
    who = res.stdout.strip()
    if res.returncode != 0 or not who:
        return _unknown("cron owner", res.stderr.strip() or "no history")
    name, _, email = who.partition("\t")
    fix = (
        f"Make a commit to the `cron` lines of {workflow_file} as seabbs "
        "(`/commit --as-me`) and push it to `main`."
    )
    if name == BOT or email == "signin@samabbott.co.uk":
        return Check(
            "cron owner",
            "warn",
            f"{name} last edited the schedule, so scheduled-run failures "
            "email the bot",
            fix,
        )
    return Check("cron owner", "ok", f"{name} gets scheduled-run failures")


def run_checks(runner: Runner, s: Settings) -> list[Check]:
    """Every check, in the order of the setup checklist."""
    checks = [check_gh(runner), check_hub(runner, s.hub)]
    values, mode = read_variables(runner, s.repo)
    fork = values.get("HUB_FORK") or s.fork_default
    model = values.get("HUB_MODEL") or s.model_default
    model_id = values.get("HUB_MODEL_ID") or s.model_id_default
    checks += check_fork(runner, fork, s.hub)
    checks.append(check_environment(runner, s.repo))
    checks += check_secret(runner, s.repo, s.now)
    checks.append(check_workflow_permissions(runner, s.repo))
    checks.append(check_workflow(runner, s.repo))
    checks.append(check_cron_owner(runner, s.workflow_file))
    checks.append(check_locations(values))
    checks.append(check_metadata(s.metadata_dir, model_id))
    checks.append(check_registered(runner, s.hub, model_id))
    checks.append(check_model_choice(s.model_choice, model))
    checks.append(mode)
    return checks


def render(checks: Sequence[Check]) -> str:
    """A plain-text report: one line per check, then what to do."""
    width = max(len(c.name) for c in checks)
    lines = [
        f"{LABEL[c.status]:<8} {c.name:<{width}}  {c.detail}" for c in checks
    ]
    todo = [c for c in checks if c.status != "ok" and c.fix]
    if todo:
        lines += ["", "To do:"]
        lines += [f"- {c.name}: {c.fix}" for c in todo]
    lines += ["", "Cannot be checked from here:"]
    lines += [f"- {m}" for m in MANUAL]
    counts = {k: sum(c.status == k for c in checks) for k in LABEL}
    lines += [
        "",
        f"{counts['missing']} missing, {counts['warn']} warnings, "
        f"{counts['unknown']} unknown, {counts['ok']} ok.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report; exit 1 if anything is missing."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="seabbs/BDBVHubRenewal")
    p.add_argument("--hub", default="InsightNet-US/BDBV-Modeling-Hub")
    p.add_argument("--fork", default="seabbs-bot/BDBV-Modeling-Hub")
    p.add_argument("--model", default="renewal")
    p.add_argument("--model-id", default="epiforecasts-seabbsrenewal")
    p.add_argument(
        "--metadata-dir", type=Path, default=Path("hub/model-metadata")
    )
    p.add_argument(
        "--model-choice", type=Path, default=Path("results/model-choice.yml")
    )
    p.add_argument(
        "--workflow-file",
        type=Path,
        default=Path(".github/workflows") / WORKFLOW,
    )
    p.add_argument("--now", default=None, help="ISO 8601 time (for tests).")
    args = p.parse_args(argv)
    now = (
        dt.datetime.fromisoformat(args.now)
        if args.now
        else dt.datetime.now(dt.UTC)
    )
    settings = Settings(
        repo=args.repo,
        hub=args.hub,
        fork_default=args.fork,
        model_default=args.model,
        model_id_default=args.model_id,
        metadata_dir=args.metadata_dir,
        model_choice=args.model_choice,
        workflow_file=args.workflow_file,
        now=now,
    )
    checks = run_checks(subprocess_runner, settings)
    sys.stdout.write(render(checks))
    return 1 if any(c.status == "missing" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
