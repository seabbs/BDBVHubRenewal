"""`gh` and `git` are mocked: the preflight only reads settings."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from conftest import FakeRunner

from submission_ci.preflight import Settings, main, render, run_checks

REPO = "seabbs/BDBVHubRenewal"
FORK = "seabbs-bot/BDBV-Modeling-Hub"
HUB = "InsightNet-US/BDBV-Modeling-Hub"
MID = "epiforecasts-seabbsrenewal"
MID2 = "epiforecasts-seabbsbaseline"
ENV = f"repos/{REPO}/environments/hub-submission"
NOW = dt.datetime(2026, 10, 27, 12, tzinfo=dt.UTC)


def api(runner: FakeRunner, path: str, data: object, code: int = 0) -> None:
    prefix = ["gh", "api", path] if path else ["gh", "api"]
    runner.on(prefix, json.dumps(data), code)


def ready_runner(**overrides: object) -> FakeRunner:
    """A runner whose answers describe a fully set-up repository."""
    r = FakeRunner(real_git=False)
    # Anything not answered below does not exist.
    api(r, "", {"message": "Not Found"}, code=1)
    r.on(["gh", "auth", "status"], "Logged in")
    r.on(["git", "ls-remote"], "abc123\tHEAD\n")
    r.on(["git", "log"], "Sam Abbott\tsam.abbott@lshtm.ac.uk\n")
    answers: dict[str, object] = {
        f"repos/{FORK}": {"fork": True, "parent": {"full_name": HUB}},
        f"repos/{FORK}/actions/permissions": {"enabled": False},
        f"repos/{REPO}/environments/hub-submission": {
            "name": "hub-submission",
            "protection_rules": [{"type": "branch_policy"}],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        },
        f"{ENV}/deployment-branch-policies": {
            "branch_policies": [{"name": "main", "type": "branch"}]
        },
        f"repos/{REPO}/environments/hub-submission/secrets": {
            "secrets": [
                {
                    "name": "HUB_SUBMIT_TOKEN",
                    "updated_at": "2026-10-20T00:00:00Z",
                }
            ]
        },
        f"repos/{REPO}/actions/secrets": {"secrets": []},
        f"repos/{REPO}/actions/permissions/workflow": {
            "default_workflow_permissions": "read"
        },
        f"repos/{REPO}/actions/variables": {
            "variables": [
                {"name": "DRY_RUN", "value": "true"},
                {"name": "EXPECTED_LOCATIONS", "value": "CD,CD-IT"},
            ]
        },
        f"repos/{REPO}/actions/workflows/submit.yml": {"state": "active"},
        f"repos/{HUB}/contents/model-metadata/{MID}.yml": {
            "name": f"{MID}.yml"
        },
    }
    answers.update(overrides)
    for path, data in answers.items():
        if data is None:
            api(r, path, {"message": "Not Found"}, code=1)
        else:
            api(r, path, data)
    return r


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / f"{MID}.yml").write_text(
        "team_name: EpiForecasts\nteam_abbr: epiforecasts\n"
        "model_name: Renewal\nmodel_abbr: seabbsrenewal\n"
    )
    (meta / f"{MID2}.yml").write_text(
        "team_abbr: epiforecasts\nmodel_abbr: seabbsbaseline\n"
    )
    choice = tmp_path / "model-choice.yml"
    choice.write_text(
        "choices:\n"
        "  - date: 2026-10-10\n    model: persistence\n"
        "  - date: 2026-10-27\n    model: renewal\n    summary: >-\n"
        "      Chosen.\n"
    )
    return Settings(
        repo=REPO,
        hub=HUB,
        fork_default=FORK,
        model_id_default=MID,
        other_model_ids=(MID2,),
        metadata_dir=meta,
        model_choice=choice,
        workflow_file=Path(".github/workflows/submit.yml"),
        now=NOW,
    )


def statuses(checks: list) -> dict[str, str]:  # type: ignore[type-arg]
    return {c.name: c.status for c in checks}


def test_everything_set_up_passes(settings: Settings) -> None:
    checks = run_checks(ready_runner(), settings)
    bad = {c.name: c.detail for c in checks if c.status != "ok"}
    # A dry run is reported, not a gap: it is the safe default.
    assert set(bad) <= {"mode"}
    assert statuses(checks)["mode"] == "warn"


def test_nothing_set_up_lists_every_gap(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{FORK}": None,
            f"repos/{FORK}/actions/permissions": None,
            f"repos/{REPO}/environments/hub-submission": None,
            f"{ENV}/deployment-branch-policies": None,
            f"repos/{REPO}/environments/hub-submission/secrets": None,
            f"repos/{REPO}/actions/permissions/workflow": {
                "default_workflow_permissions": "write"
            },
            f"repos/{REPO}/actions/variables": {"variables": []},
        }
    )
    s = statuses(run_checks(runner, settings))
    assert s["fork"] == "missing"
    assert s["environment"] == "missing"
    assert s["token secret"] == "missing"
    assert s["workflow permissions"] == "missing"
    assert s["expected locations"] == "warn"


def test_fork_of_wrong_repo_is_missing(settings: Settings) -> None:
    runner = ready_runner(
        **{f"repos/{FORK}": {"fork": True, "parent": {"full_name": "a/b"}}}
    )
    check = {c.name: c for c in run_checks(runner, settings)}["fork"]
    assert check.status == "missing"
    assert "a/b" in check.detail


def test_fork_with_actions_on_warns(settings: Settings) -> None:
    runner = ready_runner(
        **{f"repos/{FORK}/actions/permissions": {"enabled": True}}
    )
    assert statuses(run_checks(runner, settings))["fork actions"] == "warn"


def test_environment_open_to_all_branches_is_missing(
    settings: Settings,
) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/environments/hub-submission": {
                "name": "hub-submission",
                "protection_rules": [],
                "deployment_branch_policy": None,
            }
        }
    )
    check = {c.name: c for c in run_checks(runner, settings)}["environment"]
    assert check.status == "missing"
    assert "main" in check.fix


def test_required_reviewers_warn(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/environments/hub-submission": {
                "name": "hub-submission",
                "protection_rules": [
                    {"type": "branch_policy"},
                    {"type": "required_reviewers"},
                ],
                "deployment_branch_policy": {
                    "protected_branches": False,
                    "custom_branch_policies": True,
                },
            }
        }
    )
    assert statuses(run_checks(runner, settings))["environment"] == "warn"


def test_old_token_warns_before_expiry(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/environments/hub-submission/secrets": {
                "secrets": [
                    {
                        "name": "HUB_SUBMIT_TOKEN",
                        "updated_at": "2026-08-01T00:00:00Z",
                    }
                ]
            }
        }
    )
    check = {c.name: c for c in run_checks(runner, settings)}["token secret"]
    assert check.status == "warn"
    assert "87 days" in check.detail


def test_repository_level_token_warns(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/actions/secrets": {
                "secrets": [{"name": "HUB_SUBMIT_TOKEN"}]
            }
        }
    )
    assert statuses(run_checks(runner, settings))["token scope"] == "warn"


def test_live_mode_is_ok(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/actions/variables": {
                "variables": [
                    {"name": "DRY_RUN", "value": "false"},
                    {"name": "EXPECTED_LOCATIONS", "value": "CD"},
                ]
            }
        }
    )
    check = {c.name: c for c in run_checks(runner, settings)}["mode"]
    assert check.status == "ok"
    assert "live" in check.detail


def test_variables_override_model_and_id(settings: Settings) -> None:
    runner = ready_runner(
        **{
            f"repos/{REPO}/actions/variables": {
                "variables": [
                    {"name": "HUB_MODEL", "value": "persistence"},
                    {"name": "HUB_MODEL_ID", "value": "epiforecasts-other"},
                    {"name": "EXPECTED_LOCATIONS", "value": "CD"},
                ]
            }
        }
    )
    checks = {c.name: c for c in run_checks(runner, settings)}
    assert "epiforecasts-other" in checks["metadata"].detail
    assert checks["metadata"].status == "missing"
    assert checks["model choice"].status == "warn"


def test_metadata_name_mismatch_is_missing(settings: Settings) -> None:
    (settings.metadata_dir / f"{MID}.yml").write_text(
        "team_abbr: epiforecasts\nmodel_abbr: hubrenewal\n"
    )
    check = {c.name: c for c in run_checks(ready_runner(), settings)}[
        "metadata"
    ]
    assert check.status == "missing"
    assert "epiforecasts-hubrenewal" in check.detail


def test_metadata_abbr_too_long_is_missing(settings: Settings) -> None:
    mid = "epiforecasts-abcdefghijklmnop"
    (settings.metadata_dir / f"{mid}.yml").write_text(
        "team_abbr: epiforecasts\nmodel_abbr: abcdefghijklmnop\n"
    )
    runner = ready_runner(
        **{
            f"repos/{REPO}/actions/variables": {
                "variables": [{"name": "HUB_MODEL_ID", "value": mid}]
            }
        }
    )
    check = {c.name: c for c in run_checks(runner, settings)}["metadata"]
    assert check.status == "missing"
    assert "16" in check.detail


def test_chosen_model_is_ok(settings: Settings) -> None:
    check = {c.name: c for c in run_checks(ready_runner(), settings)}[
        "model choice"
    ]
    assert check.detail == "submits `renewal`, as last chosen"


def test_persistence_chosen_warns(settings: Settings) -> None:
    settings.model_choice.write_text(
        "choices:\n  - date: 2026-10-10\n    model: persistence\n"
    )
    check = {c.name: c for c in run_checks(ready_runner(), settings)}[
        "model choice"
    ]
    assert check.status == "warn"
    assert "persistence" in check.detail


def test_unregistered_model_id_is_reported(settings: Settings) -> None:
    runner = ready_runner(
        **{f"repos/{HUB}/contents/model-metadata/{MID}.yml": None}
    )
    check = {c.name: c for c in run_checks(runner, settings)}["hub model id"]
    assert check.status == "ok"
    assert "first" in check.detail


def test_bot_owned_cron_lines_warn(settings: Settings) -> None:
    runner = ready_runner()
    runner.on(["git", "log"], "seabbs-bot\tsignin@samabbott.co.uk\n")
    assert statuses(run_checks(runner, settings))["cron owner"] == "warn"


def test_unreachable_hub_is_missing(settings: Settings) -> None:
    runner = ready_runner()
    runner.on(["git", "ls-remote"], "", 128)
    assert statuses(run_checks(runner, settings))["hub reachable"] == "missing"


def test_unreadable_settings_are_unknown(settings: Settings) -> None:
    runner = ready_runner(
        **{f"repos/{REPO}/actions/permissions/workflow": None}
    )
    runner.on(
        ["gh", "api", f"repos/{REPO}/actions/permissions/workflow"],
        json.dumps({"message": "Must have admin rights"}),
        1,
    )
    s = statuses(run_checks(runner, settings))
    assert s["workflow permissions"] == "unknown"


def test_render_lists_fixes_for_gaps(settings: Settings) -> None:
    runner = ready_runner(**{f"repos/{FORK}": None})
    text = render(run_checks(runner, settings))
    assert "MISSING" in text
    assert "gh repo fork" in text
    assert text.rstrip().splitlines()[-1].startswith("1 missing")


def test_main_exits_1_on_gaps(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ready_runner(**{f"repos/{FORK}": None})
    monkeypatch.setattr("submission_ci.preflight.subprocess_runner", runner)
    hub_toml = settings.metadata_dir / "hub.toml"
    hub_toml.write_text('[submission]\nmodel_abbr = "seabbsrenewal"\n')
    args = [
        "--hub-config",
        str(hub_toml),
        "--repo",
        REPO,
        "--metadata-dir",
        str(settings.metadata_dir),
        "--model-choice",
        str(settings.model_choice),
        "--now",
        NOW.isoformat(),
    ]
    assert main(args) == 1
    monkeypatch.setattr(
        "submission_ci.preflight.subprocess_runner", ready_runner()
    )
    assert main(args) == 0


def test_second_hub_model_is_checked(settings: Settings) -> None:
    checks = {c.name: c for c in run_checks(ready_runner(), settings)}
    assert checks[f"metadata {MID2}"].status == "ok"
    assert checks[f"hub model id {MID2}"].status == "ok"
    (settings.metadata_dir / f"{MID2}.yml").unlink()
    checks = {c.name: c for c in run_checks(ready_runner(), settings)}
    assert checks[f"metadata {MID2}"].status == "missing"
