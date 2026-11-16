"""The Pages workflow deploys from main only and can read a scratch store."""

from __future__ import annotations

from pathlib import Path

import yaml

PAGES = Path(__file__).parents[1] / ".github" / "workflows" / "pages.yml"


def _workflow() -> dict:
    return yaml.safe_load(PAGES.read_text())


def test_only_main_deploys_the_site() -> None:
    condition = _workflow()["jobs"]["deploy"]["if"]
    assert "github.ref == 'refs/heads/main'" in condition


def test_a_dispatch_can_name_a_scratch_store() -> None:
    wf = _workflow()
    # PyYAML reads the key `on` as True.
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert "store_branch" in inputs
    steps = wf["jobs"]["prep"]["steps"]
    (load,) = [s for s in steps if s.get("name") == "Load the stored fits"]
    assert load["env"]["FIT_STORE_BRANCH"] == (
        "${{ inputs.store_branch || 'fit-store' }}"
    )
