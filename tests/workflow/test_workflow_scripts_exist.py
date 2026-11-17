"""Every script a workflow rule runs is in the repository."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RULES = [
    ROOT / "workflow" / "Snakefile",
    *(ROOT / "workflow" / "rules").glob("*.smk"),
]
SCRIPT = re.compile(r"(?<![\w/.-])((?:workflow/)?scripts/[\w/]+\.(?:py|R|sh))")


def _missing() -> list[str]:
    named = {m for rule in RULES for m in SCRIPT.findall(rule.read_text())}
    return sorted(p for p in named if not (ROOT / p).exists())


@pytest.mark.xfail(
    strict=True,
    reason="#260 merged before fm-finetune-slots, which brings the "
    "fine-tuning and scoring scripts (#277); remove this mark when it lands",
)
def test_every_script_a_rule_runs_exists() -> None:
    assert _missing() == []


def test_the_rules_name_scripts_at_all() -> None:
    # Guards the pattern: the rules run the simulator and the dashboard.
    named = {m for rule in RULES for m in SCRIPT.findall(rule.read_text())}
    assert "scripts/finetune_simulate.R" in named
    assert "workflow/scripts/dashboard.py" in named
