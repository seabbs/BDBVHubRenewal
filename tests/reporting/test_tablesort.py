"""The shared sort order of docs/javascripts/tablesort.js, run in node."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

TABLESORT = Path(__file__).parents[2] / "docs" / "javascripts" / "tablesort.js"


def _order(keys: list[str], direction: int) -> list[int]:
    script = (
        f"const t = require({json.dumps(str(TABLESORT))});"
        f"process.stdout.write(JSON.stringify(t.order({json.dumps(keys)}, "
        f"{direction})));"
    )
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _sorted(keys: list[str], direction: int) -> list[str]:
    return [keys[i] for i in _order(keys, direction)]


def test_numbers_sort_as_numbers() -> None:
    keys = ["10", "9", "2.5", "-1", "1e-3", "1,200"]
    assert _sorted(keys, 1) == ["-1", "1e-3", "2.5", "9", "10", "1,200"]
    assert _sorted(keys, -1) == ["1,200", "10", "9", "2.5", "1e-3", "-1"]


@pytest.mark.parametrize("direction", [1, -1])
def test_missing_values_go_last_either_way(direction: int) -> None:
    keys = ["", "3", "nan", "1", "\u2013", "NA", "2", "—", "NaN"]
    got = _sorted(keys, direction)
    numbers = ["1", "2", "3"] if direction > 0 else ["3", "2", "1"]
    assert got[:3] == numbers
    # Missing cells keep the order they had.
    assert got[3:] == ["", "nan", "\u2013", "NA", "—", "NaN"]


def test_text_sorts_as_text_and_dates_by_date() -> None:
    keys = ["renewal", "KIT", "randomwalk", "2026-10-03", "2026-09-26"]
    assert _sorted(keys, 1) == [
        "2026-09-26",
        "2026-10-03",
        "KIT",
        "randomwalk",
        "renewal",
    ]


def test_ties_keep_the_default_order() -> None:
    keys = ["1", "0", "1", "0", "1"]
    assert _order(keys, 1) == [1, 3, 0, 2, 4]
    assert _order(keys, -1) == [0, 2, 4, 1, 3]
