from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.reporting.tables import (
    filter_table,
    markdown_table,
    sort_table,
    sortable_table,
)


def test_filter_table_wraps_the_table_for_the_script() -> None:
    md = "| target | model |\n|:--|:--|\n| reported | randomwalk |\n"
    out = filter_table(md, ["target", "model"])
    assert out.startswith(
        '<div class="tablefilter" data-filters="target,model" markdown="1">'
    )
    # The table is kept whole, between blank lines, so it renders as is.
    assert f"\n\n{md.strip()}\n\n</div>" in out


def test_filter_table_opens_on_the_defaults_given() -> None:
    md = "| baseline | model |\n|:--|:--|\n| randomwalk | renewal |\n"
    out = filter_table(md, ["baseline"], {"baseline": "randomwalk"})
    assert """data-defaults='{"baseline": "randomwalk"}'""" in out
    assert "data-defaults" not in filter_table(md, ["baseline"])


def test_sort_table_wraps_the_table_for_sorting_only() -> None:
    md = "| model | wis |\n|:--|--:|\n| randomwalk | 1.2 |\n"
    out = sort_table(md)
    assert out.startswith('<div class="tablesort" markdown="1">')
    assert f"\n\n{md.strip()}\n\n</div>" in out


def test_markdown_table_leaves_missing_cells_empty() -> None:
    df = pd.DataFrame(
        {"model": ["a", "b"], "wis": [1.234, np.nan], "n": [3, 4]}
    )
    out = markdown_table(df)
    assert "nan" not in out.lower()
    assert "| 1.23 |" in out
    rows = out.strip().splitlines()
    assert [c.strip() for c in rows[-1].strip("|").split("|")] == ["b", "", "4"]


def test_sortable_table_is_a_wrapped_markdown_table() -> None:
    df = pd.DataFrame({"model": ["a"], "wis": [0.12345]})
    out = sortable_table(df, digits=3)
    assert out == sort_table(markdown_table(df, digits=3))
    assert "0.123" in out


def test_the_site_loads_the_sort_script_before_its_users() -> None:
    config = (Path(__file__).parents[2] / "mkdocs.yml").read_text()
    scripts = [
        line.strip().removeprefix("- ")
        for line in config.splitlines()
        if line.strip().startswith("- javascripts/")
    ]
    first = scripts.index("javascripts/tablesort.js")
    assert first < scripts.index("javascripts/tablefilter.js")
    assert first < scripts.index("javascripts/scoresummary.js")
