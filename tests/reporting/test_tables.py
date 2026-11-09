from __future__ import annotations

from bdbvhubrenewal.reporting.tables import filter_table


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
