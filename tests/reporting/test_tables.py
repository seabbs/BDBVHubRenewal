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
