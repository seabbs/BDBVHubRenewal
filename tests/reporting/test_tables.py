from __future__ import annotations

import markdown

from bdbvhubrenewal.reporting.tables import filter_table


def test_filter_table_keeps_a_plain_table() -> None:
    md = "| target | model |\n|:--|:--|\n| reported | randomwalk |\n"
    html = markdown.markdown(
        filter_table(md, ["target", "model"]),
        extensions=["md_in_html", "tables"],
    )
    assert 'data-filters="target,model"' in html
    assert "<table>" in html
    assert ">randomwalk</td>" in html
