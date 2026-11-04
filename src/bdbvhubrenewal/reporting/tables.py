# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Searchable tables: the block `docs/javascripts/tablefilter.js` fills.

The script adds a text search, a select per named column and sortable
headers to the Markdown table inside the block. Without JavaScript it is
a plain table.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["filter_table"]


def filter_table(table: str, filters: Sequence[str] = ()) -> str:
    """Wrap a Markdown table in a searchable, sortable block.

    Parameters
    ----------
    table : str
        A Markdown table.
    filters : sequence of str
        Column headers to filter by, each with a select.

    Returns
    -------
    str
        Markdown with the table inside a `tablefilter` block.
    """
    return (
        f'<div class="tablefilter" data-filters="{",".join(filters)}" '
        'markdown="1">\n\n'
        f"{table.strip()}\n\n"
        "</div>\n"
    )
