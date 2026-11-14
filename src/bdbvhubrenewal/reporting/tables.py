# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Searchable tables: the block `docs/javascripts/tablefilter.js` fills.

The script adds a text search, a select per named column and sortable
headers to the Markdown table inside the block. Without JavaScript it is
a plain table.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

__all__ = ["filter_table"]


def filter_table(
    table: str,
    filters: Sequence[str] = (),
    defaults: Mapping[str, str] | None = None,
) -> str:
    """Wrap a Markdown table in a searchable, sortable block.

    Parameters
    ----------
    table : str
        A Markdown table.
    filters : sequence of str
        Column headers to filter by, each with a select.
    defaults : mapping of str to str, optional
        The value each named filter opens on, such as the baseline of a
        table stacked by baseline. Others open on all rows.

    Returns
    -------
    str
        Markdown with the table inside a `tablefilter` block.
    """
    opens = "" if not defaults else f" data-defaults='{json.dumps(defaults)}'"
    return (
        f'<div class="tablefilter" data-filters="{",".join(filters)}"'
        f"{opens} "
        'markdown="1">\n\n'
        f"{table.strip()}\n\n"
        "</div>\n"
    )
