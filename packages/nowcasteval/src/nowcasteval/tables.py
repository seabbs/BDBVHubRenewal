# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Site tables: sortable and searchable blocks the page scripts fill.

`docs/javascripts/tablesort.js` makes the headers of a table inside a
`tablesort` block sortable. `docs/javascripts/tablefilter.js` also adds a
text search and a select per named column to a `tablefilter` block.
Without JavaScript each is a plain table.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = ["filter_table", "markdown_table", "sort_table", "sortable_table"]


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


def markdown_table(df: pd.DataFrame, digits: int = 2) -> str:
    """Render a data frame as a Markdown table, missing cells left empty.

    Parameters
    ----------
    df : pandas.DataFrame
        The table.
    digits : int
        Decimal places numbers are rounded to.

    Returns
    -------
    str
        A Markdown table, ending in a newline.
    """
    rounded = df.round(digits)
    shown = rounded.astype(object).where(rounded.notna(), None)
    return shown.to_markdown(index=False, missingval="") + "\n"


def sort_table(table: str) -> str:
    """Wrap a Markdown table in a block whose headers sort the rows.

    Parameters
    ----------
    table : str
        A Markdown table.

    Returns
    -------
    str
        Markdown with the table inside a `tablesort` block. It opens in the
        order given.
    """
    return (
        f'<div class="tablesort" markdown="1">\n\n{table.strip()}\n\n</div>\n'
    )


def sortable_table(df: pd.DataFrame, digits: int = 2) -> str:
    """Render a data frame as a sortable Markdown table.

    Parameters
    ----------
    df : pandas.DataFrame
        The table.
    digits : int
        Decimal places numbers are rounded to.

    Returns
    -------
    str
        `markdown_table` of the frame inside a `sort_table` block.
    """
    return sort_table(markdown_table(df, digits))
