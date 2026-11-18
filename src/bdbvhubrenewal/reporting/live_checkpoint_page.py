# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The docs' preliminary summary of the live fine-tuning run.

Only the docs build runs it (the `live-checkpoint` command), so it may use
the report-only tables (`evaluation.digest.REPORT_ONLY`). The scores it
summarises are `reporting.live_checkpoint`'s.
"""

from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.reporting.live_checkpoint import (
    LIVE_RUN,
    label,
    newest,
    summary_table,
)
from bdbvhubrenewal.reporting.tables import markdown_table

__all__ = ["docs_page"]


def docs_page(scores: pd.DataFrame | None, run: str = LIVE_RUN) -> str:
    """Return the docs' preliminary summary of the live run, as Markdown.

    Without scores, a note says they are not yet available.
    """
    if scores is None or not newest(scores, run):
        return "*Fine-tuned checkpoint scores are not yet available.*\n"
    top = newest(scores, run)
    n = max(n for _, n in top.values())
    return (
        f"**{label(n)}** Scores of the newest scored checkpoint of the "
        f"fine-tuning run (`{run}`) on development data only: national WIS "
        "on the German first half and CRPS on the BDBV release backtest, "
        "relative to `randomwalknowcast` (daily / weekly model), and the "
        "trainer's held-out loss. Brackets give each cell's simulations.\n\n"
        + markdown_table(summary_table(scores, run))
        + "\n"
    )
