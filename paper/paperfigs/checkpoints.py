"""Fine-tuning checkpoints, as models named `<model>@<outbreaks>`.

The fine-tune saves checkpoints at doublings of simulated outbreaks from
1,000 upwards. The synthetic study forecasts with each it finds, as
`fm_chronos_daily_ragged_ft@1000` and so on, and the main model is the
last. `checkpoint_of`
splits a name, and `drop_checkpoints` keeps every other model, for the
results panel and tables.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = ["SEP", "checkpoint_of", "drop_checkpoints"]

SEP: Final = "@"
"""Separates a fine-tuned model from its number of training outbreaks."""


def checkpoint_of(model: str) -> tuple[str, int] | None:
    """Return a checkpoint's model and training outbreaks, or None."""
    name, sep, size = model.rpartition(SEP)
    if not sep or not size.isdigit():
        return None
    return name, int(size)


def drop_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of models that are not checkpoints."""
    keep = [checkpoint_of(m) is None for m in df["model"]]
    return df.loc[keep].reset_index(drop=True)
