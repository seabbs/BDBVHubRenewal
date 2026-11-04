"""Which models the interactive charts and tables tick at first."""

from __future__ import annotations

from collections.abc import Collection

from bdbvhubrenewal.evaluation.nowcast import BASELINE

OFF_AT_FIRST = frozenset({BASELINE})


def starts_on(model: str, on: Collection[str]) -> bool:
    """Return whether `model` is ticked when a chart or table first draws.

    Parameters
    ----------
    model : str
        The model's name.
    on : collection of str
        The models a chart asks to tick.

    Returns
    -------
    bool
        True if `model` is in `on` and not in `OFF_AT_FIRST`.
        Models in `OFF_AT_FIRST` stay a tick box people can turn on.
    """
    return model in on and model not in OFF_AT_FIRST
