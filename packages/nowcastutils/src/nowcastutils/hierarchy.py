# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Location hierarchies read from ISO 3166-2 style codes.

A code is its parent's code with one more part after a hyphen: a
subdivision `XX-YY` sits in the country `XX`, and a unit `XX-YY-NN`
in `XX-YY`.
A code with no hyphen is a root.
The models call the levels below the root provinces and zones.
"""

from __future__ import annotations

__all__ = ["depth", "is_zone", "parent", "province_of"]

SEP = "-"
"""The separator between a code's parts."""


def parent(code: str) -> str | None:
    """Return the parent of a location code.

    Parameters
    ----------
    code : str
        A location code.

    Returns
    -------
    str or None
        The code without its last part, or None for a root.
    """
    head, sep, _ = code.rpartition(SEP)
    return head if sep else None


def depth(code: str) -> int:
    """Return how far a location code sits below its root.

    Parameters
    ----------
    code : str
        A location code.

    Returns
    -------
    int
        0 for a root, 1 for its children, and so on.
    """
    return code.count(SEP)


def is_zone(code: str) -> bool:
    """Return whether a location code is a zone, two levels below the root.

    The models' three levels are the country (`XX`), its provinces
    (`XX-YY`) and their zones (`XX-YY-NN`).

    Parameters
    ----------
    code : str
        A location code.

    Returns
    -------
    bool
        True for a zone, False for the country or a province.
    """
    return depth(code) == 2


def province_of(zone: str) -> str:
    """Return the province of a zone code.

    Parameters
    ----------
    zone : str
        A zone code, `XX-YY-NN`.

    Returns
    -------
    str
        The province, `XX-YY`.
    """
    return SEP.join(zone.split(SEP)[:2])
