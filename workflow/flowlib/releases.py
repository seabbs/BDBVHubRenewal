"""Checkpoint labels from the repository's releases.

Releases published from a cut-off, the final run's `chronos-slots-<hash>`
first as `slots-<hash>`, then `chronos-ft-<hash>` as `<hash>`, each group
oldest first. Other releases are left out. The labels must be the ones
`scripts/checkpoint_scores.py` takes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

FT = "chronos-ft-"
SLOTS = "chronos-slots-"


def labels(
    listing: Iterable[Mapping[str, str]], published_from: str
) -> list[str]:
    """Return the checkpoint labels of `gh release list --json` rows."""
    order = {}
    for r in listing:
        tag, at = r["tagName"], r["publishedAt"]
        if at < published_from:
            continue
        if tag.startswith(SLOTS):
            order["slots-" + tag.removeprefix(SLOTS)] = (0, at)
        elif tag.startswith(FT):
            order[tag.removeprefix(FT)] = (1, at)
    return sorted(order, key=order.__getitem__)
