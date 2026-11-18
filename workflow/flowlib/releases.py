"""What the repository's releases hold.

`labels`: checkpoint labels of releases published from a cut-off, the
final run's `chronos-slots-<hash>` first as `slots-<hash>`, then
`chronos-ft-<hash>` as `<hash>`, each group oldest first. Other releases
are left out. The labels must be the ones `scripts/checkpoint_scores.py`
takes.

`released_seeds`: the simulation batches a `sims-r<N>` release holds.
"""

from __future__ import annotations

import re
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


def released_seeds(names: Iterable[str]) -> set[int]:
    """Return the batches whose counts and settings are both assets."""
    have: dict[int, set[str]] = {}
    for name in names:
        m = re.fullmatch(r"(counts)-(\d+)\.csv\.gz|(settings)-(\d+)\.csv", name)
        if m:
            kind = m[1] or m[3]
            have.setdefault(int(m[2] or m[4]), set()).add(kind)
    return {s for s, kinds in have.items() if len(kinds) == 2}
