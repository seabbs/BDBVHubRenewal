"""What the repository's releases hold.

`labels`: checkpoint labels of releases published from a cut-off, the
final run's `chronos-slots-<hash>` first as `slots-<hash>`, then
`chronos-ft-<hash>` as `<hash>`, each group oldest first. Other releases
are left out. The labels must be the ones `bdbvhubrenewal.evaluation
.checkpoint` takes.

`by_time` orders the same labels oldest first across both kinds, and
`priority` gives newer checkpoints' jobs a higher priority.

`by_time` orders the same labels oldest first across both kinds, and
`priority` gives newer checkpoints' jobs a higher priority.

`released_seeds`: the simulation batches a `sims-r<N>` release holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

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


def by_time(
    listing: Iterable[Mapping[str, str]], published_from: str
) -> list[str]:
    """Return the checkpoint labels of `labels`, oldest published first."""
    at = {r["tagName"]: r["publishedAt"] for r in listing}
    tag = {
        lb: (SLOTS + lb.removeprefix("slots-"))
        if lb.startswith("slots-")
        else FT + lb
        for lb in labels(listing, published_from)
    }
    return sorted(tag, key=lambda lb: at[tag[lb]])


def priority(order: Sequence[str], label: str, base: float) -> float:
    """Return a job's priority, newer checkpoints higher, within `base`.

    The labels of `order` (oldest first) take `base` plus their rank as
    a fraction below one, so scoring keeps its place among the rules;
    any other label (the reference) takes `base`.
    """
    if label not in order:
        return base
    return base + (list(order).index(label) + 1) / (len(order) + 1)


def released_seeds(names: Iterable[str]) -> set[int]:
    """Return the batches whose counts and settings are both assets."""
    ext = {"counts": ".csv.gz", "settings": ".csv"}
    have: dict[int, set[str]] = {}
    for name in names:
        m = re.fullmatch(r"(counts|settings)-(\d+)(\.csv(?:\.gz)?)", name)
        if m and m[3] == ext[m[1]]:
            have.setdefault(int(m[2]), set()).add(m[1])
    return {s for s, kinds in have.items() if len(kinds) == 2}
