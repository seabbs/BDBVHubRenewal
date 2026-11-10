# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How province samples are paired before national is summed.

The hub file takes national as the sum of the provinces in each sample,
and zones as a split of their province (`to_hub_table`). Both are
coherent in every sample. What the pairing sets is the dependence: which
draw of one province is summed with which draw of another, and so the
width of national.

`reconcile` gives two methods, chosen in `config/hub.toml`
(`[reconciliation]`):

- `sum`, the default: samples are summed as the model gave them.
- `copula`: each province's whole trajectories are reordered across
  samples to the ranks of a Gaussian copula with exchangeable
  correlation `rho`, ranked on each trajectory's mean of
  $\log(1 + y)$ over the weeks, as in ensemble copula coupling or the
  Schaake shuffle. Each province keeps its own trajectories, so its
  forecast is unchanged, and its zones move with it, so they still sum
  to it. National, the sum, is wider for `rho > 0`.

`zone_pairing` sets how the random walks' zone run is paired with their
province run before the split (`forecasters.zones.reconcile_zones`):
`sample` pairs draws of the same id, `rank` pairs whole trajectories by
rank of their total.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.zones import province_of

__all__ = [
    "METHODS",
    "PAIRINGS",
    "Reconciliation",
    "reconcile",
    "reconciliation_settings",
]

METHODS: Final = ("sum", "copula")
"""Ways to pair province samples before national is summed."""

PAIRINGS: Final = ("sample", "rank")
"""Ways to pair the random walks' zone and province runs."""


@dataclass(frozen=True)
class Reconciliation:
    """The reconciliation settings.

    Attributes
    ----------
    method : str
        One of `METHODS`.
    rho : float
        The copula's correlation between provinces, in [0, 1].
    zone_pairing : str
        One of `PAIRINGS`.
    """

    method: str = "sum"
    rho: float = 0.2
    zone_pairing: str = "sample"

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be between 0 and 1.")
        if self.zone_pairing not in PAIRINGS:
            raise ValueError(f"zone_pairing must be one of {PAIRINGS}.")


def reconciliation_settings(
    path: Path | str = Path("config/hub.toml"),
) -> Reconciliation:
    """Read `[reconciliation]` from the hub config.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub config.

    Returns
    -------
    Reconciliation
        The settings, the defaults where the file gives none.
    """
    with Path(path).open("rb") as fh:
        table = tomllib.load(fh).get("reconciliation", {})
    return Reconciliation(**table)


def _order(
    summary: np.ndarray, z: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample ids reordered so the summary ranks as `z` does.

    Ties in the summary are broken at random.
    """
    by_summary = np.lexsort((rng.random(summary.size), summary))
    return by_summary[np.argsort(np.argsort(z))]


def reconcile(
    samples: pd.DataFrame,
    zones: pd.DataFrame | None,
    settings: Reconciliation,
    seed: int | np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Pair province samples, and their zones, by the chosen method.

    Parameters
    ----------
    samples : pandas.DataFrame
        Province samples, columns `location`, `target_end_date`, `sample`
        and `value`. National rows are dropped by `to_hub_table` and are
        left here as they are.
    zones : pandas.DataFrame or None
        Zone samples that sum to their province, or None.
    settings : Reconciliation
        The method and its settings.
    seed : int or numpy.random.Generator
        Seeds the copula.

    Returns
    -------
    tuple
        The samples and zones, with the sample ids of each province, and
        of its zones, reordered. Unchanged for `sum`.
    """
    if settings.method == "sum":
        return samples, zones
    rng = np.random.default_rng(seed)
    wide = samples.pivot_table(
        index="sample",
        columns=["location", "target_end_date"],
        values="value",
        aggfunc="first",
    )
    ids = wide.index.to_numpy()
    places = sorted(
        loc for loc in set(wide.columns.get_level_values(0)) if len(loc) == 5
    )
    summary = {
        loc: np.log1p(wide[loc].clip(lower=0)).mean(axis=1).to_numpy()
        for loc in places
    }
    active = [loc for loc in places if np.ptp(summary[loc]) > 0]
    k = len(active)
    if k < 2:
        return samples, zones
    rho = settings.rho
    cov = np.full((k, k), rho) + (1.0 - rho) * np.eye(k)
    z = rng.multivariate_normal(np.zeros(k), cov, size=ids.size, method="eigh")
    # New id of each old id, by province: the draw ranked r in the
    # province goes to the copula draw ranked r.
    moves: dict[str, dict[int, int]] = {}
    for j, loc in enumerate(active):
        old = ids[_order(summary[loc], z[:, j], rng)]
        moves[loc] = dict(zip(old.tolist(), ids.tolist(), strict=True))

    def move(table: pd.DataFrame, parent: pd.Series) -> pd.DataFrame:
        out = table.copy()
        for loc, mapping in moves.items():
            rows = parent == loc
            out.loc[rows, "sample"] = table.loc[rows, "sample"].map(mapping)
        return out.sort_values(
            ["location", "sample", "target_end_date"]
        ).reset_index(drop=True)[list(table.columns)]

    out = move(samples, samples["location"])
    out_zones = None
    if zones is not None:
        out_zones = move(zones, zones["location"].map(province_of))
    return out, out_zones
