# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Renewal process with spatial importation and susceptible depletion.

A NumPy reference for the model's latent process.
The simulator uses it, and the model phase checks its JAX version
(built on PyRenew's `SubpopulationInfections`) against it.

For province $p$ on day $t$,

$$
\lambda_p(t) = R_p(t)\,\frac{S_p(t)}{N_p}\sum_{s \ge 1} g(s)
\Big[(1 - \varepsilon) I_p(t - s) + \varepsilon \sum_{q} K_{qp}
I_q(t - s)\Big] + \iota_p(t),
$$

$$
S_p(t) = N_p - \sum_{u < t} I_p(u),
$$

where $g$ is the generation interval pmf, $K$ a row-normalised
importation kernel with zero diagonal, $\varepsilon$ the share of
transmission sent to other provinces, $N_p$ the effective population and
$\iota_p(t)$ seeding.
$I_p(t) = \lambda_p(t)$ for expected values, or
$I_p(t) \sim \text{Poisson}(\lambda_p(t))$ when a random generator is
given.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["renewal"]


def renewal(
    rt: NDArray[np.float64],
    generation_pmf: NDArray[np.float64],
    seeding: NDArray[np.float64],
    population: NDArray[np.float64],
    kernel: NDArray[np.float64] | None = None,
    importation: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the renewal process forward.

    Parameters
    ----------
    rt : numpy.ndarray
        Reproduction numbers, shape (provinces, days).
    generation_pmf : numpy.ndarray
        Generation interval pmf on days 0, 1, ...; day 0 is ignored.
    seeding : numpy.ndarray
        Seeded infections, shape (provinces, days).
    population : numpy.ndarray
        Effective population of each province.
    kernel : numpy.ndarray, optional
        Importation kernel, rows summing to one. Needed if
        `importation > 0`.
    importation : float
        Share of transmission sent to other provinces.
    rng : numpy.random.Generator, optional
        If given, draw Poisson infections; otherwise use expected values.

    Returns
    -------
    tuple of numpy.ndarray
        Infections and the susceptible fraction, both (provinces, days).
    """
    n_loc, n_days = rt.shape
    g = np.asarray(generation_pmf, dtype=np.float64).copy()
    g[0] = 0.0
    g = g / g.sum()
    k = np.zeros((n_loc, n_loc)) if kernel is None else kernel
    infections = np.zeros((n_loc, n_days))
    susceptible = np.ones((n_loc, n_days))
    cumulative = np.zeros(n_loc)
    for t in range(n_days):
        lo = max(0, t - g.size + 1)
        past = infections[:, lo:t][:, ::-1]
        force = past @ g[1 : t - lo + 1] if t > lo else np.zeros(n_loc)
        mixed = (1 - importation) * force + importation * (k.T @ force)
        susceptible[:, t] = np.clip(1 - cumulative / population, 0, 1)
        lam = rt[:, t] * susceptible[:, t] * mixed + seeding[:, t]
        infections[:, t] = rng.poisson(lam) if rng is not None else lam
        cumulative += infections[:, t]
    return infections, susceptible
