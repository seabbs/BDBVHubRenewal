# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Discretised generation time and infection-to-reference delay pmfs.

Each is the continuous distribution's mass on whole days, cut at its 99.9%
quantile and renormalised, as the simulator's R engine makes them. They
are the values of PyRenew `DeterministicPMF` components.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["gt_pmf", "incubation_pmf"]


def gt_pmf(mean: float, family: str, cv: float) -> np.ndarray:
    """Return a generation time pmf on 1, ..., n days.

    Parameters
    ----------
    mean : float
        The mean, in days.
    family : str
        `"gamma"` or `"lognormal"`.
    cv : float
        The coefficient of variation.

    Returns
    -------
    numpy.ndarray
        The pmf, element 0 at lag 1 (PyRenew's convention), with n the
        99.9% quantile rounded up and at least 2.

    Raises
    ------
    ValueError
        If `family` is neither.
    """
    if family == "gamma":
        k = 1 / cv**2
        d = stats.gamma(k, scale=mean / k)
    elif family == "lognormal":
        sdlog = math.sqrt(math.log(1 + cv**2))
        d = stats.lognorm(sdlog, scale=math.exp(math.log(mean) - sdlog**2 / 2))
    else:
        raise ValueError(f"unknown family {family!r}")
    n = max(2, math.ceil(d.ppf(0.999)))
    pmf = np.diff(d.cdf(np.arange(n + 1)))
    return np.asarray(pmf / pmf.sum(), dtype=float)


def incubation_pmf(mean: float, cv: float = 0.5) -> np.ndarray:
    """Return a gamma infection-to-reference delay pmf on 0, 1, ... days.

    Parameters
    ----------
    mean : float
        The mean, in days.
    cv : float
        The coefficient of variation.

    Returns
    -------
    numpy.ndarray
        The pmf to the 99.9% quantile, element 0 at a delay of 0 days.
    """
    shape = 1 / cv**2
    d = stats.gamma(shape, scale=mean / shape)
    n = max(1, math.ceil(d.ppf(0.999)))
    pmf = np.diff(d.cdf(np.arange(n + 1)))
    return np.asarray(pmf / pmf.sum(), dtype=float)
