# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Infections imported into each stratum from outside the population.

Imports are added to a day's infections after depletion, so they seed or
reseed a stratum whatever its force of infection.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

__all__ = ["KINDS", "Imports"]

KINDS = ("none", "constant", "pulses")
"""Import kinds into strata with cases: none, a constant rate, or pulses."""


class Imports(RandomVariable):
    """Each stratum's imported infections on each day.

    A late stratum is imported its unit over `late_days` days from a
    share `late_at` of the way through, and nothing else. Other strata
    with a positive unit are imported a constant `rate` times their unit
    a day, or pulses of `pulse_size` times it at shares `pulse_at` of the
    way through, plus a constant `trickle` a day.

    Parameters
    ----------
    name : str
        Name of the component.
    kind : str
        One of `KINDS`.
    rate : float
        For `constant`, the daily import as a share of the unit.
    pulse_at : sequence of float
        For `pulses`, the shares of the way through (each rounded to a
        day, at least day 1; pulses on the same day count once).
    pulse_size : float
        For `pulses`, each pulse as a multiple of the unit.
    late : array_like of bool, optional
        Strata seeded late. `None` is none.
    late_at : float
        The share of the way through at which late strata are seeded.
    late_days : int
        The days over which they are seeded.
    trickle : float
        A constant daily import into every stratum with cases that is not
        late.

    Raises
    ------
    ValueError
        If `kind` is not one of `KINDS`.
    """

    def __init__(
        self,
        name: str,
        kind: str = "none",
        rate: float = 0.0,
        pulse_at: Sequence[float] | ArrayLike = (),
        pulse_size: float = 0.0,
        late: ArrayLike | None = None,
        late_at: float = 0.0,
        late_days: int = 1,
        trickle: float = 0.0,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        super().__init__(name=name)
        self.kind = kind
        self.rate = rate
        self.pulse_at = np.atleast_1d(np.asarray(pulse_at, dtype=float))
        self.pulse_size = pulse_size
        self.late = None if late is None else np.asarray(late, dtype=bool)
        self.late_at = late_at
        self.late_days = int(late_days)
        self.trickle = trickle

    def sample(self, n: int, unit: ArrayLike, **kwargs: object) -> Array:
        """Return the imports.

        Parameters
        ----------
        n : int
            Days.
        unit : array_like
            Each stratum's unit (its seed or level), 0 for strata without
            cases.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Days by strata.
        """
        unit = np.asarray(unit, dtype=float)
        k = unit.shape[0]
        out = np.zeros((n, k))
        late = np.zeros(k, dtype=bool) if self.late is None else self.late
        if late.any():
            first = max(1, int(np.round(self.late_at * n)))
            last = min(n, first + self.late_days - 1)
            days = np.arange(first, last + 1) - 1
            out[np.ix_(days, np.flatnonzero(late))] = unit[late] / len(days)
        on = ~late & (unit > 0)
        if self.kind == "constant":
            out[:, on] = self.rate * unit[on]
        elif self.kind == "pulses":
            at = np.maximum(1, np.round(self.pulse_at * n)).astype(int)
            days = np.unique(at) - 1
            out[np.ix_(days, np.flatnonzero(on))] += self.pulse_size * unit[on]
        out[:, on] += self.trickle
        return jnp.asarray(out)
