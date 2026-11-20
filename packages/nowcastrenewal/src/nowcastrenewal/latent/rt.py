# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Reproduction number paths for simulated outbreaks, by place and stratum.

- `RtPath`: a deterministic log R path of one of `KINDS`.
- `WeeklyNoise`: an AR(1) on the weekly scale, held for 7 days, from
  PyRenew's `ARProcess`, starting at 0 in the first week.
- `PlaceRt`: each stratum's log R. Places share a path, follow their own,
  add correlated weekly noise to it or lag it. A slow drift and weekly
  noise scaled by `1 - coupling` are added to every stratum.

The random draws are numpyro sample sites of standard normals (and a
categorical for an independent place's path), so a run is set by a key
under `numpyro.handlers.seed`, or replayed with
`numpyro.handlers.substitute`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable
from pyrenew.process import ARProcess

__all__ = [
    "KINDS",
    "RELATIONS",
    "PlaceRt",
    "RtPath",
    "WeeklyNoise",
    "daily_walk",
]

KINDS = (
    "endemic",
    "controlled",
    "growth",
    "decline",
    "flat",
    "rise_fall_gradual",
    "fall_rise",
    "rise_fall_sharp",
    "oscillating",
    "multi_wave",
)
"""Path kinds: a weekly walk around R = 1 (`endemic`), a start at the
amplitude decaying onto that walk (`controlled`), and shapes on log R."""

RELATIONS = ("shared", "independent", "correlated", "lagged")
"""How places' paths relate."""

# The weekly noise's autocorrelation.
_AUTOREG = 0.8


def daily_walk(
    days: ArrayLike | Sequence[float], walk: ArrayLike | Sequence[float]
) -> np.ndarray:
    """Read a weekly walk on `days` (0 the first), linear between weeks.

    Parameters
    ----------
    days : array_like
        Days.
    walk : array_like
        The walk's weekly values; past its end it stays at the last.

    Returns
    -------
    numpy.ndarray
        The walk on each day.
    """
    x = np.asarray(days, dtype=float) / 7
    w = np.floor(x).astype(int)
    f = x - w
    walk = np.atleast_1d(np.asarray(walk, dtype=float))
    last = len(walk) - 1
    out = walk[np.minimum(w, last)] * (1 - f)
    return np.asarray(out + walk[np.minimum(w + 1, last)] * f, dtype=float)


def _plogis(x: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-x)))


def _vector(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray([] if x is None else x, dtype=float))


@dataclass(frozen=True)
class RtPath:
    """A log R path.

    Attributes
    ----------
    kind : str
        One of `KINDS`.
    amplitude, peak, flip_days, period, phase, waves, control_days, flat :
        float
        The shapes' settings; each kind uses some of them.
    wave_scale : sequence of float
        Each wave's scale, for `multi_wave`.
    walk : sequence of float
        The weekly walk, for `endemic` and `controlled`.
    """

    kind: str
    amplitude: float = 0.0
    peak: float = 0.5
    flip_days: float = 1.0
    period: float = 1.0
    phase: float = 0.0
    waves: float = 1.0
    control_days: float = 1.0
    flat: float = 0.0
    wave_scale: Sequence[float] = field(default=(1.0,))
    walk: Sequence[float] = field(default=(0.0,))

    @classmethod
    def from_values(cls, kind: str, values: Mapping[str, Any]) -> RtPath:
        """Build a path from a simulation's values.

        Parameters
        ----------
        kind : str
            One of `KINDS`.
        values : mapping
            The values, with the grid's names (`walk_r` for the walk).

        Returns
        -------
        RtPath
            The path.
        """
        names = (
            "amplitude",
            "peak",
            "flip_days",
            "period",
            "phase",
            "waves",
            "control_days",
            "flat",
        )
        kw = {k: float(values[k]) for k in names if values.get(k) is not None}
        return cls(
            kind=kind,
            wave_scale=tuple(_vector(values.get("wave_scale", [1.0]))),
            walk=tuple(_vector(values.get("walk_r", [0.0]))),
            **kw,
        )

    def log_r(self, n: int) -> np.ndarray:
        """Return log R on `n` days.

        Parameters
        ----------
        n : int
            Days.

        Returns
        -------
        numpy.ndarray
            Shape `(n,)`.

        Raises
        ------
        ValueError
            If the kind is unknown.
        """
        t = np.arange(n, dtype=float)
        u = t / n
        a, peak = self.amplitude, self.peak
        tilt = max(peak, 1 - peak)
        match self.kind:
            case "endemic":
                out = daily_walk(t, self.walk)
            case "controlled":
                out = a * np.exp(-t / self.control_days) + daily_walk(
                    t, self.walk
                )
            case "growth":
                out = np.full(n, a)
            case "decline":
                out = np.full(n, -a)
            case "flat":
                out = np.full(n, self.flat)
            case "rise_fall_gradual":
                out = a * (peak - u) / tilt
            case "fall_rise":
                out = -a * (peak - u) / tilt
            case "rise_fall_sharp":
                out = a * (
                    1 - 2 * _plogis((t - peak * n) / (self.flip_days / 4.4))
                )
            case "oscillating":
                out = a * np.sin(2 * np.pi * t / self.period + self.phase)
            case "multi_wave":
                scale = np.asarray(self.wave_scale)
                k = np.minimum(
                    len(scale),
                    np.floor(self.waves * u + self.phase / (2 * np.pi)) + 1,
                ).astype(int)
                out = (
                    2
                    * a
                    * scale[k - 1]
                    * np.sin(2 * np.pi * self.waves * u + self.phase)
                )
            case _:
                raise ValueError(f"unknown path {self.kind!r}")
        return np.asarray(out, dtype=float)


class WeeklyNoise(RandomVariable):
    r"""An AR(1) on the weekly scale, 0 in the first week, held for 7 days.

    $e_0 = 0$ and $e_w = 0.8 e_{w-1} + \sigma z_w$, from PyRenew's
    `ARProcess`. The standard normals $z$ are the sample site
    `"{noise_name}_decentered"`.

    Parameters
    ----------
    name : str
        Name of the component.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.ar = ARProcess(name=f"{name}_ar")

    def sample(
        self, n: int, sd: float, noise_name: str, **kwargs: object
    ) -> Array:
        r"""Return the noise on `n` days.

        Parameters
        ----------
        n : int
            Days.
        sd : float
            The innovations' sd, $\sigma$.
        noise_name : str
            The innovations' sample site.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(n,)`, constant within each week.
        """
        weeks = math.ceil(n / 7)
        e = self.ar(
            noise_name=noise_name,
            n=weeks,
            autoreg=jnp.array([_AUTOREG]),
            init_vals=jnp.array([0.0]),
            noise_sd=sd,
        )
        return jnp.repeat(jnp.asarray(e), 7)[:n]


class PlaceRt(RandomVariable):
    r"""Each stratum's log R.

    Parameters
    ----------
    name : str
        Name of the component; it prefixes the sample sites.
    path : RtPath
        The outbreak's path, the first place's under `independent`.
    relation : str
        One of `RELATIONS`.
    kinds : sequence of str
        The kinds an independent place draws its own path from, each
        equally likely (an `endemic` or `controlled` outbreak keeps its
        kind in every place).
    values : mapping
        The values each own path takes its settings from.
    noise_scale : float
        The scale of the weekly noise: places' sd is `place_sd` times it,
        strata's 0.01 times it (the generation time's mean).
    place_sd : float
        For `correlated`, the places' noise sd as a share of the scale.
    place_rho : float
        For `correlated`, the share of the places' noise variance that
        they have in common.
    place_lags : sequence of int
        For `lagged`, each place's lag in days.
    drift : float
        R's multiplicative drift over the series, $e^{\log(d) t / n}$.
    coupling : float
        Strata's own noise is scaled by `1 - coupling`.

    Raises
    ------
    ValueError
        If `relation` is not one of `RELATIONS`.
    """

    def __init__(
        self,
        name: str,
        path: RtPath,
        relation: str,
        kinds: Sequence[str] = KINDS,
        values: Mapping[str, Any] | None = None,
        noise_scale: float = 1.0,
        place_sd: float = 0.0,
        place_rho: float = 0.0,
        place_lags: Sequence[int] = (),
        drift: float = 1.0,
        coupling: float = 1.0,
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}")
        super().__init__(name=name)
        self.path = path
        self.relation = relation
        self.kinds = list(kinds)
        self.values = dict(values or {})
        self.noise_scale = noise_scale
        self.place_sd = place_sd
        self.place_rho = place_rho
        self.place_lags = [int(x) for x in place_lags]
        self.drift = drift
        self.coupling = coupling
        self.noise = WeeklyNoise(f"{name}_noise")

    def _own_kind(self, i: int) -> str:
        if self.path.kind in ("endemic", "controlled"):
            return self.path.kind
        j = numpyro.sample(
            f"{self.name}_path_{i}",
            dist.Categorical(probs=jnp.ones(len(self.kinds)) / len(self.kinds)),
        )
        return self.kinds[int(j)]

    def _place_paths(self, n: int, n_places: int) -> list[Array]:
        shared = jnp.asarray(self.path.log_r(n))
        if self.relation == "shared":
            return [shared] * n_places
        if self.relation == "independent":
            own = [
                jnp.asarray(
                    RtPath.from_values(self._own_kind(i), self.values).log_r(n)
                )
                for i in range(1, n_places)
            ]
            return [shared, *own]
        if self.relation == "correlated":
            sd = self.noise_scale * self.place_sd
            common = self.noise(n=n, sd=sd, noise_name=f"{self.name}_common")
            rho = self.place_rho
            return [
                shared
                + math.sqrt(rho) * common
                + math.sqrt(1 - rho)
                * self.noise(n=n, sd=sd, noise_name=f"{self.name}_place_{i}")
                for i in range(n_places)
            ]
        lags = self.place_lags
        if len(lags) < n_places:
            raise ValueError(
                f"lagged R paths need a lag for each of {n_places} places, "
                f"got {len(lags)}"
            )
        return [
            jnp.concatenate([jnp.full(lags[i], shared[0]), shared])[:n]
            for i in range(n_places)
        ]

    def sample(self, n: int, place: ArrayLike, **kwargs: object) -> Array:
        """Return each stratum's log R.

        Parameters
        ----------
        n : int
            Days.
        place : array_like
            Each stratum's place index (0-based).
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Days by strata.
        """
        idx = np.asarray(place, dtype=int)
        paths = self._place_paths(n, int(idx.max()) + 1)
        drift = math.log(self.drift) * jnp.arange(n) / n
        sd = 0.01 * self.noise_scale
        cols = [
            paths[p]
            + drift
            + (1 - self.coupling)
            * self.noise(n=n, sd=sd, noise_name=f"{self.name}_stratum_{j}")
            for j, p in enumerate(idx)
        ]
        out = jnp.stack(cols, axis=1)
        numpyro.deterministic(f"{self.name}::log_r", out)
        return out
