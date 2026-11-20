# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Simulated outbreaks' infections and reference events, in JAX.

The infection side of the fine-tuning simulator (layers 1 and 2 of the
grid): each stratum's log R path, mixing between places, imports, a
renewal with susceptible depletion, shifts in R and immune-escape
variants, then the infection-to-reference delay and ascertainment. It is
a port of the R engine's `infections.R` (`latent_onsets` and the
functions it calls) and gives the same numbers for the same draws.

The renewal step is not `renewal.coupled_renewal`, which the fitted
models use: the simulator mixes in counts, $F + \varepsilon(KF -
\operatorname{diag}(K\mathbf{1})F)$, records effective R after the day's
infections, and reacts to its own effective R through shifts. Keeping it
apart also keeps the fitted models' code, and so their fit keys,
unchanged.

The random draws (place and age weights, the weekly noise on log R and
the path kind of each independent place) come from a `Stream`, read in
the order the R engine makes them. `KeyStream` draws them from a JAX key;
`Tape` replays draws recorded from R.

Run as a module, it reads a batch of designs and values and writes their
latent paths (see `main`).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from scipy import stats

__all__ = [
    "DEPLETION",
    "KeyStream",
    "Latent",
    "Stream",
    "Tape",
    "convolve_rows",
    "daily_walk",
    "gt_pmf",
    "incubation_pmf",
    "latent_onsets",
    "main",
    "path_log_r",
    "place_mixing",
    "renewal_infections",
    "simulate_batch",
    "strata_log_r",
    "stratum_imports",
]

DEPLETION = ("linear", "exponential")
"""How the renewal depletes susceptibles: r7's linear form, capped at the
susceptibles, or the exponential form of `renewal.coupled_renewal`."""

# Values that are vectors, though a draw of one may arrive as a scalar.
VECTORS = (
    "population",
    "mixing_order",
    "mixing_x",
    "mixing_y",
    "place_lags",
    "import_at",
    "wave_scale",
    "walk_r",
    "shift_at",
    "shift_size",
    "shift_days",
    "shift_reactive",
    "shift_up",
    "variant_at",
    "variant_escape",
    "variant_r",
    "variant_days",
    "quiet_places",
    "late_places",
    "ascertainment",
    "incubation_mean",
)


class Stream(Protocol):
    """The random draws of a latent run, in the R engine's order."""

    def normal(self, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        """Return `n` normal draws."""
        ...

    def choice(self, options: Sequence[str]) -> str:
        """Return one of `options`, each equally likely."""
        ...


@dataclass
class KeyStream:
    """Draws from a JAX key, split for each call.

    Attributes
    ----------
    key : jax.Array
        The key.
    """

    key: jax.Array

    def _next(self) -> jax.Array:
        self.key, sub = jax.random.split(self.key)
        return sub

    def normal(self, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        """Return `n` normal draws."""
        with jax.enable_x64(True):
            z = np.asarray(
                jax.random.normal(self._next(), (n,), dtype=jnp.float64)
            )
        return mean + sd * z

    def choice(self, options: Sequence[str]) -> str:
        """Return one of `options`, each equally likely."""
        i = jax.random.randint(self._next(), (), 0, len(options))
        return options[int(i)]


@dataclass
class Tape:
    """Draws recorded from the R engine, replayed in order.

    Attributes
    ----------
    normal_draws : sequence of float
        Standard normal draws.
    choices : sequence of str
        Path kinds drawn.
    """

    normal_draws: Sequence[float] | np.ndarray
    choices: Sequence[str]
    _i: int = field(default=0, init=False)
    _j: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Box a single recorded draw, which R writes unboxed."""
        if isinstance(self.choices, str):
            self.choices = [self.choices]
        self.normal_draws = np.atleast_1d(
            np.asarray(self.normal_draws, dtype=float)
        )

    def normal(self, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        """Return the next `n` draws, scaled."""
        z = np.asarray(self.normal_draws[self._i : self._i + n], dtype=float)
        if z.size != n:
            raise ValueError("the tape has run out of normal draws")
        self._i += n
        return mean + sd * z

    def choice(self, options: Sequence[str]) -> str:
        """Return the next recorded choice."""
        out = self.choices[self._j]
        if out not in options:
            raise ValueError(f"recorded choice {out!r} is not an option")
        self._j += 1
        return out

    def finished(self) -> bool:
        """Whether every recorded draw was used."""
        return self._i == len(self.normal_draws) and self._j == len(
            self.choices
        )


def _vector(values: Mapping[str, Any], name: str) -> np.ndarray:
    x = values.get(name)
    return np.atleast_1d(np.asarray([] if x is None else x))


def _round(x: float) -> int:
    # R rounds half to even, as numpy does.
    return int(np.round(x))


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
        The pmf, n its 99.9% quantile rounded up and at least 2.
    """
    if family == "gamma":
        k = 1 / cv**2
        d = stats.gamma(k, scale=mean / k)
    else:
        sdlog = math.sqrt(math.log(1 + cv**2))
        d = stats.lognorm(sdlog, scale=math.exp(math.log(mean) - sdlog**2 / 2))
    n = max(2, math.ceil(d.ppf(0.999)))
    pmf = np.diff(d.cdf(np.arange(n + 1)))
    return np.asarray(pmf / pmf.sum(), dtype=float)


def incubation_pmf(mean: float, cv: float = 0.5) -> np.ndarray:
    """Return the infection-to-reference delay pmf on 0, 1, ... days.

    Parameters
    ----------
    mean : float
        The mean, in days.
    cv : float
        The coefficient of variation of the gamma.

    Returns
    -------
    numpy.ndarray
        The pmf to the 99.9% quantile.
    """
    shape = 1 / cv**2
    d = stats.gamma(shape, scale=mean / shape)
    n = max(1, math.ceil(d.ppf(0.999)))
    pmf = np.diff(d.cdf(np.arange(n + 1)))
    return np.asarray(pmf / pmf.sum(), dtype=float)


def convolve_rows(x: np.ndarray, pmfs: Sequence[np.ndarray]) -> np.ndarray:
    """Convolve each column of `x` (days by strata) with its own pmf.

    Parameters
    ----------
    x : numpy.ndarray
        Days by strata.
    pmfs : sequence of numpy.ndarray
        A pmf on delays 0, 1, ... for each column.

    Returns
    -------
    numpy.ndarray
        The convolution, the shape of `x`.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if not len(pmfs):
        return np.zeros(x.shape)
    with jax.enable_x64(True):
        cols = [
            jnp.convolve(jnp.asarray(x[:, j]), jnp.asarray(p[:n]), mode="full")[
                :n
            ]
            for j, p in enumerate(pmfs)
        ]
        return np.asarray(jnp.stack(cols, axis=1))


def daily_walk(days: np.ndarray, walk: np.ndarray) -> np.ndarray:
    """Read a weekly walk on `days` (0 the first), linear between weeks.

    Parameters
    ----------
    days : numpy.ndarray
        Days.
    walk : numpy.ndarray
        The walk's weekly values.

    Returns
    -------
    numpy.ndarray
        The walk on each day.
    """
    x = np.asarray(days) / 7
    w = np.floor(x).astype(int)
    f = x - w
    last = len(walk) - 1
    walk = np.asarray(walk)
    out = walk[np.minimum(w, last)] * (1 - f)
    return np.asarray(out + walk[np.minimum(w + 1, last)] * f, dtype=float)


def weekly_noise(stream: Stream, n: int, sd: float) -> np.ndarray:
    """Return an AR(1) weekly noise on `n` days, 0 in the first week.

    Parameters
    ----------
    stream : Stream
        The draws.
    n : int
        Days.
    sd : float
        The innovations' sd.

    Returns
    -------
    numpy.ndarray
        The noise, constant within each week.
    """
    weeks = math.ceil(n / 7)
    e = np.zeros(weeks)
    for w in range(1, weeks):
        e[w] = 0.8 * e[w - 1] + stream.normal(1, 0.0, sd)[0]
    return np.repeat(e, 7)[:n]


def path_log_r(n: int, kind: str, values: Mapping[str, Any]) -> np.ndarray:
    """Return the log R of a path on `n` days.

    Parameters
    ----------
    n : int
        Days.
    kind : str
        The path: r6's shapes on log R, `endemic` (the weekly walk around
        R = 1) or `controlled` (a start at the amplitude decaying onto
        that walk).
    values : mapping
        The simulation's values.

    Returns
    -------
    numpy.ndarray
        Log R on each day.
    """
    return np.asarray(_shape(n, kind, values), dtype=float)


def _shape(n: int, kind: str, values: Mapping[str, Any]) -> Any:
    v = values
    t = np.arange(n, dtype=float)
    u = t / n
    a = v["amplitude"]
    peak = v["peak"]
    if kind == "endemic":
        return daily_walk(t, _vector(v, "walk_r"))
    if kind == "controlled":
        return a * np.exp(-t / v["control_days"]) + daily_walk(
            t, _vector(v, "walk_r")
        )
    if kind == "growth":
        return np.full(n, a)
    if kind == "decline":
        return np.full(n, -a)
    if kind == "flat":
        return np.full(n, v["flat"])
    if kind == "rise_fall_gradual":
        return a * (peak - u) / max(peak, 1 - peak)
    if kind == "fall_rise":
        return -a * (peak - u) / max(peak, 1 - peak)
    if kind == "rise_fall_sharp":
        return a * (1 - 2 * _plogis((t - peak * n) / (v["flip_days"] / 4.4)))
    if kind == "oscillating":
        return a * np.sin(2 * np.pi * t / v["period"] + v["phase"])
    if kind == "multi_wave":
        scale = _vector(v, "wave_scale")
        waves = v["waves"]
        k = np.minimum(
            len(scale), np.floor(waves * u + v["phase"] / (2 * np.pi)) + 1
        ).astype(int)
        return 2 * a * scale[k - 1] * np.sin(2 * np.pi * waves * u + v["phase"])
    raise ValueError(f"unknown path {kind!r}")


def _plogis(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-np.asarray(x, dtype=float))))


def _places(strata: pd.DataFrame) -> list[str]:
    return sorted(strata["place"].unique())


def strata_log_r(
    grid: Mapping[str, Any],
    n: int,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    stream: Stream,
) -> np.ndarray:
    """Return each stratum's log R on `n` days.

    Places share the path, follow their own, add correlated weekly noise
    to it or lag it, as `design["place_rt"]` says. A slow drift and
    weekly noise scaled by `1 - coupling` are added to every stratum.

    Parameters
    ----------
    grid : mapping
        The grid config.
    n : int
        Days.
    strata : pandas.DataFrame
        The strata, with `place` and `age`, in epinowcast's group order.
    design : mapping
        The simulation's design.
    values : mapping
        The simulation's values.
    stream : Stream
        The draws.

    Returns
    -------
    numpy.ndarray
        Days by strata.
    """
    v = values
    places = _places(strata)
    scale = v["gt_mean"]
    shared = path_log_r(n, v["path"], v)
    kinds = list(grid["factors"]["trajectory"])

    def own_path() -> str:
        if v["path"] in ("endemic", "controlled"):
            return str(v["path"])
        return stream.choice(kinds)

    relation = design["place_rt"]
    if relation == "shared":
        paths = [shared] * len(places)
    elif relation == "independent":
        paths = [shared] + [path_log_r(n, own_path(), v) for _ in places[1:]]
    elif relation == "correlated":
        common = weekly_noise(stream, n, scale * v["place_sd"])
        rho = v["place_rho"]
        paths = [
            shared
            + math.sqrt(rho) * common
            + math.sqrt(1 - rho)
            * weekly_noise(stream, n, scale * v["place_sd"])
            for _ in places
        ]
    elif relation == "lagged":
        lags = _vector(v, "place_lags").astype(int)
        paths = [
            np.concatenate([np.full(lags[i], shared[0]), shared])[:n]
            for i in range(len(places))
        ]
    else:
        raise ValueError(f"unknown place_rt {relation!r}")
    path = [paths[places.index(p)] for p in strata["place"]]
    drift = math.log(v["r_drift"]) * np.arange(n) / n
    cols = [
        path[i]
        + drift
        + (1 - v["coupling"]) * weekly_noise(stream, n, 0.01 * scale)
        for i in range(len(strata))
    ]
    return np.stack(cols, axis=1)


def place_mixing(
    n: int,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mixing kernel between strata and its share on each day.

    Each age mixes with the same age elsewhere. Kinds: `none`,
    `symmetric`, `asymmetric` (only from places earlier in
    `mixing_order`), `distance` (decaying over `mixing_range`) and
    `time_varying` (the distance kernel switching on or off). Places
    without cases neither give nor take infections.

    Parameters
    ----------
    n : int
        Days.
    strata : pandas.DataFrame
        The strata.
    design : mapping
        The simulation's design.
    values : mapping
        The simulation's values.

    Returns
    -------
    tuple of numpy.ndarray
        The row-normalised kernel `K` (strata by strata, rows receive)
        and the share `eps` on each day.
    """
    v = values
    places = _places(strata)
    n_p = len(places)
    kind = design["mixing"]
    kernel = np.zeros((n_p, n_p))
    if n_p > 1 and kind != "none":
        if kind == "symmetric":
            kernel = 1 - np.eye(n_p)
        elif kind == "asymmetric":
            drawn = _vector(v, "mixing_order").astype(int)
            order = [int(i) for i in drawn[drawn <= n_p]]
            at = np.array([order.index(i) for i in range(1, n_p + 1)])
            kernel = (at[:, None] > at[None, :]).astype(float)
        else:
            xy = np.stack(
                [_vector(v, "mixing_x")[:n_p], _vector(v, "mixing_y")[:n_p]],
                axis=1,
            ).astype(float)
            dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            kernel = np.exp(-dist / v["mixing_range"]) * (1 - np.eye(n_p))
        quiet = _vector(v, "quiet_places").astype(int)
        quiet = quiet[quiet <= n_p] - 1
        kernel[quiet, :] = 0
        kernel[:, quiet] = 0
        rows = kernel.sum(axis=1)
        give = rows > 0
        kernel[give] = kernel[give] / rows[give, None]
    idx = np.array([places.index(p) for p in strata["place"]])
    ages = strata["age"].to_numpy()
    K = kernel[np.ix_(idx, idx)] * (ages[:, None] == ages[None, :])
    if kind == "none":
        eps = np.zeros(n)
    elif kind == "time_varying":
        start = _round(v["mixing_switch_at"] * n)
        ramp = np.clip(
            (np.arange(1, n + 1) - start) / v["mixing_switch_days"], 0, 1
        )
        eps = v["mixing_eps"] * (
            ramp if v["mixing_switch"] == "on" else 1 - ramp
        )
    else:
        eps = np.full(n, float(v["mixing_eps"]))
    return K, eps


def _late(
    strata: pd.DataFrame, design: Mapping[str, Any], values: Mapping[str, Any]
) -> np.ndarray:
    late = {f"P{i:02d}" for i in _vector(values, "late_places").astype(int)}
    return np.asarray(
        (design["seeding"] == "mid") & strata["place"].isin(late), dtype=bool
    )


def stratum_imports(
    n: int,
    share: np.ndarray,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
) -> np.ndarray:
    """Return the imports into each stratum on `n` days.

    A late place is imported its share over `late_days` from a share
    `late_at` of the way through. Other strata with cases are imported a
    constant `import_rate` of their share a day, or pulses of
    `import_size` times it at shares `import_at` of the way through.

    Parameters
    ----------
    n : int
        Days.
    share : numpy.ndarray
        Each stratum's unit, the seed or level times its weight.
    strata : pandas.DataFrame
        The strata.
    design : mapping
        The simulation's design.
    values : mapping
        The simulation's values.

    Returns
    -------
    numpy.ndarray
        Days by strata.
    """
    v = values
    share = np.asarray(share, dtype=float)
    out = np.zeros((n, len(strata)))
    late = _late(strata, design, v)
    if late.any():
        first = max(1, _round(v["late_at"] * n))
        days = np.arange(first, min(n, first + int(v["late_days"]) - 1) + 1) - 1
        out[np.ix_(days, np.flatnonzero(late))] = share[late] / len(days)
    on = ~late & (share > 0)
    if design["imports"] == "constant":
        out[:, on] = v["import_rate"] * share[on]
    elif design["imports"] == "pulses":
        at = np.maximum(1, np.round(_vector(v, "import_at") * n)).astype(int)
        days = np.unique(at) - 1
        out[np.ix_(days, np.flatnonzero(on))] += v["import_size"] * share[on]
    return out


def renewal_infections(
    start: np.ndarray,
    gt: np.ndarray,
    log_r: np.ndarray,
    population: np.ndarray,
    mixing: tuple[np.ndarray, np.ndarray] | None = None,
    imports: np.ndarray | None = None,
    shifts: Mapping[str, np.ndarray] | None = None,
    variants: Mapping[str, np.ndarray] | None = None,
    depletion: str = "linear",
) -> dict[str, np.ndarray]:
    r"""Run each stratum's renewal with depletion, shifts and variants.

    Each day, with $F$ the generation-time-weighted past infections,

    $$\lambda_t = R_t \frac{S_t}{N} \{F + \varepsilon_t (K F -
    \operatorname{diag}(K\mathbf{1}) F)\} + \iota_t,$$

    at most $S_t$ (`depletion="linear"`, the R engine), or with
    $S_t \{1 - e^{-R_t F^*_t / N}\}$ in place of $R_t (S_t / N) F^*_t$, $F^*$
    the mixed pressure (`"exponential"`, as `renewal.coupled_renewal`).

    - Shift $j$ moves R by $e^{\pm s_j}$ along a logistic over `days[j]`
      centred on `at[j]`. Its direction is set on the first day at or
      after `at[j] - days[j] / 2`: a reactive shift lowers R where the
      previous day's effective R is at least 1 and raises it where below.
      The others go `up` or down.
    - Variant $j$ raises R by `mult[j]` along a logistic over `days[j]`
      centred on `at[j]`, and on the first day at or after `at[j]` returns
      a share `escape[j]` of the immune to the susceptibles.

    Parameters
    ----------
    start : numpy.ndarray
        Infections over the generation time's span before day 1, `[G, k]`.
    gt : numpy.ndarray
        Generation time pmf on 1, ..., G days.
    log_r : numpy.ndarray
        Log R, `[n, k]`.
    population : numpy.ndarray
        Each stratum's population, `[k]`.
    mixing : tuple of numpy.ndarray, optional
        The kernel `K` `[k, k]` and its share `eps` `[n]`.
    imports : numpy.ndarray, optional
        Imports, `[n, k]`.
    shifts : mapping, optional
        `at`, `size`, `days`, `reactive` and `up`, one entry per shift.
    variants : mapping, optional
        `at`, `escape`, `mult` and `days`, one entry per variant.
    depletion : str
        One of `DEPLETION`.

    Returns
    -------
    dict of numpy.ndarray
        `infections` `[G + n, k]` (with `start`), and `r`, `reff` and
        `susceptible` after each day, `[n, k]`.
    """
    if depletion not in DEPLETION:
        raise ValueError(f"depletion must be one of {DEPLETION}")
    with jax.enable_x64(True):
        return _renewal(
            start,
            gt,
            log_r,
            population,
            mixing,
            imports,
            shifts,
            variants,
            depletion,
        )


def _events(
    spec: Mapping[str, Any] | None, names: Sequence[str]
) -> dict[str, jax.Array]:
    spec = spec or {}
    return {
        k: jnp.atleast_1d(jnp.asarray(spec.get(k, []), dtype=float))
        for k in names
    }


def _renewal(
    start: np.ndarray,
    gt: np.ndarray,
    log_r: np.ndarray,
    population: np.ndarray,
    mixing: tuple[np.ndarray, np.ndarray] | None,
    imports: np.ndarray | None,
    shifts: Mapping[str, Any] | None,
    variants: Mapping[str, Any] | None,
    depletion: str,
) -> dict[str, np.ndarray]:
    start_ = jnp.asarray(start, dtype=float)
    log_r_ = jnp.asarray(log_r, dtype=float)
    pop = jnp.asarray(population, dtype=float)
    n, k = log_r_.shape
    rev_gt = jnp.flip(jnp.asarray(gt, dtype=float))
    if mixing is None:
        K, eps = jnp.zeros((k, k)), jnp.zeros(n)
    else:
        K, eps = (
            jnp.asarray(mixing[0], dtype=float),
            jnp.asarray(mixing[1], dtype=float),
        )
    give = K.sum(axis=1)
    imp = (
        jnp.zeros((n, k))
        if imports is None
        else jnp.asarray(imports, dtype=float)
    )
    sh = _events(shifts, ("at", "size", "days", "reactive", "up"))
    va = _events(variants, ("at", "escape", "mult", "days"))
    n_sh, n_va = sh["at"].shape[0], va["at"].shape[0]
    days = jnp.arange(1, n + 1, dtype=float)
    # Variants' R multipliers depend only on the day.
    va_mult = jnp.prod(
        1
        + (va["mult"][None, :] - 1)
        * jax.nn.sigmoid(
            (days[:, None] - va["at"][None, :]) / (va["days"][None, :] / 4.4)
        ),
        axis=1,
    )
    s0 = jnp.maximum(pop - start_.sum(axis=0), 0)

    def step(carry: Any, inputs: Any) -> tuple[Any, tuple[jax.Array, ...]]:
        hist, s, prev, mult, set_, escaped = carry
        t, lr, e, im, vm = inputs
        r = jnp.exp(lr)
        # Shifts: the direction is fixed on the first day it is due.
        due = (~set_) & (t >= sh["at"] - sh["days"] / 2)
        up = jnp.where(
            sh["reactive"][:, None] > 0,
            prev[None, :] < 1,
            (sh["up"][:, None] > 0) & jnp.ones((n_sh, k), dtype=bool),
        )
        new_mult = jnp.exp(jnp.where(up, 1.0, -1.0) * sh["size"][:, None])
        mult = jnp.where(due[:, None], new_mult, mult)
        set_ = set_ | due
        weight = jax.nn.sigmoid((t - sh["at"]) / (sh["days"] / 4.4))
        factor = jnp.where(set_[:, None], 1 + (mult - 1) * weight[:, None], 1.0)
        r = r * jnp.prod(factor, axis=0)
        # Variants: escape once, on the first day at or after `at`.
        go = (~escaped) & (t >= va["at"])
        for j in range(n_va):
            s = jnp.where(go[j], s + va["escape"][j] * (pop - s), s)
        escaped = escaped | go
        r = r * vm
        force = rev_gt @ hist
        force = force + jnp.where(e > 0, e * (K @ force - give * force), 0.0)
        if depletion == "linear":
            new = r * s / pop * force + im
            new = jnp.minimum(new, s)
        else:
            # Per capita as `renewal.coupled_renewal`: S (1 - e^{-R F / N}),
            # which is R S F / N to first order.
            new = jnp.minimum(-s * jnp.expm1(-r * force / pop) + im, s)
        s = s - new
        reff = r * s / pop
        hist = jnp.concatenate([hist[1:], new[None, :]])
        return (hist, s, reff, mult, set_, escaped), (new, r, reff, s)

    carry = (
        start_,
        s0,
        jnp.ones(k),
        jnp.ones((n_sh, k)),
        jnp.zeros(n_sh, dtype=bool),
        jnp.zeros(n_va, dtype=bool),
    )
    _, (new, r, reff, s) = jax.lax.scan(
        step, carry, (days, log_r_, eps, imp, va_mult)
    )
    return {
        "infections": np.asarray(jnp.concatenate([start_, new])),
        "r": np.asarray(r),
        "reff": np.asarray(reff),
        "susceptible": np.asarray(s),
    }


@dataclass
class Latent:
    """A simulation's latent run.

    Attributes
    ----------
    onsets : numpy.ndarray
        Expected reference events (ascertained), days by strata.
    infections : numpy.ndarray
        Infections, days by strata.
    r, reff : numpy.ndarray
        Each stratum's R and effective R, NaN over the generation time's
        span, days by strata.
    path : pandas.DataFrame
        The whole outbreak by `day`: `infections`, `onsets`, and `r` and
        `reff` weighted by each day's infections (by population on days
        with none).
    stratum_population : numpy.ndarray
        Each stratum's population.
    attack : float
        Infections over the population of the strata with cases.
    n_gt : int
        Days in the generation time's span.
    """

    onsets: np.ndarray
    infections: np.ndarray
    r: np.ndarray
    reff: np.ndarray
    path: pd.DataFrame
    stratum_population: np.ndarray
    attack: float
    n_gt: int


def latent_onsets(
    grid: Mapping[str, Any],
    n: int,
    gt: np.ndarray,
    strata: pd.DataFrame,
    design: Mapping[str, Any],
    values: Mapping[str, Any],
    stream: Stream,
    depletion: str = "linear",
) -> Latent:
    """Return a simulation's expected reference events.

    Strata weights are log-normal; quiet places have none, and each
    place's population is split over its ages by their weights. An early
    outbreak starts from a seed on the last day before day 1 (at least
    1); an endemic or established one runs at the drawn level over the
    generation time's span. Late places, imports and a trickle into every
    stratum with cases come in as imports.

    Parameters
    ----------
    grid : mapping
        The grid config.
    n : int
        Days after the generation time's span.
    gt : numpy.ndarray
        Generation time pmf (`gt_pmf`).
    strata : pandas.DataFrame
        The strata, with `place`, `age` and `.group`.
    design : mapping
        The simulation's design.
    values : mapping
        The simulation's values (layers 1 and 2).
    stream : Stream
        The draws.
    depletion : str
        One of `DEPLETION`.

    Returns
    -------
    Latent
        The run.

    Raises
    ------
    ValueError
        If the generation time is longer than the series, or an
        established epidemic is larger than its population.
    """
    if n < 1:
        raise ValueError("the generation time is longer than the series")
    v = values
    n_places, n_ages = int(design["n_places"]), int(design["n_ages"])
    place_ids = [f"P{i:02d}" for i in range(1, n_places + 1)]
    age_ids = [f"A{i}" for i in range(1, n_ages + 1)]
    sd = v["stratum_sd"]
    place_weight = np.exp(stream.normal(n_places, 0.0, sd))
    quiet = _vector(v, "quiet_places").astype(int)
    place_weight[quiet - 1] = 0
    age_weight = np.exp(stream.normal(n_ages, 0.0, sd / 2))
    at_place = np.array([place_ids.index(p) for p in strata["place"]])
    at_age = np.array([age_ids.index(a) for a in strata["age"]])
    weight = place_weight[at_place] * age_weight[at_age]
    active = weight > 0
    rel = weight / weight[active].mean()
    population = (
        _vector(v, "population").astype(float)[at_place]
        * (age_weight / age_weight.sum())[at_age]
    )
    late = _late(strata, design, v)
    on = active & ~late
    n_gt = len(gt)
    start = np.zeros((n_gt, len(strata)))
    early = design["window"] == "early"
    if early:
        start[n_gt - 1, on] = np.maximum(1, v["seed"] * rel[on])
    else:
        start[:, on] = v["level"] * rel[on]
    if (start.sum(axis=0) > population).any():
        raise ValueError("an established epidemic larger than its population")
    unit = (v["seed"] if early else v["level"]) * rel
    imports = stratum_imports(n, unit, strata, design, v)
    imports[:, on] += v["trickle"]
    shifts = {
        "at": _vector(v, "shift_at") * n,
        "size": _vector(v, "shift_size"),
        "days": _vector(v, "shift_days"),
        "reactive": _vector(v, "shift_reactive"),
        "up": _vector(v, "shift_up"),
    }
    variants = {
        "at": _vector(v, "variant_at") * n,
        "escape": _vector(v, "variant_escape"),
        "mult": _vector(v, "variant_r"),
        "days": _vector(v, "variant_days"),
    }
    log_r = strata_log_r(grid, n, strata, design, v, stream)
    run = renewal_infections(
        start,
        gt,
        log_r,
        population,
        mixing=place_mixing(n, strata, design, v),
        imports=imports,
        shifts=shifts,
        variants=variants,
        depletion=depletion,
    )
    infections = run["infections"]
    attack = float(infections[:, active].sum() / population[active].sum())
    group = strata[".group"].to_numpy().astype(int) - 1
    asc = _vector(v, "ascertainment").astype(float)[group]
    cv = grid["constants"]["incubation_cv"]
    pmfs = [
        incubation_pmf(m, cv)
        for m in _vector(v, "incubation_mean").astype(float)[group]
    ]
    onsets = convolve_rows(infections, pmfs) * asc
    inf = infections[n_gt:]
    total = inf.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = inf / total
    none = ~np.isfinite(w.sum(axis=1))
    w[none] = population / population.sum()
    pad = np.full(n_gt, np.nan)
    path = pd.DataFrame(
        {
            "day": np.arange(n_gt + n),
            "infections": infections.sum(axis=1),
            "onsets": onsets.sum(axis=1),
            "r": np.concatenate([pad, (run["r"] * w).sum(axis=1)]),
            "reff": np.concatenate([pad, (run["reff"] * w).sum(axis=1)]),
        }
    )
    pad2 = np.full((n_gt, len(strata)), np.nan)
    return Latent(
        onsets=onsets,
        infections=infections,
        r=np.concatenate([pad2, run["r"]]),
        reff=np.concatenate([pad2, run["reff"]]),
        path=path,
        stratum_population=population,
        attack=attack,
        n_gt=n_gt,
    )


def _long(sim_id: str, latent: Latent) -> pd.DataFrame:
    days, k = latent.onsets.shape
    strata = pd.DataFrame(
        {
            "id": sim_id,
            "day": np.tile(np.arange(days), k),
            "group": np.repeat(np.arange(1, k + 1), days),
            "infections": latent.infections.T.ravel(),
            "onsets": latent.onsets.T.ravel(),
            "r": latent.r.T.ravel(),
            "reff": latent.reff.T.ravel(),
        }
    )
    whole = latent.path.assign(id=sim_id, group=0)
    return pd.concat([whole[strata.columns], strata], ignore_index=True)


def simulate_batch(
    grid: Mapping[str, Any],
    draws: Mapping[str, Any],
    depletion: str = "linear",
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the latent side of a batch of simulations.

    Parameters
    ----------
    grid : mapping
        The grid config.
    draws : mapping
        `simulations`, each with `id`, `key`, `n_days`, `design`, `values`
        and `strata` (records with `place`, `age` and `.group`).
    depletion : str
        One of `DEPLETION`.

    Returns
    -------
    tuple
        The latent paths, long (`id`, `day`, `group`, `infections`,
        `onsets`, `r`, `reff`; group 0 the whole outbreak), and each
        simulation's `n_gt`, generation time pmf `gt`,
        `stratum_population`, `attack` and `error` (None, or why it has
        no run: any error in one simulation is recorded, as the R batch
        redraws on any error).
    """
    frames: list[pd.DataFrame] = []
    out: dict[str, dict[str, Any]] = {}
    for sim in draws["simulations"]:
        sim_id = str(sim["id"])
        design, values = sim["design"], sim["values"]
        try:
            gt = gt_pmf(values["gt_mean"], design["gt_family"], values["gt_cv"])
            latent = latent_onsets(
                grid,
                int(sim["n_days"]) - len(gt),
                gt,
                pd.DataFrame(sim["strata"]),
                design,
                values,
                KeyStream(jax.random.key(int(sim["key"]))),
                depletion,
            )
        except Exception as e:
            # As the R batch, which redraws a simulation on any error.
            out[sim_id] = {
                "n_gt": None,
                "gt": None,
                "stratum_population": None,
                "attack": None,
                "error": str(e)
                if type(e) is ValueError
                else f"{type(e).__name__}: {e}",
            }
            continue
        frames.append(_long(sim_id, latent))
        out[sim_id] = {
            "n_gt": latent.n_gt,
            "gt": gt.tolist(),
            "stratum_population": latent.stratum_population.tolist(),
            "attack": latent.attack,
            "error": None,
        }
    cols = ["id", "day", "group", "infections", "onsets", "r", "reff"]
    long = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=cols)
    )
    return long, out


def main(argv: Sequence[str] | None = None) -> None:
    """Write a batch's latent paths from its draws.

    `--grid` is the grid config and `--draws` the batch's draws (JSON, as
    `simulate_batch` takes). Writes `latent-<seed>.csv.gz` and
    `latent-values-<seed>.json` to `--out`, with the draws' `seed`.
    """
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--grid", required=True)
    p.add_argument("--draws", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depletion", choices=DEPLETION, default="linear")
    a = p.parse_args(argv)
    grid = yaml.safe_load(Path(a.grid).read_text())
    draws = json.loads(Path(a.draws).read_text())
    long, values = simulate_batch(grid, draws, a.depletion)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = draws["seed"]
    long.to_csv(
        out / f"latent-{seed}.csv.gz", index=False, float_format="%.17g"
    )
    (out / f"latent-values-{seed}.json").write_text(json.dumps(values))


if __name__ == "__main__":
    main()
