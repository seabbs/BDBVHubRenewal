# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A generative random walk nowcast with partial pooling.

`PPRandomWalkNowcast` (`pprandomwalknowcast`) is one joint model of the
expected final counts and the reporting delay, in the style of
epinowcast, fitted with NUTS in numpyro to the hub's release snapshots.
It is `randomwalknowcast`'s random walk on the log scale, with the
nowcast made by the same model rather than by a chain ladder, and with
the delay, the walk spread and the dispersion partially pooled over
zones, provinces and the country.

Each place $p$ the task forecasts is a leaf of the place hierarchy.
Its expected final count in hub week $w$ is $\lambda_{p,w}$, a random
walk on the log scale from the first fitted week,

$$
\log \lambda_{p,w} = \log \lambda_{p,w-1} + \sigma_p \varepsilon_{p,w},
\qquad \varepsilon_{p,w} \sim \mathcal{N}(0, 1).
$$

Each onset day $t$ of week $w(t)$ expects $\lambda_{p,w(t)} / 7$ cases.
A case is reported $d$ days after onset, with $d$ a log-normal delay
($\mu_p$, $s_p$) cut to whole days and capped at $D$, so the share of a
day's final count reported by age $a$ is

$$
\pi_p(a) = \frac{F_p(a + 1)}{F_p(D + 1)}, \quad 0 \le a \le D,
$$

with $F_p$ the log-normal CDF, $\pi_p(a) = 1$ for $a \ge D$ and
$\pi_p(-1) = 0$.
Release $r$ has cut-off $c_r$, and day $t$ is $a_r(t) = c_r - t$ days old
there.
The new reports of day $t$ between the previous release that observes it
and release $r$, or all reports by $r$ for the first release that does,
are

$$
n_{p,r}(t) \sim \text{NB}\Big(
  \tfrac{\lambda_{p,w(t)}}{7}\,
  \big\{\pi_p(a_r(t)) - \pi_p(a_{r-1}(t))\big\},\ \phi_p
\Big),
$$

so the data are right truncated by construction (`triangle`).
Counts first go through `chain`'s preprocessing of downward revisions,
baselinenowcast's `preprocess_negative_values`, so no increment is
negative.
Reports of a day after it was $D$ days old at an earlier release have no
probability under the model and are left out, as the chain ladder of
`randomwalknowcast` leaves out growth after $D$.

Each of $\mu$, $\log s$, $\log \sigma$ and $\log \phi$ of a place is its
parent's plus a level offset (`Tree.sample`),

$$
\theta_n = \theta_{\mathrm{pa}(n)} + \tau_{\ell(n)} z_n,
\qquad z_n \sim \mathcal{N}(0, 1),
\qquad \tau_\ell \sim \mathcal{N}^+(0, 0.5^2).
$$

A parent's counts are the sum of its children's, so only the leaves
enter the likelihood and their ancestors are latent centres.

The forecast takes one posterior draw for each sample.
A week's final count is, for each of its days up to the cut-off, the
count reported plus an NB draw of the rest, with mean
$\lambda/7 \cdot \{1 - \pi(a_o)\}$, and for each later day an NB draw with
mean $\lambda/7$.
Weeks after the latest data week continue the walk from that draw's
$\lambda$ with its $\sigma_p$, with counts NB with mean $\lambda$ and size
$7 \phi_p$, the sum of seven days.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax.scipy.special import ndtr
from numpyro.diagnostics import summary
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.reparam import LocScaleReparam

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.snapshots import (
    SnapshotArrays,
    snapshot_arrays,
    week_sum,
    weekly_sums,
)
from bdbvhubrenewal.forecasters.base import (
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from bdbvhubrenewal.forecasters.chain import (
    MissingSnapshots,
    NoDelayPair,
    Settings,
    _delay_counts,
    _pairs_at,
)
from bdbvhubrenewal.forecasters.introductions import (
    introduction_rates,
    silent_places,
    silent_trajectories,
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalkNowcast,
    _clear_caches,
    frame,
)

__all__ = [
    "NoRecentCases",
    "NotConverged",
    "PPRandomWalkNowcast",
    "Posterior",
    "Sampler",
    "Tree",
    "Triangle",
    "fit_posterior",
    "reported_share",
    "triangle",
]

FIT_WEEKS = 16
"""Hub weeks, ending at the cut-off week, that the walk is fitted over.

These cover `randomwalknowcast`'s nine-week spread window and the six
weeks a delay of 42 days reaches back.
"""


CENTRING = 0.5
"""Centring of the walk steps, between 0 (non-centred) and 1 (centred).

A large place's weekly counts pin its walk, which suits centred steps,
and a place with few cases leaves it to the prior, which suits
non-centred ones.
Halfway gave 1 divergence in 1500 draws at 11-08 where non-centred
steps gave 12.
"""


class NotConverged(ModelUnavailableError):  # noqa: N818
    """The NUTS fit did not converge."""


class NoRecentCases(ModelUnavailableError):  # noqa: N818
    """No place has a case in the latest `walk_weeks` weeks."""


@dataclass(frozen=True)
class Sampler:
    """NUTS settings and the convergence check.

    Attributes
    ----------
    num_warmup, num_samples, num_chains : int
        Warm-up and kept draws per chain, and chains, run in sequence.
    target_accept : float
        Target acceptance.
    max_rhat : float
        Largest split R-hat over parameters before the fit raises.
    max_divergent : float
        Largest share of divergent draws before the fit raises.
    """

    num_warmup: int = 500
    num_samples: int = 750
    num_chains: int = 2
    target_accept: float = 0.9
    max_rhat: float = 1.05
    max_divergent: float = 0.01


@dataclass(frozen=True)
class Tree:
    """The place hierarchy of a snapshot, for the pooled parameters.

    Attributes
    ----------
    parent : numpy.ndarray
        Parent of each place, -1 for the country, `[L]`.
    level : numpy.ndarray
        Depth of each place, 0 for the country, `[L]`.
    path : numpy.ndarray
        `[L, L]`, 1 where column $m$ is place $n$ or one of its
        ancestors other than the country.
    leaf : numpy.ndarray
        Places with no children, `[L]`.
    """

    parent: np.ndarray
    level: np.ndarray
    path: np.ndarray
    leaf: np.ndarray

    @classmethod
    def of(cls, parent: np.ndarray) -> Tree:
        """Build the tree from each place's parent index.

        Parameters
        ----------
        parent : numpy.ndarray
            Parent of each place, -1 for the country, `[L]`.

        Returns
        -------
        Tree
            The tree.
        """
        parent = np.asarray(parent, dtype=int)
        n = parent.size
        level = np.zeros(n, dtype=int)
        path = np.zeros((n, n))
        for node in range(n):
            m = node
            while parent[m] >= 0:
                path[node, m] = 1.0
                level[node] += 1
                m = parent[m]
        leaf = ~np.isin(np.arange(n), parent)
        return cls(parent, level, path, leaf)

    def sample(self, name: str, loc: float, scale: float) -> jax.Array:
        r"""Sample a pooled parameter for every place, non-centred.

        Parameters
        ----------
        name : str
            Site prefix.
        loc, scale : float
            Normal prior of the country's value.

        Returns
        -------
        jax.Array
            `[L]`.
        """
        root = numpyro.sample(f"{name}_root", dist.Normal(loc, scale))
        n_lvl = max(int(self.level.max()), 1)
        tau = numpyro.sample(
            f"{name}_tau", dist.HalfNormal(0.5).expand((n_lvl,)).to_event()
        )
        z = numpyro.sample(
            f"{name}_z",
            dist.Normal(0.0, 1.0).expand((self.level.size,)).to_event(),
        )
        step = tau[jnp.maximum(self.level - 1, 0)] * (self.level > 0) * z
        out: jax.Array = root + jnp.asarray(self.path) @ step
        return out

    def rebuild(self, draws: dict[str, np.ndarray], name: str) -> np.ndarray:
        """Return a pooled parameter of every place from its draws.

        Parameters
        ----------
        draws : dict of numpy.ndarray
            Posterior draws by site, flat over chains.
        name : str
            Site prefix, as for `sample`.

        Returns
        -------
        numpy.ndarray
            `[N, L]`, one row per draw.
        """
        root, tau, z = (draws[f"{name}_{p}"] for p in ("root", "tau", "z"))
        lvl = np.maximum(self.level - 1, 0)
        step = tau[:, lvl] * (self.level > 0) * z
        return np.asarray(root[:, None] + step @ self.path.T)


@dataclass(frozen=True)
class Triangle:
    """The reporting triangle by release of the leaves, as flat cells.

    Attributes
    ----------
    leaves : numpy.ndarray
        Snapshot index of each leaf, `[P]`.
    first : int
        Hub week index of the first fitted week.
    n_weeks : int
        Fitted weeks, from `first` to the origin's cut-off week.
    place, week : numpy.ndarray
        Leaf position and fitted week of each cell, `[C]`.
    lo, hi : numpy.ndarray
        Age of the cell's day at the previous release that observes it,
        -1 for none, and at the cell's release, capped at $D$, `[C]`.
    count : numpy.ndarray
        New reports in the cell, `[C]`.
    """

    leaves: np.ndarray
    first: int
    n_weeks: int
    place: np.ndarray
    week: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    count: np.ndarray


def triangle(
    snap: SnapshotArrays,
    o: int,
    settings: Settings,
    fit_weeks: int,
    leaves: np.ndarray | None = None,
) -> Triangle:
    """Return the new reports of each leaf, day and release up to `o`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release. Later releases are not read.
    settings : Settings
        Settings, for $D$ (`max_delay`).
    fit_weeks : int
        Hub weeks fitted, ending at the cut-off week of `o`.
    leaves : numpy.ndarray, optional
        Snapshot index of each place fitted. Defaults to every leaf.

    Returns
    -------
    Triangle
        One cell for each leaf, day in the fitted weeks and release that
        observes it, except where the day was already $D$ days old at
        the previous release.
    """
    d = settings.max_delay
    n_day = snap.week.size
    counts = _delay_counts(snap, o)[:, : o + 1]
    seen = snap.observed[: o + 1]
    age = snap.cut[: o + 1, None] - np.arange(n_day)[None, :]
    before = np.vstack([np.zeros((1, n_day), bool), seen[:-1]])
    lo = np.where(before, np.vstack([np.full((1, n_day), -1), age[:-1]]), -1)
    hi = np.minimum(age, d)
    prev = np.concatenate([np.zeros_like(counts[:, :1]), counts[:, :-1]], 1)
    inc = counts - prev * before[None]
    first = max(snap.cut_week(o) - fit_weeks + 1, 0)
    keep = seen & (lo < d) & (lo < hi) & (snap.week >= first)[None, :]
    if leaves is None:
        leaves = np.flatnonzero(Tree.of(snap.parent).leaf)
    leaves = np.asarray(leaves, dtype=int)
    r_i, t_i = np.nonzero(keep)
    n_cell = r_i.size
    place = np.repeat(np.arange(leaves.size), n_cell)
    rr, tt = np.tile(r_i, leaves.size), np.tile(t_i, leaves.size)
    return Triangle(
        leaves=leaves,
        first=first,
        n_weeks=snap.cut_week(o) - first + 1,
        place=place,
        week=snap.week[tt] - first,
        lo=lo[rr, tt],
        hi=hi[rr, tt],
        count=np.round(inc[leaves[place], rr, tt]),
    )


def reported_share(mu: Any, sd: Any, age: Any, max_delay: int) -> jax.Array:
    r"""Return $\pi(a)$, the share of a day's final count reported by age.

    Parameters
    ----------
    mu, sd : array_like
        Log-normal location and scale of the delay in days.
    age : array_like
        Ages $a$, broadcast against `mu` and `sd`.
    max_delay : int
        $D$.

    Returns
    -------
    jax.Array
        $F(a + 1) / F(D + 1)$, 0 for $a < 0$ and 1 for $a \ge D$.
    """
    age = jnp.asarray(age)
    a = jnp.clip(age, 0, max_delay)
    z = (jnp.log1p(a) - mu) / sd
    z_d = (jnp.log1p(max_delay) - mu) / sd
    pi = ndtr(z) / ndtr(z_d)
    return jnp.where(age < 0, 0.0, jnp.where(age >= max_delay, 1.0, pi))


def _interval(
    mu: jax.Array, sd: jax.Array, lo: Any, hi: Any, max_delay: int
) -> jax.Array:
    r"""Return $\pi(\mathrm{hi}) - \pi(\mathrm{lo})$, accurate in both tails."""
    z_lo = (jnp.log1p(jnp.clip(lo, 0, max_delay)) - mu) / sd
    z_hi = (jnp.log1p(jnp.minimum(hi, max_delay)) - mu) / sd
    z_d = (jnp.log1p(max_delay) - mu) / sd
    mid = jnp.where(
        z_lo > 0, ndtr(-z_lo) - ndtr(-z_hi), ndtr(z_hi) - ndtr(z_lo)
    )
    out: jax.Array = jnp.where(jnp.asarray(lo) < 0, ndtr(z_hi), mid)
    return out / ndtr(z_d)


def model(tree: Tree, tri: Triangle, max_delay: int) -> None:
    """Sample the walk, the delay and the new reports of the leaves."""
    leaves = tri.leaves
    mu = tree.sample("delay_mu", 1.6, 1.0)[leaves]
    sd = jnp.exp(numpyro.sample("delay_sd", dist.Normal(np.log(0.7), 0.5)))
    sigma = jnp.exp(tree.sample("sigma", np.log(0.2), 1.0))[leaves]
    phi = jnp.exp(numpyro.sample("phi", dist.Normal(np.log(10.0), 1.5)))
    n_p = leaves.size
    init = numpyro.sample(
        "init", dist.Normal(2.0, 3.0).expand((n_p,)).to_event()
    )
    # Partly centred steps (`fit_posterior`), as the leaves range from
    # silent to thousands of cases.
    steps = jnp.cumsum(
        numpyro.sample(
            "step",
            dist.Normal(
                jnp.zeros((n_p, tri.n_weeks - 1)), sigma[:, None]
            ).to_event(),
        ),
        axis=1,
    )
    log_lam = init[:, None] + jnp.concatenate(
        [jnp.zeros((n_p, 1)), steps], axis=1
    )
    numpyro.deterministic("log_lam", log_lam)
    p = tri.place
    mass = _interval(mu[p], sd, tri.lo, tri.hi, max_delay)
    mean = jnp.exp(log_lam[p, tri.week]) / 7.0 * mass
    numpyro.sample("obs", dist.NegativeBinomial2(mean, phi), obs=tri.count)


@dataclass(frozen=True)
class Posterior:
    """Posterior draws of the joint model, flat over chains.

    Attributes
    ----------
    mu, sigma : numpy.ndarray
        Delay location and walk spread of every place, `[N, L]`.
    sd, phi : numpy.ndarray
        Delay scale and dispersion, shared by the places, `[N]`.
    log_lam : numpy.ndarray
        Log expected final count of each leaf and fitted week,
        `[N, P, W]`.
    info : dict
        Convergence and timing summaries.
    """

    mu: np.ndarray
    sd: np.ndarray
    sigma: np.ndarray
    phi: np.ndarray
    log_lam: np.ndarray
    info: dict[str, float]


def fit_posterior(
    tri: Triangle,
    tree: Tree,
    sampler: Sampler,
    key: jax.Array,
    max_delay: int,
) -> Posterior:
    """Fit the joint model with NUTS and check that it converged.

    Parameters
    ----------
    tri : Triangle
        The data.
    tree : Tree
        The place hierarchy.
    sampler : Sampler
        NUTS settings.
    key : jax.Array
        Random key.
    max_delay : int
        $D$.

    Returns
    -------
    Posterior
        The draws.

    Raises
    ------
    NotConverged
        If the worst split R-hat is above `sampler.max_rhat` or more than
        `sampler.max_divergent` of the draws diverge.
    """
    mcmc = MCMC(
        NUTS(
            reparam(model, config={"step": LocScaleReparam(CENTRING)}),
            target_accept_prob=sampler.target_accept,
            init_strategy=init_to_median(),
        ),
        num_warmup=sampler.num_warmup,
        num_samples=sampler.num_samples,
        num_chains=sampler.num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    tic = time.perf_counter()
    mcmc.run(
        key,
        tree=tree,
        tri=tri,
        max_delay=max_delay,
        extra_fields=("diverging", "num_steps", "adapt_state.step_size"),
    )
    seconds = time.perf_counter() - tic
    stats = summary(mcmc.get_samples(group_by_chain=True))
    extra = mcmc.get_extra_fields()
    n_div = int(extra["diverging"].sum())
    info: dict[str, float] = {
        "worst_rhat": max(float(np.nanmax(s["r_hat"])) for s in stats.values()),
        "min_ess": min(float(np.nanmin(s["n_eff"])) for s in stats.values()),
        "divergences": n_div,
        "median_steps": float(np.median(extra["num_steps"])),
        "step_size": float(np.median(extra["adapt_state.step_size"])),
        "seconds": seconds,
    }
    draws = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    for name, value in draws.items():
        if name.endswith("_tau"):
            for i, t in enumerate(np.median(value, axis=0)):
                info[f"{name}_{i + 1}"] = float(t)
    n_draws = sampler.num_samples * sampler.num_chains
    if info["worst_rhat"] > sampler.max_rhat or (
        n_div > sampler.max_divergent * n_draws
    ):
        raise NotConverged(
            f"The pooled fit did not converge: worst R-hat "
            f"{info['worst_rhat']:.3f} (most {sampler.max_rhat}) and "
            f"{n_div} of {n_draws} draws divergent."
        )
    return Posterior(
        mu=tree.rebuild(draws, "delay_mu"),
        sd=np.exp(draws["delay_sd"]),
        sigma=np.exp(tree.rebuild(draws, "sigma")),
        phi=np.exp(draws["phi"]),
        log_lam=draws["log_lam"],
        info=info,
    )


def _neg_binomial(
    rng: np.random.Generator, mean: np.ndarray, size: np.ndarray
) -> np.ndarray:
    """Draw NB counts with `mean` and `size` as a gamma-Poisson mixture."""
    rate = rng.gamma(size, 1.0, mean.shape) * mean / size
    return np.asarray(rng.poisson(rate), dtype=float)


def _weeks(snap: SnapshotArrays, task: ForecastTask) -> list[str]:
    """Return every hub week from the first data week to the last target."""
    weeks = list(snap.week_ends)
    last = as_date(max(task.target_end_dates))
    while as_date(weeks[-1]) < last:
        weeks.append((as_date(weeks[-1]) + dt.timedelta(days=7)).isoformat())
    return weeks


@dataclass
class PPRandomWalkNowcast(RandomWalkNowcast):
    """A generative random walk nowcast with partially pooled parameters.

    Attributes
    ----------
    name : str
        Model name.
    sampler : Sampler
        NUTS settings.
    fit_weeks : int
        Hub weeks fitted, ending at the cut-off week.
    """

    name: str = "pprandomwalknowcast"
    sampler: Sampler = field(default_factory=Sampler)
    fit_weeks: int = FIT_WEEKS

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return posterior predictive trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots (`MissingSnapshots`), no pair of
            releases within the delay window (`NoDelayPair`), no place
            with a recent case (`NoRecentCases`), or the fit did not
            converge (`NotConverged`).
        """
        _clear_caches(task)
        daily = task.daily_history
        if daily is None or daily.empty:
            raise MissingSnapshots(
                f"{self.name} needs daily release snapshots, and the task "
                f"for {task.vintage} has none."
            )
        snap = snapshot_arrays(daily, task.vintage, task.locations)
        o = len(snap.releases) - 1
        d = self.settings.max_delay
        if _pairs_at(snap, o, d).size == 0:
            raise NoDelayPair(
                f"No pair of releases up to {snap.releases[o]} has a step "
                f"of at most {d} days."
            )
        tree = Tree.of(snap.parent)
        leaves = np.flatnonzero(tree.leaf)
        silent = silent_places(
            weekly_sums(snap, o), snap.cut_week(o), self.settings.walk_weeks
        )
        live = leaves[~silent[leaves]]
        if live.size == 0:
            raise NoRecentCases(
                f"No place has a case in the {self.settings.walk_weeks} "
                f"weeks to {snap.releases[o]}."
            )
        tri = triangle(snap, o, self.settings, self.fit_weeks, live)
        key = jax.random.key(int(task.rng().integers(0, 2**31 - 1)))
        post = fit_posterior(tri, tree, self.sampler, key, d)
        ages = np.arange(d + 1)
        share = np.median(
            np.asarray(
                reported_share(
                    post.mu[..., None], post.sd[:, None, None], ages, d
                )
            ),
            axis=0,
        )  # [L, D + 1]
        values, weeks = self.predict(post, tri, snap, task, share, silent)
        self.delay_summary = self._delay_summary(share, snap, task.vintage)
        self._describe(post, snap, task, silent)
        rows = [snap.locations.index(c) for c in task.locations]
        return frame(task, weeks, values, rows)

    def predict(
        self,
        post: Posterior,
        tri: Triangle,
        snap: SnapshotArrays,
        task: ForecastTask,
        share: np.ndarray,
        silent: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Draw final counts for every place and week from the posterior.

        Parameters
        ----------
        post : Posterior
            The draws.
        tri : Triangle
            The data the draws were fitted to.
        snap : SnapshotArrays
            The arrays.
        task : ForecastTask
            The task, for the number of samples, the weeks and the seed.
        share : numpy.ndarray
            Posterior median share reported by age of each place,
            `[L, D + 1]`, for the introductions.
        silent : numpy.ndarray
            Places with no recent case, `[L]`. A silent leaf is not
            fitted and draws introductions.

        Returns
        -------
        tuple
            Samples `[S, L, H]` for every hub week from the first data
            week to the last target, and those weeks. A parent is the
            sum of its leaves.
        """
        rng = task.rng("pprandomwalknowcast")
        o = len(snap.releases) - 1
        d = self.settings.max_delay
        live = tri.leaves
        n = task.n_samples
        weeks = _weeks(snap, task)
        n_data, n_all = snap.n_weeks, len(weeks)
        reported = weekly_sums(snap, o)
        values = np.zeros((n, len(snap.locations), n_all))
        # Each sample takes a posterior draw, every draw used once before
        # any is used again.
        pick = np.resize(rng.permutation(post.mu.shape[0]), n)
        mu = post.mu[pick][:, live, None]
        sd = post.sd[pick][:, None, None]
        phi = post.phi[pick][:, None, None]
        sigma = post.sigma[pick][:, live, None]
        log_lam = post.log_lam[pick]  # [S, P, W]
        # Days of the fitted weeks: reported count plus the rest to come.
        day = np.flatnonzero(snap.week >= tri.first)
        age = snap.cut[o] - day
        seen = age >= 0
        rest = np.where(
            seen, 1.0 - np.asarray(reported_share(mu, sd, age, d)), 1.0
        )
        wk = snap.week[day] - tri.first
        # The clip only undoes float32 rounding of a share at 1.
        mean = np.exp(log_lam[:, :, wk]) / 7.0 * np.clip(rest, 0.0, 1.0)
        draw = _neg_binomial(rng, mean, np.broadcast_to(phi, mean.shape))
        known = np.where(seen, snap.counts[live][:, o, day], 0.0)
        onehot = np.zeros((day.size, tri.n_weeks))
        onehot[np.arange(day.size), wk] = 1.0
        # Weeks after the data continue the walk.
        h = n_all - n_data
        steps = np.cumsum(sigma * rng.standard_normal((n, live.size, h)), 2)
        ahead = np.exp(log_lam[:, :, -1:] + steps)
        values[:, live, : tri.first] = reported[live, : tri.first][None]
        values[:, live, tri.first : n_data] = (draw + known[None]) @ onehot
        values[:, live, n_data:] = _neg_binomial(
            rng, ahead, np.broadcast_to(7.0 * phi, ahead.shape)
        )
        self._introductions(values, snap, share, silent, post, rng)
        # A parent is the sum of the leaves below it.
        leaves = np.flatnonzero(Tree.of(snap.parent).leaf)
        below = Tree.of(snap.parent).path[leaves].T.copy()  # [L, leaves]
        below[snap.parent < 0] = 1.0
        values = np.einsum("lp,sph->slh", below, values[:, leaves])
        return values, weeks

    def _introductions(
        self,
        values: np.ndarray,
        snap: SnapshotArrays,
        share: np.ndarray,
        silent: np.ndarray,
        post: Posterior,
        rng: np.random.Generator,
    ) -> None:
        """Fill the silent leaves of `values` with their introductions.

        As `randomwalknowcast` does for a silent place
        (`introductions.silent_trajectories`), with the week's share
        reported and the start week from this model's delay, and the walk
        spread its posterior median.
        """
        quiet = silent & Tree.of(snap.parent).leaf
        if not quiet.any():
            return
        o = len(snap.releases) - 1
        settings = self.settings
        reported = weekly_sums(snap, o)
        age = snap.cut[o] - np.arange(snap.week.size)
        pi = np.where(
            age >= 0, share[:, np.clip(age, 0, settings.max_delay)], 0.0
        )
        complete = week_sum(snap, pi) / 7.0
        good = complete >= settings.completeness
        start = np.where(
            good.any(axis=1),
            snap.n_weeks - 1 - np.argmax(good[:, ::-1], axis=1),
            0,
        )
        q = introduction_rates(
            reported, start, snap.parent, settings.walk_weeks
        )
        n, _, n_all = values.shape
        y = silent_trajectories(
            jax.random.key(int(rng.integers(0, 2**31 - 1))),
            quiet,
            q,
            1.0 - np.clip(complete, 0.0, 1.0),
            start,
            np.median(post.sigma, axis=0),
            n_all,
            n,
        )
        for loc in np.flatnonzero(quiet):
            s = int(start[loc])
            values[:, loc, : s + 1] = reported[loc, : s + 1]
            values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
            values[:, loc, : snap.n_weeks] = np.maximum(
                values[:, loc, : snap.n_weeks], reported[loc]
            )

    def _delay_summary(
        self, share: np.ndarray, snap: SnapshotArrays, vintage: str
    ) -> pd.DataFrame:
        """Return the posterior median share reported by age of each place.

        In the form of `chain.delay_summary`, so the backtest maps this
        model's final counts through its own delay.
        """
        ages = np.arange(share.shape[1])
        places = snap.locations
        return pd.DataFrame(
            {
                "model": self.name,
                "vintage": vintage,
                "location": np.repeat(places, ages.size),
                "kind": "cumulative",
                "x": np.tile(ages, len(places)),
                "quantity": "reported_share",
                "q50": share.ravel(),
            }
        )

    def _describe(
        self,
        post: Posterior,
        snap: SnapshotArrays,
        task: ForecastTask,
        silent: np.ndarray,
    ) -> None:
        """Record each place's posterior medians and the fit's summary."""
        rows: list[dict[str, object]] = []
        for loc in task.locations:
            i = snap.locations.index(loc)
            rows.append(
                {
                    "location": loc,
                    "silent": bool(silent[i]),
                    "sigma": float(np.median(post.sigma[:, i])),
                    "delay_median": float(np.median(np.exp(post.mu[:, i]))),
                }
            )
        shared = {
            "delay_sd": float(np.median(post.sd)),
            "phi": float(np.median(post.phi)),
        }
        self.fit_diagnostics = [*rows, {"fit": "joint", **shared, **post.info}]


register_model(
    "pprandomwalknowcast",
    PPRandomWalkNowcast,
    "A generative random walk nowcast (epinowcast style): weekly expected "
    "counts as a log random walk and a log-normal reporting delay, fitted "
    "jointly to the release snapshots with negative binomial new reports, "
    "the delay, walk spread and dispersion partially pooled over zones, "
    "provinces and country (numpyro).",
    fitted=True,
)
