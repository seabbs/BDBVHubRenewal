# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""`randomwalknowcast` with partially pooled estimates.

`PPRandomWalkNowcast` (`pprandomwalknowcast`) is `randomwalknowcast`
with its three estimates, the delay, the walk spread and the nowcast
uncertainty, fitted in numpyro with partial pooling over the place
hierarchy (country, province, zone) in place of the own-or-pooled rule
of `pooling`.
The nowcast draws, the walk and the zone reconciliation are those of
`randomwalknowcast`.

Each parameter $\theta$ of a place $n$ is its parent's plus a level
offset,

$$
\theta_n = \theta_{\mathrm{pa}(n)} + \tau_{\ell(n)} z_n,
\qquad z_n \sim \mathcal{N}(0, 1),
\qquad \tau_\ell \sim \mathcal{N}^+(0, 0.5^2),
$$

with $\ell(n)$ the level of $n$ (province or zone) and one $\tau$ per
level for each parameter.
A parent's counts are the sum of its children's, so for the walk spread
and the nowcast uncertainty only the task's places, the leaves, enter
the likelihood and their ancestors are latent centres.

- **Delay.** $g_n(j) = \exp \eta_n(j)$ with $\eta$ pooled by age $j$.
  The increment of each gap and age between the releases of a pair,
  as `chain.delay_sums` gives it, is Poisson with mean
  $\mathrm{den} \cdot (\exp \sum_{j \in \text{span}} g_n(j) - 1)$, the
  model whose maximum likelihood estimate is the chain ladder.
  As in `randomwalknowcast` (`chain.own_ages`), a place's own increments
  at an age enter only where its denominator there, summed over the
  pairs, is at least `Settings.min_level` cases, and the country's
  always do.
  Below that a day often has no cases at one release and some at the
  next, which a factor cannot give, so the place's few days with cases
  would speak for all of them and give too fast a delay.
  Those ages take their parent's value, through the pooling.
  It is fitted at the origin and at each earlier release, whose nowcasts
  give the errors of the observation model.
- **Walk spread.** $\log \sigma_n$ pooled, with the weekly changes of
  $\log(1 + y)$ up to the start week normal with that spread.
  A change between two weeks with no cases is at the walk's floor and
  says nothing about its spread, so it is left out.
- **Nowcast uncertainty.** $\log \phi_{n,k}$ and $\log b_{n,k}$ pooled by
  weeks back $k$, with the new reports negative binomial on the count
  nowcast to come and the removals Poisson on the count reported, as
  in `obs`.

Posterior medians are plugged into `randomwalknowcast`'s draws, so the
posterior spread of the estimates does not reach the forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_median

from bdbvhubrenewal.data.snapshots import SnapshotArrays, snapshot_arrays
from bdbvhubrenewal.forecasters.base import ForecastTask, register_model
from bdbvhubrenewal.forecasters.chain import (
    DelayFit,
    DelayFn,
    Errors,
    MissingSnapshots,
    Origin,
    Settings,
    build_origin,
    delay_sums,
    levels,
    own_ages,
    retro_errors,
    walk_changes,
    weeks_back,
)
from bdbvhubrenewal.forecasters.obs import MAX_SIZE
from bdbvhubrenewal.forecasters.randomwalk import RandomWalkNowcast

__all__ = ["PPRandomWalkNowcast", "Sampler", "Tree"]

MIN_SIZE = 0.1
"""Lower bound of `fit_nb`'s size, kept for the plugged-in sizes."""


@dataclass(frozen=True)
class Sampler:
    """NUTS settings for the pooled fits.

    Attributes
    ----------
    num_warmup, num_samples, num_chains : int
        Warm-up and kept draws per chain, and chains.
    target_accept : float
        Target acceptance.
    """

    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    target_accept: float = 0.9


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

    def sample(
        self, name: str, loc: float, scale: float, shape: tuple[int, ...] = ()
    ) -> jax.Array:
        r"""Sample a pooled parameter for every place, non-centred.

        Parameters
        ----------
        name : str
            Site prefix.
        loc, scale : float
            Normal prior of the country's value.
        shape : tuple of int
            Shape of the parameter of one place.

        Returns
        -------
        jax.Array
            `[L, *shape]`.
        """
        root = numpyro.sample(
            f"{name}_root", dist.Normal(loc, scale).expand(shape).to_event()
        )
        n_lvl = int(self.level.max())
        tau = numpyro.sample(
            f"{name}_tau", dist.HalfNormal(0.5).expand((n_lvl,)).to_event()
        )
        z = numpyro.sample(
            f"{name}_z",
            dist.Normal(0.0, 1.0).expand((self.level.size, *shape)).to_event(),
        )
        scale_n = tau[jnp.maximum(self.level - 1, 0)] * (self.level > 0)
        step = scale_n.reshape(-1, *([1] * len(shape))) * z
        out: jax.Array = root + jnp.tensordot(
            jnp.asarray(self.path), step, axes=1
        )
        return out


def _gap_sums(
    num: np.ndarray, den: np.ndarray, gaps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the pairs of each gap by age, as `log_factors` does.

    Returns
    -------
    tuple of numpy.ndarray
        Increments and denominators `[L, S * D]` and the 0/1 matrix of the
        ages each row spans, `[S * D, D]`.
    """
    uniq = np.unique(gaps)
    member = (gaps[:, None] == uniq[None, :]).astype(float)
    n_sum = np.einsum("lpa,ps->lsa", num, member)
    d_sum = np.einsum("lpa,ps->lsa", den, member)
    d = num.shape[-1]
    j = np.arange(d)[None, None, :]
    lo = np.arange(d)[None, :, None]
    span = ((j >= lo) & (j < lo + uniq[:, None, None])).reshape(-1, d)
    n_loc = num.shape[0]
    return (
        (n_sum - d_sum).reshape(n_loc, -1),
        d_sum.reshape(n_loc, -1),
        span.astype(float),
    )


def _nb(mean: jax.Array, size: jax.Array) -> dist.Distribution:
    return dist.NegativeBinomial2(jnp.maximum(mean, 1e-8), size)


def delay_model(
    tree: Tree,
    inc: np.ndarray,
    den: np.ndarray,
    span: np.ndarray,
    keep: np.ndarray,
) -> None:
    """Pooled daily delay fitted to the release-pair increments in `keep`."""
    eta = tree.sample("delay", -3.0, 2.0, (span.shape[1],))
    g = jnp.exp(eta)
    mean = den * jnp.expm1(g @ jnp.asarray(span).T)
    with numpyro.handlers.mask(mask=keep):
        numpyro.sample(
            "delay_obs", dist.Poisson(mean + 1e-8), obs=np.round(inc)
        )


def spread_model(
    tree: Tree,
    changes: np.ndarray,
    seen: np.ndarray,
    added: np.ndarray,
    remaining: np.ndarray,
    removed: np.ndarray,
    reported: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Pooled walk spread, negative binomial size and removal rate."""
    k_n = added.shape[-1]
    log_sigma = tree.sample("sigma", np.log(0.2), 1.0)
    log_size = tree.sample("size", 0.0, 1.5, (k_n,))
    log_b = tree.sample("removal", np.log(0.005), 1.5, (k_n,))
    leaf = tree.leaf
    with numpyro.handlers.mask(mask=leaf[:, None] & seen):
        numpyro.sample(
            "walk_obs",
            dist.Normal(0.0, jnp.exp(log_sigma)[:, None]),
            obs=changes,
        )
    fit = leaf[None, :, None] & valid & (remaining > 0)
    with numpyro.handlers.mask(mask=fit):
        numpyro.sample(
            "added_obs",
            _nb(jnp.asarray(remaining), jnp.exp(log_size)[None]),
            obs=np.round(added),
        )
    gone = leaf[None, :, None] & valid & (reported > 0)
    with numpyro.handlers.mask(mask=gone):
        numpyro.sample(
            "removed_obs",
            dist.Poisson(jnp.exp(log_b)[None] * reported + 1e-8),
            obs=np.round(removed),
        )


def _run(
    model: Any, key: jax.Array, sampler: Sampler, **data: Any
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run NUTS and return posterior medians and a diagnostics summary."""
    mcmc = MCMC(
        # From the prior medians: uniform starts put the delay factors
        # where their growth over a gap overflows.
        NUTS(
            model,
            target_accept_prob=sampler.target_accept,
            init_strategy=init_to_median(),
        ),
        num_warmup=sampler.num_warmup,
        num_samples=sampler.num_samples,
        num_chains=sampler.num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        key,
        **data,
        extra_fields=("diverging", "num_steps", "adapt_state.step_size"),
    )
    draws = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields()
    stats = summary(draws)
    rhat = max(float(np.nanmax(s["r_hat"])) for s in stats.values())
    ess = min(float(np.nanmin(s["n_eff"])) for s in stats.values())
    info = {
        "worst_rhat": rhat,
        "min_ess": ess,
        "divergences": int(extra["diverging"].sum()),
        "median_steps": float(np.median(extra["num_steps"])),
        "step_size": float(np.median(extra["adapt_state.step_size"])),
    }
    flat = mcmc.get_samples()
    for name, value in flat.items():
        if name.endswith("_tau"):
            for i, t in enumerate(np.median(np.asarray(value), axis=0)):
                info[f"{name}_{i + 1}"] = float(t)
    return {k: np.asarray(v) for k, v in flat.items()}, info


def _median_of(tree: Tree, draws: dict[str, np.ndarray], name: str) -> Any:
    """Rebuild a pooled parameter from its draws and take the median."""
    root, tau, z = (draws[f"{name}_{p}"] for p in ("root", "tau", "z"))
    lvl = np.maximum(tree.level - 1, 0)
    scale = tau[:, lvl] * (tree.level > 0)
    extra = z.ndim - 2
    step = scale.reshape(*scale.shape, *([1] * extra)) * z
    full = root[:, None] + np.einsum("nm,sm...->sn...", tree.path, step)
    return np.median(full, axis=0)


@dataclass
class PPRandomWalkNowcast(RandomWalkNowcast):
    """`randomwalknowcast` with partially pooled delay, spread and errors.

    Attributes
    ----------
    name : str
        Model name.
    sampler : Sampler
        NUTS settings.
    fits : list of dict
        Diagnostics of each NUTS fit of the last forecast.
    """

    name: str = "pprandomwalknowcast"
    sampler: Sampler = field(default_factory=Sampler)
    fits: list[dict[str, object]] = field(default_factory=list, repr=False)

    def origin_at(self, task: ForecastTask, errors: bool) -> Origin:
        """Fit the pooled estimates at the task's vintage.

        Parameters
        ----------
        task : ForecastTask
            The task. Its `daily_history` holds the release snapshots.
        errors : bool
            Also fit the observation model.

        Returns
        -------
        Origin
            The fit, with the pooled delay, spread and observation model.

        Raises
        ------
        MissingSnapshots
            If the task has no release snapshots.
        """
        daily = task.daily_history
        if daily is None or daily.empty:
            raise MissingSnapshots(
                f"{self.name} needs daily release snapshots, and the task "
                f"for {task.vintage} has none."
            )
        snap = snapshot_arrays(daily, task.vintage, task.locations)
        tree = Tree.of(snap.parent)
        key = jax.random.key(int(task.rng().integers(0, 2**31 - 1)))
        self.fits = []

        def delay(s: SnapshotArrays, r: int, st: Settings) -> DelayFit | None:
            sums = delay_sums(s, r, st)
            if sums is None:
                return None
            pairs, num, den, gaps = sums
            inc, d_sum, span = _gap_sums(num, den, gaps)
            own = own_ages(den, s.parent, st)
            rows = np.tile(own, (1, d_sum.shape[1] // own.shape[1]))
            draws, info = _run(
                delay_model,
                jax.random.fold_in(key, r),
                self.sampler,
                tree=tree,
                inc=inc,
                den=d_sum,
                span=span,
                keep=rows & (d_sum > 0),
            )
            self.fits.append({"fit": f"delay at {s.releases[r]}", **info})
            g = np.exp(_median_of(tree, draws, "delay"))
            n = len(s.locations)
            return DelayFit(pairs, g, levels(s, r, st), np.arange(n), own)

        origin = build_origin(snap, self.settings, False, delay)
        self._pool_spreads(
            origin, tree, delay if errors else None, jax.random.fold_in(key, 0)
        )
        return origin

    def _pool_spreads(
        self, origin: Origin, tree: Tree, delay: DelayFn | None, key: jax.Array
    ) -> None:
        """Replace the walk spread and, with `delay`, fit the errors.

        `delay` fits the delay at each earlier release for the errors of
        the observation model. Without it there are no errors and only
        the walk spread is fitted.
        """
        snap, settings = origin.snap, origin.settings
        diffs = walk_changes(origin.reported, origin.start, settings)
        width = max(max(d.size for d in diffs), 1)
        changes = np.zeros((len(diffs), width))
        seen = np.zeros((len(diffs), width), bool)
        for n, d in enumerate(diffs):
            lo = max(int(origin.start[n]) - settings.walk_weeks + 1, 0)
            y = origin.reported[n, lo : int(origin.start[n]) + 1]
            changes[n, : d.size] = d
            seen[n, : d.size] = (y[1:] > 0) | (y[:-1] > 0)
        if delay is None:
            n_k = weeks_back(snap, origin.o, settings.max_delay)
            empty = np.zeros((0, len(snap.locations), n_k))
            err = Errors(empty, empty, empty, empty, empty > 0)
        else:
            err = retro_errors(snap, origin.o, settings, delay)
        draws, info = _run(
            spread_model,
            key,
            self.sampler,
            tree=tree,
            changes=changes,
            seen=seen,
            added=err.added,
            remaining=err.remaining,
            removed=err.removed,
            reported=err.reported,
            valid=err.valid,
        )
        self.fits.append({"fit": "spread and errors", **info})
        origin.sigma = np.exp(_median_of(tree, draws, "sigma"))
        origin.errors = err
        origin.size = np.clip(
            np.exp(_median_of(tree, draws, "size")), MIN_SIZE, MAX_SIZE
        )
        origin.removal = np.exp(_median_of(tree, draws, "removal"))

    def _diagnostics(
        self, origin: Origin, task: ForecastTask, start: np.ndarray
    ) -> None:
        rows: list[dict[str, object]] = []
        for loc, i in zip(task.locations, self.rows(origin, task), strict=True):
            rows.append(
                {
                    "location": loc,
                    "start_week": origin.snap.week_ends[int(start[i])],
                    "sigma": float(origin.sigma[i]),
                }
            )
        self.fit_diagnostics = [*rows, *self.fits]


register_model(
    "pprandomwalknowcast",
    PPRandomWalkNowcast,
    "Random walk on a daily nowcast, with the delay, walk spread and "
    "nowcast uncertainty partially pooled over zones, provinces and "
    "country (numpyro).",
    fitted=True,
)
