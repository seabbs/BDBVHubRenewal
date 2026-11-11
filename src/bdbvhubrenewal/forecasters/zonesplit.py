# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Zone forecasts as a second stage on a model's province samples.

As BVDOutbreakSize's zone model, the zones of a province are a second
stage conditioned on the province: each province sample is taken as
given and split across its zones by a fitted share model, so the zones
sum to the province by construction and province and national forecasts
are unchanged.

For province $p$ with zones $z$, the share of zone $z$ in week $w$ is
$\theta_z(w) = \operatorname{softmax}_z \eta_z(w)$ with

$$
\eta_z(w) =
\begin{cases}
a_z + e_z(w) & z \text{ has at least 30 cases, in a province with two} \\
a_z & z \text{ has a case} \\
u & z \text{ has none},
\end{cases}
$$

$a$ sum-to-zero over the zones with a case, $a \sim \mathcal{N}(0, 1.5^2)$
on its basis, $u \sim \mathcal{N}(-4, 1.5^2)$ the shared level of the
zones with no case relative to the mean of those with one, and $e$
sum-to-zero AR(1) deviations over the zones with at least 30 cases,
$e(w) = \kappa e(w - 1) + \sigma z(w)$, $\sigma \sim \mathcal{N}^+(0,
0.3^2)$, $\kappa = 2^{-1/t_{1/2}}$, $t_{1/2} \sim \text{LogNormal}(\log 6,
0.6^2)$ weeks (BVDOutbreakSize's deviation half-life prior). Zones with
fewer cases keep their level, as BVDOutbreakSize gives week-to-week
changes only to zones with at least 30 cases.
The published zone counts of the last `window` weeks are
Dirichlet-multinomial given the province's published count, with
concentration $\phi\,\theta(w)$, $\phi \sim \text{LogNormal}(\log 50,
1)$. This is BVDOutbreakSize's zone likelihood: it informs the split,
not the level.

To forecast, each province sample takes one posterior draw. The
deviations are carried past the data on their AR(1), each week's shares
are drawn from $\text{Dirichlet}(\phi\,\theta(w))$, and the province
sample, less the zone counts already published, is split by them (as
`ZoneAllocator`, whose published-count handling it shares).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
)
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    _final,
    _zone_frame,
)

__all__ = ["ShareFit", "ZoneShares", "share_model"]

_STREAM = 11
"""Extra seed word, so share draws do not reuse other streams."""


def _sum_to_zero(n: int) -> np.ndarray:
    from bdbvhubrenewal.model.pooled import sum_to_zero_basis

    return sum_to_zero_basis(n)


def share_model(
    counts: np.ndarray,
    seen: np.ndarray,
    own: np.ndarray,
) -> None:
    """NumPyro model of one province's weekly zone shares.

    Parameters
    ----------
    counts : numpy.ndarray
        Published zone counts, `(weeks, zones)`.
    seen : numpy.ndarray
        Boolean `(zones,)`: zones with a case.
    own : numpy.ndarray
        Boolean `(zones,)`: zones with week-to-week deviations.
    """
    n_weeks, n_zones = counts.shape
    k_seen = int(seen.sum())
    eta = jnp.zeros((n_weeks, n_zones))
    if k_seen > 1:
        basis = jnp.asarray(_sum_to_zero(k_seen))
        a = numpyro.sample("level", dist.Normal(0.0, 1.5).expand([k_seen - 1]))
        eta = eta.at[:, np.flatnonzero(seen)].set((basis @ a)[None, :])  # noqa: PD008
    if (~seen).any():
        u = numpyro.sample("unseen", dist.Normal(-4.0, 1.5))
        eta = eta.at[:, np.flatnonzero(~seen)].set(u)  # noqa: PD008
    k_own = int(own.sum())
    if k_own > 1 and n_weeks > 1:
        basis = jnp.asarray(_sum_to_zero(k_own))
        sd = numpyro.sample("dev_sd", dist.HalfNormal(0.3))
        half = numpyro.sample("dev_half_life", dist.LogNormal(np.log(6.0), 0.6))
        kappa = 0.5 ** (1.0 / half)
        z = numpyro.sample(
            "dev_z", dist.Normal(0.0, 1.0).expand([n_weeks - 1, k_own - 1])
        )

        def step(prev: jax.Array, zt: jax.Array) -> tuple[jax.Array, jax.Array]:
            nxt = kappa * prev + sd * zt
            return nxt, nxt

        _, path = jax.lax.scan(step, jnp.zeros(k_own - 1), z)
        coords = jnp.vstack([jnp.zeros((1, k_own - 1)), path])
        eta = eta.at[:, np.flatnonzero(own)].add(coords @ basis.T)  # noqa: PD008
    phi = numpyro.sample("concentration", dist.LogNormal(np.log(50.0), 1.0))
    theta = jax.nn.softmax(eta, axis=-1)
    totals = counts.sum(axis=1)
    keep = totals > 0
    if keep.any():
        numpyro.sample(
            "zone_counts",
            dist.DirichletMultinomial(
                phi * theta[keep] + 1e-6, total_count=totals[keep]
            ),
            obs=counts[keep],
        )


@dataclass
class ShareFit:
    """Posterior draws of one province's share model.

    Attributes
    ----------
    zones : tuple of str
        The province's zones.
    seen, own : numpy.ndarray
        As `share_model`.
    eta : numpy.ndarray
        Logits at the last data week, `(draws, zones)`.
    dev : numpy.ndarray
        Deviation coordinates at the last data week, `(draws, k_own - 1)`.
    sd, kappa, phi : numpy.ndarray
        `(draws,)`.
    seconds : float
        Sampling time.
    max_rhat : float or None
        Worst split R-hat over the sampled sites, None with one chain.
    """

    zones: tuple[str, ...]
    seen: np.ndarray
    own: np.ndarray
    eta: np.ndarray
    dev: np.ndarray
    sd: np.ndarray
    kappa: np.ndarray
    phi: np.ndarray
    seconds: float
    max_rhat: float | None = None


@dataclass
class ZoneShares:
    """Split province samples across zones with a fitted share model.

    Attributes
    ----------
    window : int
        Weeks of zone counts before the vintage that the shares fit.
    min_own_cases : int
        Cases a zone needs for week-to-week deviations of its own.
    num_warmup, num_samples, num_chains : int
        NUTS settings of each province's fit.
    seed : int
        Seed of the fits.
    """

    window: int = 16
    min_own_cases: int = 30
    num_warmup: int = 400
    num_samples: int = 400
    num_chains: int = 2
    seed: int = 1

    def fit_province(
        self,
        counts: pd.DataFrame,
        zones: tuple[str, ...],
        vintage: str,
    ) -> ShareFit:
        """Fit one province's share model to its zone counts.

        Parameters
        ----------
        counts : pandas.DataFrame
            Weekly zone counts at the vintage (`weekly_zone_counts`).
        zones : tuple of str
            The province's zones (all of the hub's).
        vintage : str
            The vintage.

        Returns
        -------
        ShareFit
            The posterior.
        """
        import time

        end = pd.Timestamp(as_date(vintage))
        dates = pd.to_datetime(counts["target_end_date"])
        past = counts.loc[(dates <= end) & counts["location"].isin(zones)]
        by_zone = past.groupby("location")["total"].sum()
        cum = np.asarray(
            by_zone.reindex(list(zones), fill_value=0), dtype=float
        )
        seen = np.asarray(cum > 0)
        own = np.asarray(cum >= self.min_own_cases)
        if own.sum() < 2:
            own = np.zeros_like(own)
        weeks = pd.date_range(
            end - pd.Timedelta(weeks=self.window - 1), end, freq="7D"
        )
        # Week ends are Saturdays; snap the window to them.
        sat = pd.to_datetime(past["target_end_date"]).max()
        if pd.notna(sat):
            weeks = pd.date_range(
                sat - pd.Timedelta(weeks=self.window - 1), sat, freq="7D"
            )
        wide = (
            past.assign(target_end_date=pd.to_datetime(past["target_end_date"]))
            .pivot_table(
                index="target_end_date",
                columns="location",
                values="total",
                aggfunc="sum",
            )
            .reindex(index=weeks, columns=list(zones))
            .fillna(0)
            .to_numpy(float)
        )
        start = time.perf_counter()
        mcmc = MCMC(
            NUTS(share_model, init_strategy=init_to_median),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(jax.random.key(self.seed), wide, seen, own)
        seconds = time.perf_counter() - start
        post = mcmc.get_samples()
        n = next(iter(post.values())).shape[0]
        eta, dev = self._last(post, wide.shape[0], seen, own, n)
        rhat = None
        if self.num_chains > 1:
            from numpyro.diagnostics import split_gelman_rubin

            grouped = mcmc.get_samples(group_by_chain=True)
            values = [
                float(np.nanmax(split_gelman_rubin(np.asarray(v))))
                for v in grouped.values()
            ]
            rhat = max(values) if values else None
        return ShareFit(
            zones=tuple(zones),
            seen=seen,
            own=own,
            eta=eta,
            dev=dev,
            sd=np.asarray(post.get("dev_sd", np.zeros(n))),
            kappa=0.5
            ** (1.0 / np.asarray(post.get("dev_half_life", np.ones(n)))),
            phi=np.asarray(post["concentration"]),
            seconds=seconds,
            max_rhat=rhat,
        )

    @staticmethod
    def _last(
        post: dict[str, Any],
        n_weeks: int,
        seen: np.ndarray,
        own: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Logits and deviation coordinates at the last data week."""
        zones = seen.size
        eta = np.zeros((n, zones))
        k_seen = int(seen.sum())
        if k_seen > 1:
            eta[:, seen] = np.asarray(post["level"]) @ _sum_to_zero(k_seen).T
        if (~seen).any():
            eta[:, ~seen] = np.asarray(post["unseen"])[:, None]
        k_own = int(own.sum())
        dev = np.zeros((n, max(k_own - 1, 0)))
        if k_own > 1 and n_weeks > 1:
            z = np.asarray(post["dev_z"])
            sd = np.asarray(post["dev_sd"])[:, None]
            kappa = 0.5 ** (1.0 / np.asarray(post["dev_half_life"]))[:, None]
            for t in range(z.shape[1]):
                dev = kappa * dev + sd * z[:, t]
            eta[:, own] += dev @ _sum_to_zero(k_own).T
        return eta, dev

    def allocate(
        self,
        samples: pd.DataFrame,
        task: ForecastTask,
        counts: pd.DataFrame,
        zones: tuple[str, ...],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Split each province sample across its zones.

        Parameters
        ----------
        samples : pandas.DataFrame
            A model's province samples for `task`. National and `as_of`
            rows are ignored.
        task : ForecastTask
            The task.
        counts : pandas.DataFrame
            Weekly zone counts at the vintage (`weekly_zone_counts`).
        zones : tuple of str
            Zone codes to cover.

        Returns
        -------
        tuple
            Zone samples (`SAMPLE_COLUMNS`), summing to the province
            samples, and one diagnostics row per province fit.
        """
        rng = np.random.default_rng(
            [task.seed, as_date(task.vintage).toordinal(), _STREAM]
        )
        final = _final(samples)
        final = final.loc[final["location"] != NATIONAL]
        weeks = list(task.target_end_dates)
        ids = np.arange(1, task.n_samples + 1)
        published = counts.set_index(["location", "target_end_date"])["total"]
        last = pd.Timestamp(as_date(task.vintage))
        frames, diagnostics = [], []
        for prov in sorted(set(final["location"])):
            mine = tuple(z for z in zones if province_of(z) == prov)
            if not mine:
                continue
            totals = (
                final.loc[final["location"] == prov]
                .pivot(
                    index="sample", columns="target_end_date", values="value"
                )
                .reindex(index=ids, columns=weeks)
                .fillna(0)
                .clip(lower=0)
                .round()
                .to_numpy(dtype=np.int64)
            )
            pub = np.array(
                [[int(published.get((z, w), 0)) for w in weeks] for z in mine]
            )
            has_cases = counts["location"].isin(mine).any()
            if not has_cases or totals.sum() == 0:
                alpha = ZoneAllocator().alpha(counts, mine, task.vintage)
                values = ZoneAllocator._split(totals, pub, alpha, rng)
            else:
                fit = self.fit_province(counts, mine, task.vintage)
                diagnostics.append(
                    {
                        "province": prov,
                        "zones": len(mine),
                        "own": int(fit.own.sum()),
                        "seconds": fit.seconds,
                        "max_rhat": fit.max_rhat,
                    }
                )
                ahead = [
                    max(0, (pd.Timestamp(w) - last).days // 7) for w in weeks
                ]
                values = self._split(fit, totals, pub, ahead, rng)
            frames.append(_zone_frame(mine, weeks, ids, values))
        if not frames:
            return pd.DataFrame(columns=list(SAMPLE_COLUMNS)), diagnostics
        out = pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]
        return out, diagnostics

    @staticmethod
    def _split(
        fit: ShareFit,
        totals: np.ndarray,
        published: np.ndarray,
        ahead: list[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Split `(samples, weeks)` totals into `(samples, weeks, zones)`."""
        n, n_weeks = totals.shape
        n_zones = fit.seen.size
        draw = rng.choice(fit.eta.shape[0], n)
        eta = fit.eta[draw].copy()
        dev = fit.dev[draw].copy()
        basis = (
            _sum_to_zero(int(fit.own.sum()))
            if dev.shape[1]
            else np.zeros((0, 0))
        )
        base = eta.copy()
        if dev.shape[1]:
            base[:, fit.own] -= dev @ basis.T
        out = np.broadcast_to(published.T, (n, n_weeks, n_zones)).copy()
        extra = totals - published.sum(axis=0)[None, :]
        horizon = max(ahead) if ahead else 0
        devs = [dev]
        for _ in range(horizon):
            step = fit.kappa[draw, None] * devs[-1] + fit.sd[draw, None] * (
                rng.standard_normal(devs[-1].shape)
            )
            devs.append(step)
        for j in range(n_weeks):
            logit = base.copy()
            if dev.shape[1]:
                logit[:, fit.own] += devs[ahead[j]] @ basis.T
            theta = np.exp(logit - logit.max(axis=1, keepdims=True))
            theta /= theta.sum(axis=1, keepdims=True)
            alpha = fit.phi[draw, None] * theta + 1e-6
            p = rng.gamma(alpha)
            p /= p.sum(axis=1, keepdims=True)
            out[:, j] += rng.multinomial(np.maximum(extra[:, j], 0), p)
            for k in np.flatnonzero(extra[:, j] < 0):
                out[k, j] = rng.multivariate_hypergeometric(
                    published[:, j], totals[k, j]
                )
        return out
