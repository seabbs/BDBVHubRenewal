# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""External Rt estimates as a tempered pseudo-observation of a latent Rt."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class RtPseudoObservation(RandomVariable):
    r"""External Rt estimates as a tempered pseudo-observation.

    For groups $G$ of provinces (the nation, or BVDOutbreakSize patches)
    and the day $s_b$ of the external estimate, the model's group Rt is
    onset weighted over the week of $s_b$,

    $$R_G(s_b) = \frac{\sum_{p \in G} I_p R_p(s_b)}{\sum_{p \in G} I_p},$$

    and the summaries enter as

    $$\mu_N \sim N(\log R_N(s_b), \sigma_N / \sqrt{\alpha}), \qquad
    d_k \sim N(\log R_k(s_b) - \log R_N(s_b), \tau_k / \sqrt{\alpha}).$$

    This is a power prior on the external summary with tempering
    $\alpha$. With $\alpha = 0$ nothing is added, so the model is
    unchanged. PyRenew routes only infections to observation processes,
    so this is a `RandomVariable` that the latent process calls, as it
    holds both Rt and infections. Its sites are `{name}_log_rt_model`,
    `{name}_z` and `{name}_rt_obs`.
    """

    def __init__(
        self,
        name: str,
        alpha: float,
        day: int,
        groups: ArrayLike,
        values: ArrayLike,
        scales: ArrayLike,
        kinds: ArrayLike,
    ) -> None:
        """Initialize the pseudo-observation.

        Parameters
        ----------
        name
            Prefix for the sites.
        alpha
            Tempering, in [0, 1].
        day
            $s_b$ as a day index from the first data day.
        groups
            Boolean membership, `(G, P)`.
        values
            Log-scale summaries, `(G,)`.
        scales
            Their SDs, positive, `(G,)`.
        kinds
            0 for the national level, 1 for a deviation from it, `(G,)`.

        Raises
        ------
        ValueError
            If `alpha` is outside [0, 1], `day` is negative, the shapes do
            not match, a scale is not positive or a kind is not 0 or 1.
        """
        groups = jnp.asarray(groups, dtype=float)
        values = jnp.asarray(values, dtype=float)
        scales = jnp.asarray(scales, dtype=float)
        kinds = jnp.asarray(kinds, dtype=int)
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if int(day) < 0:
            raise ValueError(f"day must be non-negative, got {day}.")
        if groups.ndim != 2:
            raise ValueError(
                f"groups must have shape (G, P), got {groups.shape}."
            )
        n_groups = groups.shape[0]
        for label, x in (
            ("values", values),
            ("scales", scales),
            ("kinds", kinds),
        ):
            if x.shape != (n_groups,):
                raise ValueError(
                    f"{label} must have shape ({n_groups},), got {x.shape}."
                )
        if bool(jnp.any(scales <= 0)):
            raise ValueError("scales must be positive.")
        if not bool(jnp.all((kinds == 0) | (kinds == 1))):
            raise ValueError("kinds must be 0 (national) or 1 (deviation).")
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.day = int(day)
        self.groups = groups
        self.values = values
        self.scales = scales
        self.kinds = kinds

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RtPseudoObservation(name={self.name!r}, alpha={self.alpha}, "
            f"day={self.day}, n_groups={self.groups.shape[0]})"
        )

    def log_group_rt(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
    ) -> tuple[Array, Array]:
        """Return log $R_G(s_b)$ for each group and for the nation.

        Infections are per resident, so the weights are scaled by the
        population `fractions` to be proportional to counts.

        Parameters
        ----------
        rt
            Daily Rt by province.
        infections
            Daily infections per resident by province.
        n_init
            Seeding days at the start of the axis.
        fractions
            Population share of each province.

        Returns
        -------
        tuple[Array, Array]
            `(G,)` group values and the national scalar.
        """
        week = n_init + 7 * (self.day // 7)
        w = infections[week : week + 7].sum(axis=0)
        if fractions is not None:
            w = w * fractions
        r = rt[n_init + self.day]

        def log_rt(member: Array) -> Array:
            return jnp.log((member * w * r).sum(-1) / (member * w).sum(-1))

        return log_rt(self.groups), log_rt(jnp.ones_like(w))

    def sample(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
        **kwargs: object,
    ) -> None:
        """Add the pseudo-likelihood and record its diagnostics.

        Parameters
        ----------
        rt
            Daily Rt by province.
        infections
            Daily infections per resident by province.
        n_init
            Seeding days at the start of the axis.
        fractions
            Population share of each province.
        **kwargs
            Ignored, for the PyRenew interface.
        """
        group, national = self.log_group_rt(rt, infections, n_init, fractions)
        model = jnp.where(self.kinds == 0, group, group - national)
        numpyro.deterministic(f"{self.name}_log_rt_model", model)
        numpyro.deterministic(
            f"{self.name}_z", (self.values - model) / self.scales
        )
        if self.alpha > 0:
            numpyro.sample(
                f"{self.name}_rt_obs",
                dist.Normal(model, self.scales / jnp.sqrt(self.alpha)),
                obs=self.values,
            )
