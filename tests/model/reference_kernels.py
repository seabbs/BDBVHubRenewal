"""Reference (gather and cumsum) versions of the fast kernels.

The package uses the banded-matrix `onset_expectations` and the
matrix-product `hazard_to_pmf`, which are faster under autodiff. These
are the direct versions they replaced, kept to test against.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bdbv_hub_renewal.model.observation import backsample


def onset_expectations_reference(
    onsets: ArrayLike,
    delay: ArrayLike,
    missing: ArrayLike,
    hub_delay: ArrayLike,
    cutoff: int,
) -> dict[str, ArrayLike]:
    r"""Compute expected known and imputed onsets by the cut-off and eventually.

    Parameters
    ----------
    onsets : ArrayLike
        $\lambda_p(s)$, shape `(n, P)`.
    delay : ArrayLike
        $f_s(d)$, shape `(n, D + 1)`.
    missing : ArrayLike
        $\pi_p(t)$ by report day, shape `(n, P)`. Report days past the
        end use the last row.
    hub_delay : ArrayLike
        The hub's empirical pmf $\hat f$.
    cutoff : int
        Cut-off day $T$ on the same axis. Static.

    Returns
    -------
    dict
        `known` and `imputed` by $T$, `known_final` and `imputed_final`
        once all reports are in (all `(n, P)`), and `triangle`, the
        expected known cases by onset day and delay reported by $T$,
        summed over provinces, `(n, D + 1)`.
    """
    n, n_d = delay.shape
    report_day = jnp.arange(n)[:, None] + jnp.arange(n_d)[None, :]
    pi_at = missing[jnp.minimum(report_day, n - 1)]
    cells = onsets[:, None, :] * delay[:, :, None]
    known_cells = cells * (1 - pi_at)
    by_cutoff = (report_day <= cutoff)[:, :, None]
    reports = jnp.zeros((n + n_d, onsets.shape[1])).at[report_day].add(cells)
    pi_ext = missing[jnp.minimum(jnp.arange(n + n_d), n - 1)]
    missing_reports = pi_ext * reports
    allocate = jax.vmap(backsample, in_axes=(1, None), out_axes=1)
    imputed_final = allocate(missing_reports, hub_delay)[:n]
    imputed = allocate(missing_reports[: cutoff + 1], hub_delay)
    imputed = jnp.zeros_like(onsets).at[: cutoff + 1].set(imputed[:n])
    return {
        "known": jnp.sum(known_cells * by_cutoff, axis=1),
        "imputed": imputed,
        "known_final": known_cells.sum(axis=1),
        "imputed_final": imputed_final,
        "triangle": jnp.sum(known_cells * by_cutoff, axis=2),
    }


def hazard_to_pmf_reference(logit_hazard: ArrayLike) -> ArrayLike:
    r"""Convert discrete-time hazards to a pmf on the last axis.

    $$f(d) = h(d) \prod_{j < d} \{1 - h(j)\}, \qquad h(D) = 1,$$

    so each row sums to one.

    Parameters
    ----------
    logit_hazard : ArrayLike
        Logit hazards for delays 0..D-1, shape `(..., D)`.

    Returns
    -------
    ArrayLike
        Pmf on 0..D, shape `(..., D + 1)`.
    """
    log_h = jax.nn.log_sigmoid(logit_hazard)
    log_1mh = jax.nn.log_sigmoid(-logit_hazard)
    log_surv = jnp.cumsum(log_1mh, axis=-1)
    zero = jnp.zeros((*log_surv.shape[:-1], 1))
    log_surv_before = jnp.concatenate([zero, log_surv], axis=-1)
    log_h = jnp.concatenate([log_h, zero], axis=-1)
    return jnp.exp(log_h + log_surv_before)
