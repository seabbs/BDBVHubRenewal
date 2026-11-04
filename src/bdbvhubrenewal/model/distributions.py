# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A Dirichlet-multinomial scored on the non-zero cells of its value."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jax.scipy.special import gammaln
from jax.typing import ArrayLike


class ObservedCellsDirichletMultinomial(dist.DirichletMultinomial):
    r"""Dirichlet-multinomial whose log density skips cells with no count.

    With concentration $\alpha$, $A = \sum_k \alpha_k$ and $n = \sum_k x_k$,

    $$\log p(x) = \log B_1(A, n) - \sum_k \log B_1(\alpha_k, x_k),
    \qquad \log B_1(a, x) = \log\Gamma(1 + x) + \log\Gamma(a)
    - \log\Gamma(a + x),$$

    as `numpyro.distributions.DirichletMultinomial`. A cell with
    $x_k = 0$ adds $\log\Gamma(\alpha_k) - \log\Gamma(\alpha_k) = 0$
    exactly, with zero gradient, so only the non-zero cells are evaluated,
    as epinowcast scores only the observed cells. Sparse delay tables,
    such as delays by onset week, then cost a fraction of the dense
    density.

    The value must be concrete, not traced, because its non-zero cells
    are found with NumPy. Observed data passed to a NumPyro model are.
    Sampling and every other method are the parent's.
    """

    def log_prob(self, value: ArrayLike) -> Array:
        """Log density of a concrete count table.

        Parameters
        ----------
        value : ArrayLike
            Counts, shape `batch_shape + (K,)`.

        Returns
        -------
        Array
            Log density, shape `batch_shape`.
        """
        x = np.asarray(value)
        alpha = jnp.broadcast_to(self.concentration, x.shape)
        k = x.shape[-1]
        rows, cols = np.nonzero(x.reshape(-1, k))
        a = alpha.reshape(-1, k)[rows, cols]
        xk = x.reshape(-1, k)[rows, cols]
        cells = gammaln(1 + xk) + gammaln(a) - gammaln(a + xk)
        n = x.sum(-1)
        total = alpha.sum(-1)
        by_row = jax.ops.segment_sum(
            cells, rows, num_segments=x.size // k
        ).reshape(n.shape)
        return gammaln(1 + n) + gammaln(total) - gammaln(total + n) - by_row
