"""Enable float64 for the port.

Float32 log-likelihood sums give negative binomial sizes far from R.
"""

import jax

jax.config.update("jax_enable_x64", True)
