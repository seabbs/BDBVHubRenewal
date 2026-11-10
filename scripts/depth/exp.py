"""Short single-chain NUTS on renewal at 11-01: diagonal adaptation vs a
Laplace dense metric from the MAP Hessian (geom_renewal.npz).

Usage: METRIC=diag|laplace WARM=150 DRAWS=100 python exp.py
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS

sys.path.insert(0, os.path.dirname(__file__))
import rsetup

metric = os.environ.get("METRIC", "diag")
warm, draws = int(os.environ.get("WARM", 150)), int(os.environ.get("DRAWS", 100))
here = os.environ.get("DEPTH_OUT", os.path.join(os.path.dirname(__file__), "out"))
os.makedirs(here, exist_ok=True)
anchor = os.environ.get("BDBV_WALK_ANCHOR", "first")
info, data, model, args = rsetup.setup("renewal")
z0 = info.param_info.z
_, unravel = ravel_pytree(z0)
kw = dict(target_accept_prob=0.9, max_tree_depth=int(os.environ.get("DEPTH", 10)))
init = z0
if os.environ.get("INIT") == "map":
    g = np.load(os.path.join(here, f"geom_renewal_{anchor}.npz"))
    init = unravel(jnp.asarray(g["x"], jnp.float32))
if metric == "laplace":
    g = np.load(os.path.join(here, f"geom_renewal_{anchor}.npz"))
    H = (g["H"] + g["H"].T) / 2
    w, V = np.linalg.eigh(H)
    w = np.maximum(np.abs(w), 1e-3)
    inv_mass = (V / w) @ V.T
    kw |= dict(dense_mass=True, inverse_mass_matrix=jnp.asarray(inv_mass, jnp.float32),
               adapt_mass_matrix=os.environ.get("ADAPT", "0") == "1")
    init = unravel(jnp.asarray(g["x"], jnp.float32))
kernel = NUTS(model.sample, **kw)
mcmc = MCMC(kernel, num_warmup=warm, num_samples=draws, num_chains=1,
            progress_bar=False)
t0 = time.time()
mcmc.run(jax.random.key(int(os.environ.get("SEED", 1))), init_params=init,
         extra_fields=("num_steps", "adapt_state.step_size", "diverging"), **args)
samples = mcmc.get_samples(group_by_chain=True)
secs = time.time() - t0
ex = mcmc.get_extra_fields()
steps = np.asarray(ex["num_steps"])
ess = {k: float(np.min(effective_sample_size(np.asarray(v))))
       for k, v in samples.items() if k in z0}
lowest = sorted(ess.items(), key=lambda t: t[1])[:5]
out = {
    "metric": metric, "warm": warm, "draws": draws, "seconds": secs,
    "mean_steps": float(steps.mean()), "share_1023": float((steps >= 1023).mean()),
    "step_size": float(np.asarray(ex["adapt_state.step_size"])[-1]),
    "divergent": int(np.asarray(ex["diverging"]).sum()),
    "grads": int(steps.sum()), "min_ess": lowest,
}
out["depth"] = kw["max_tree_depth"]; out["seed"] = int(os.environ.get("SEED", 1))
out["anchor"] = anchor
out["ess_per_s"] = lowest[0][1] / secs
out["depth_hist"] = {int(k): int(v) for k, v in zip(*np.unique(np.floor(np.log2(steps + 1)).astype(int), return_counts=True))}
out["ess_per_1k_grads"] = lowest[0][1] / max(1, out["grads"]) * 1000
tag = f"{anchor}_{metric}_d{out['depth']}_s{out['seed']}"
if os.environ.get("INIT") == "map":
    tag += "_map"
out["init"] = os.environ.get("INIT", "median")
keep = {k: np.asarray(v) for k, v in samples.items()
        if k in z0 or k in ("onsets_expected_known", "onsets_expected_imputed")}
np.savez(os.path.join(here, f"draws_{tag}.npz"), **keep)
with open(os.path.join(here, f"result_{tag}.json"), "w") as fh:
    json.dump(out, fh)
print(json.dumps(out))
