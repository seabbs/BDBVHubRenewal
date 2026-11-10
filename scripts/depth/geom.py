"""Laplace geometry of the renewal posterior: does a fixed rotation of the
walk steps cut the condition number a diagonal mass matrix leaves?

Usage: python geom.py MODEL  (x64)
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import rsetup

name = sys.argv[1]
OUT = os.environ.get("DEPTH_OUT", os.path.join(os.path.dirname(__file__), "out"))
os.makedirs(OUT, exist_ok=True)
tag = os.environ.get("BDBV_WALK_ANCHOR", "first")
out = os.path.join(OUT, f"geom_{name.replace(':', '_')}_{tag}.npz")
t0 = time.time()
info, data, model, margs = rsetup.setup(name)
z0 = info.param_info.z
flat0, unravel = ravel_pytree(z0)
pot = jax.jit(lambda x: info.potential_fn(unravel(x)))
vg = jax.jit(jax.value_and_grad(lambda x: info.potential_fn(unravel(x))))
print("setup", time.time() - t0, flush=True)
open(out.replace(".npz", "_sizes.txt"), "w").write(repr({k: tuple(np.shape(v)) for k, v in z0.items()}))
if os.path.exists(out):
    x = np.load(out)["x"]
    if "H" in np.load(out):
        H = np.load(out)["H"]
else:
    res = minimize(lambda x: tuple(np.asarray(a, dtype=float) for a in vg(x)),
                   np.asarray(flat0, dtype=float), jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 40000, "gtol": 1e-6})
    x = res.x
    np.savez(out, x=x)
    print("map", res.message, res.nit, float(res.fun),
          float(np.max(np.abs(res.jac))), time.time() - t0, flush=True)
if "H" not in dir():
    H = np.asarray(jax.jit(jax.hessian(lambda x: info.potential_fn(unravel(x))))(jnp.asarray(x)))
    np.savez(out, x=x, H=H)
print("hessian", time.time() - t0, flush=True)

# Site layout in the flat vector.
layout, i = {}, 0
for k in sorted(z0):  # ravel_pytree orders dict keys sorted
    v = np.asarray(z0[k])
    layout[k] = (i, v.shape)
    i += v.size


def cond_diag(H):
    """Condition number after the diagonal scaling NUTS adapts towards
    (posterior variances = diag of H^-1)."""
    w, V = np.linalg.eigh(H)
    if w.min() <= 0:
        return np.inf, w.min()
    cov = (V / w) @ V.T
    d = np.sqrt(np.diag(cov))
    Hs = H * d[:, None] * d[None, :]
    e = np.linalg.eigvalsh(Hs)
    return e.max() / e.min(), e


def rotation(m):
    L = np.tril(np.ones((m, m)))
    _, Q = np.linalg.eigh(L.T @ L)
    return Q


def rotate(H, sites):
    n = H.shape[0]
    R = np.eye(n)
    for s in sites:
        start, shape = layout[s]
        m = shape[0]
        k = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        Q = rotation(m)
        idx = np.arange(start, start + m * k).reshape(m, k)  # row-major
        for j in range(k):
            ii = idx[:, j]
            R[np.ix_(ii, ii)] = Q
    return R.T @ H @ R


walks = [k for k in layout if k.endswith("_z") and len(layout[k][1]) >= 1
         and layout[k][1][0] > 2]
print("walk sites", {k: layout[k][1] for k in walks})
c0, e0 = cond_diag(H)
print(f"all: cond after diag scaling, current = {c0:.3g}")
c1, e1 = cond_diag(rotate(H, walks))
print(f"all: rotated walks = {c1:.3g}")
for k in walks:
    c, _ = cond_diag(rotate(H, [k]))
    print(f"  rotate only {k}: {c:.3g}")
# Spectrum tails: how many directions are far from the bulk?
for lab, e in (("current", e0), ("rotated", e1)):
    e = np.sort(e)
    print(lab, "smallest", np.round(e[:5], 6), "largest", np.round(e[-5:], 2))
