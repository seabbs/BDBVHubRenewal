"""Summarise the sampler geometry a `renewal_probe.py` run dumped.

For each chain, the unconstrained draws are scaled by the adapted
diagonal mass matrix. The eigenvalues of their covariance give the
condition number NUTS faces. Their eigenvectors name the parameters
along the longest and the shortest directions, and the per-parameter
ratio of posterior SD to adapted scale shows any adaptation lag.

Usage::

    uv run python scripts/renewal_geometry.py output/probe/bvd-base
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _flat(g: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    sites = sorted(k[3:] for k in g.files if k.startswith("z::"))
    blocks, names = [], []
    for s in sites:
        x = g[f"z::{s}"]
        c, n = x.shape[:2]
        x = x.reshape(c, n, -1)
        blocks.append(x)
        names += [f"{s}[{i}]" for i in range(x.shape[2])]
    return np.concatenate(blocks, axis=2), names


def main(path: str, top: int = 8) -> None:
    """Print the geometry summary of one probe directory."""
    g = np.load(Path(path) / "geometry.npz")
    z, names = _flat(g)
    imm_keys = [k for k in g.files if k.startswith("imm::")]
    imm = g[imm_keys[0]]
    steps = g["step_size"]
    num_steps = g["num_steps"]
    print(f"{path}: {z.shape[2]} unconstrained dims, {z.shape[1]} draws")
    for c in range(z.shape[0]):
        m = imm[c] if imm.ndim == 2 else imm
        scale = np.sqrt(np.diag(m)) if m.ndim == 2 else np.sqrt(m)
        u = z[c] / scale
        cov = np.cov(u, rowvar=False)
        ev, vec = np.linalg.eigh(cov)
        ev = np.clip(ev, 1e-12, None)
        ratio = z[c].std(axis=0) / scale
        print(
            f"\nchain {c}: step {float(np.ravel(steps)[c]):.4g}, "
            f"median steps {np.median(num_steps[c]):.0f}, "
            f"sqrt condition {np.sqrt(ev[-1] / ev[0]):.0f} "
            f"(sd range {np.sqrt(ev[0]):.3g} to {np.sqrt(ev[-1]):.3g})"
        )
        print(
            "  posterior sd / adapted scale: "
            f"min {ratio.min():.2f} ({names[ratio.argmin()]}), "
            f"max {ratio.max():.2f} ({names[ratio.argmax()]})"
        )
        for label, idx in (
            ("smallest", range(3)),
            ("largest", range(len(ev) - 1, len(ev) - 4, -1)),
        ):
            for i in idx:
                load = vec[:, i] ** 2
                lead = np.argsort(load)[::-1][:top]
                parts = ", ".join(f"{names[j]} {load[j]:.2f}" for j in lead)
                print(f"  {label} sd {np.sqrt(ev[i]):.3g}: {parts}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
