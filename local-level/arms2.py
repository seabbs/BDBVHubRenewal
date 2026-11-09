"""Run the CLI with variant patches: OWNSIG (own walk spread), NBPRIOR (no Poisson-end size)."""
import os, sys
import numpy as np
from bdbvhubrenewal.forecasters import chain
from bdbvhubrenewal.forecasters import randomwalk as rw
_bo = rw.build_origin
def _build(*a, **k):
    o = _bo(*a, **k)
    if os.environ.get("OWNSIG"):
        own, _ = getattr(chain, "walk_spreads", getattr(chain, "_walk_spreads", None))(o.reported, o.start, o.snap, o.fit.level, o.settings)
        o.sigma = np.where(own > 0, own, o.sigma)
    if os.environ.get("NBPRIOR") and o.size is not None:
        o.size = np.where(o.size >= 1000, 2.2, o.size)
    return o
rw.build_origin = _build
if os.environ.get("NOISE"):
    def _filter(start, origin, w0, cw):
        n_loc, n_data = origin.reported.shape
        level = np.repeat(start[:, :, None], n_data, axis=2)
        var = np.zeros((n_loc, n_data))
        for u in range(n_data):
            after = (u > w0) & (u <= cw)
            share, y = origin.complete[:, u], origin.reported[:, u]
            prior = var[:, u - 1] + origin.sigma**2 if u else var[:, 0]
            noise = 1.0 / (share + y)
            gain = np.where(after & (share > 0), prior / (prior + noise), 0.0)
            seen = np.log1p(y / np.maximum(share, 1e-12))
            prev = level[:, :, u - 1] if u else level[:, :, 0]
            step = prev + gain[:, None] * (seen[:, None] - prev)
            level[:, :, u] = np.where(after[:, None], step, level[:, :, u])
            var[:, u] = np.where(after, (1.0 - gain) * prior, var[:, u])
        level[:, :, cw + 1 :] = level[:, :, cw : cw + 1]
        return level, var
    rw._filter = _filter
from bdbvhubrenewal.cli import main
sys.exit(main(sys.argv[1:]))
