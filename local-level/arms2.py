"""Run the CLI with variant patches: OWNSIG (own walk spread), NBPRIOR (no Poisson-end size)."""
import os, sys
import numpy as np
from bdbvhubrenewal.forecasters import chain
from bdbvhubrenewal.forecasters import randomwalk as rw
_bo = rw.build_origin
def _build(*a, **k):
    o = _bo(*a, **k)
    if os.environ.get("OWNSIG"):
        own, _ = chain._walk_spreads(o.reported, o.start, o.snap, o.fit.level, o.settings)
        o.sigma = np.where(own > 0, own, o.sigma)
    if os.environ.get("NBPRIOR") and o.size is not None:
        o.size = np.where(o.size >= 1000, 2.2, o.size)
    return o
rw.build_origin = _build
from bdbvhubrenewal.cli import main
sys.exit(main(sys.argv[1:]))
