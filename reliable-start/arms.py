"""Run the bdbvhubrenewal CLI with start thresholds from ARM=share,count[,or]."""
import os, sys
from bdbvhubrenewal.forecasters import chain
arm = os.environ.get("ARM")
if arm:
    share, count = (float(x) for x in arm.split(",")[:2])
    d = list(chain.Settings.__init__.__defaults__)
    d[-2], d[-1] = share, count
    chain.Settings.__init__.__defaults__ = tuple(d)
    if arm.endswith(",or"):
        from bdbvhubrenewal.forecasters import randomwalk as rw

        def _either(origin):
            st = origin.settings
            return (origin.complete >= st.start_share) | (origin.reported >= st.start_count)
        rw._reliable = _either
if os.environ.get("INTRO_ONLY"):
    import numpy as np
    from bdbvhubrenewal.forecasters import randomwalk as rw
    rw._reliable = lambda origin: np.broadcast_to(
        np.arange(origin.complete.shape[1]) <= origin.start_nowcast, origin.complete.shape
    )
if os.environ.get("OLD_INTRO"):
    import dataclasses
    import numpy as np
    from bdbvhubrenewal.forecasters import randomwalk as rw
    rw._intro_from = lambda origin, start, h: np.broadcast_to(origin.intro[:, None], (origin.intro.size, h))
    _w = rw._walks
    def _walks0(key, x0, origin, intro):
        o = dataclasses.replace(origin, sigma=np.where(origin.silent, 0.0, origin.sigma))
        return _w(key, x0, o, intro)
    rw._walks = _walks0
from bdbvhubrenewal.cli import main
sys.exit(main(sys.argv[1:]))
