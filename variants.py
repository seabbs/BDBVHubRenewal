"""Experimental variants, applied by monkeypatching. VARIANT in {base, A, B, AB}."""
import os
import numpy as np
import jax, jax.numpy as jnp
from bdbvhubrenewal.forecasters import chain, randomwalk as rwmod
from bdbvhubrenewal.forecasters.obs import sample_count
from bdbvhubrenewal.forecasters.pooling import pool_matrix

V = os.environ.get("VARIANT", "base")
C = float(os.environ.get("START_PI", "0.5"))

if "A" in V:
    def start_nowcast(self):
        root = int(np.flatnonzero(self.snap.parent < 0)[0])
        ok = np.flatnonzero(self.complete[root] >= C)
        return int(ok.max()) if ok.size else 0
    chain.Origin.start_nowcast = property(start_nowcast)

    def trajectories(self, origin, task, point):
        snap = origin.snap
        weeks = rwmod._timeline(origin, task)
        n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(snap.locations); k_n = origin.size.shape[1]
        back = snap.cut_week(origin.o) - np.arange(n_data)
        k = np.clip(back, 0, k_n - 1)
        b = np.where(back < k_n, origin.removal[:, k], 0.0)
        key_x, key_walk, _ = jax.random.split(rwmod._key(task), 3)
        shape = (n, *point.shape)
        reported = jnp.broadcast_to(jnp.asarray(origin.reported, jnp.float32), shape)
        nowcast = np.asarray(sample_count(key_x, reported,
            jnp.maximum(jnp.asarray(point, jnp.float32) - reported, 0.0),
            jnp.broadcast_to(jnp.asarray(origin.size[:, k], jnp.float32), shape),
            jnp.broadcast_to(jnp.asarray(b, jnp.float32), shape)), np.float64)
        w0 = origin.start_nowcast
        _, y = rwmod._walks(key_walk, nowcast[:, :, w0].T, origin.sigma, n_all)
        values = np.zeros((n, n_loc, n_all))
        for u in range(n_all):
            if u <= w0:
                values[:, :, u] = nowcast[:, :, u]
            else:
                floor = origin.reported[:, u] if u < n_data else 0.0
                values[:, :, u] = np.maximum(y[:, :, u - w0 - 1].T, floor)
        return values, weeks
    rwmod.RandomWalkNowcast.trajectories = trajectories

if "B" in V:
    def _walk_spreads(reported, start, snap, level, settings):
        n_loc = reported.shape[0]
        diffs = chain.walk_changes(reported, start, settings)
        noise = []
        for n in range(n_loc):
            lo = max(int(start[n]) - settings.walk_weeks + 1, 0)
            y = reported[n, lo:int(start[n]) + 1]
            noise.append(1 / (1 + y[1:]) + 1 / (1 + y[:-1]))
        def sd(d, e):
            if d.size < 2:
                return 0.0
            return float(np.sqrt(max(np.var(d, ddof=1) - np.mean(e), 0.0)))
        own = np.array([sd(diffs[n], noise[n]) for n in range(n_loc)])
        pool = pool_matrix(level, snap.parent, settings.min_level)
        pooled = np.array([sd(np.concatenate([diffs[m] for m in np.flatnonzero(pool[n])]),
                              np.concatenate([noise[m] for m in np.flatnonzero(pool[n])]))
                           for n in range(n_loc)])
        return own, pooled
    chain._walk_spreads = _walk_spreads
