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

if V == "OLDNOW":
    from bdbvhubrenewal.baselinenowcast import growth
    from bdbvhubrenewal.data.snapshots import week_sum

    def week_nowcast(snap, r, g, age, gap):
        """The old nowcast: each day's count times its growth, summed to weeks."""
        grow = growth(g, age, age + gap)
        return week_sum(snap, np.where(snap.observed[r][None, :], snap.counts[:, r, :] * grow, 0.0))
    chain.week_nowcast = week_nowcast

if V == "anc":
    def source_index(level, parent, n_own=None, n_pooled=None, min_level=10.0, min_err=3):
        """A place below the rule uses its nearest qualifying ancestor's own estimate."""
        level = np.asarray(level, dtype=float); parent = np.asarray(parent, dtype=int)
        n = level.size
        big = (level >= min_level)[:, None]
        ok = big if n_own is None else big & (np.asarray(n_own) >= min_err)
        out = np.empty((n, ok.shape[1]), dtype=int)
        for node in range(n):
            root = node
            while parent[root] >= 0:
                root = parent[root]
            for k in range(ok.shape[1]):
                a = node
                while a >= 0 and not ok[a, k]:
                    a = parent[a]
                out[node, k] = a if a >= 0 else root
        return out[:, 0] if n_own is None else out
    chain.source_index = source_index

if V == "ancnb":
    from bdbvhubrenewal.forecasters import pooling as _pool
    _orig = _pool.source_index
    def source_index(level, parent, n_own=None, n_pooled=None, min_level=10.0, min_err=3):
        """Only the NB size: a place below the rule uses its nearest qualifying ancestor's own size."""
        if n_own is None:
            return _orig(level, parent, min_level=min_level)
        level = np.asarray(level, dtype=float); parent = np.asarray(parent, dtype=int)
        ok = (level >= min_level)[:, None] & (np.asarray(n_own) >= min_err)
        out = np.empty(ok.shape, dtype=int)
        for node in range(level.size):
            root = node
            while parent[root] >= 0:
                root = parent[root]
            for k in range(ok.shape[1]):
                a = node
                while a >= 0 and not ok[a, k]:
                    a = parent[a]
                out[node, k] = a if a >= 0 else root
        return out
    chain.source_index = source_index

if V == "noquiet":
    import dataclasses as _dc
    _orig_add, _orig_retro = chain._add_spread, chain.retro_errors
    _mask = {}
    def _retro(snap, o, settings, delay=chain.fit_delay):
        e = _orig_retro(snap, o, settings, delay)
        return _dc.replace(e, valid=e.valid & ~_mask["silent"][None, :, None])
    def _add(origin):
        _mask["silent"] = origin.silent
        return _orig_add(origin)
    chain.retro_errors = _retro
    chain._add_spread = _add

if V.startswith("blk"):
    _k = float(V[3:])
    _lf = chain.log_factors
    chain.log_factors = lambda num, den, gaps, min_count=0.0: _lf(num, den, gaps, _k)

if V == "skipzero":
    from bdbvhubrenewal.baselinenowcast import snapshots as _sn
    _ab = _sn.age_blocks
    def age_blocks(den, min_count):
        """Blocks over the ages with cases; an age with no case is its own block."""
        total = np.asarray(den, float).sum(axis=0)
        has = total > 0
        block = np.full(total.size, -1)
        inner = _ab(np.asarray(den)[:, has], min_count)
        block[has] = inner
        n = inner.max() + 1 if inner.size else 0
        block[~has] = n + np.arange((~has).sum())
        return block
    _sn.age_blocks = age_blocks
