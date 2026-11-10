"""Decompose randomwalknowcast at the 11-08 release (round 11-14).

Run from a worktree root: uv run python -I decomp.py OUTDIR [NSAMP]
"""

import argparse
import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from bdbvhubrenewal import cli  # noqa: E402
from bdbvhubrenewal.forecasters import randomwalk as rw  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["main"]

ns = argparse.Namespace(
    vintage="2026-11-08",
    reference_date="2026-11-14",
    data="data",
    backtest_config="config/backtest.toml",
)
task, _ = cli._task(ns)
task = dataclasses.replace(task, n_samples=nsamp)
m = rw.RandomWalkNowcast()
origin = m.origin_at(task, errors=True)
snap = origin.snap
locs = list(snap.locations)
cw = snap.cut_week(origin.o)
print("cut", snap.cut[origin.o], "cw", snap.week_ends[cw], "n_weeks", snap.n_weeks)
print("start_nowcast", snap.week_ends[origin.start_nowcast])
w0 = rw._start_weeks(origin)
quiet = rw._quiet(origin)
rows = []
for i, c in enumerate(locs):
    rows.append(
        dict(
            location=c,
            parent=int(snap.parent[i]),
            quiet=bool(quiet[i]),
            silent=bool(origin.silent[i]),
            start_rw=snap.week_ends[int(origin.start[i])],
            w0=snap.week_ends[int(w0[i])],
            sigma=float(origin.sigma[i]),
            sigma_src=rw._label(origin, int(origin.sigma_source[i])),
            delay_src=rw._label(origin, int(origin.fit.source[i])),
            intro=float(origin.intro[i]),
            **{
                f"size_k{k}": float(origin.size[i, k])
                for k in range(origin.size.shape[1])
            },
            **{
                f"rem_k{k}": float(origin.removal[i, k])
                for k in range(origin.removal.shape[1])
            },
        )
    )
pd.DataFrame(rows).to_csv(out / "places.csv", index=False)
# reported, point, complete for last 8 weeks
wk = list(range(cw - 7, cw + 1))
tab = []
for i, c in enumerate(locs):
    for w in wk:
        tab.append(
            dict(
                location=c,
                week=snap.week_ends[w],
                reported=origin.reported[i, w],
                point=origin.point[i, w],
                complete=origin.complete[i, w],
            )
        )
pd.DataFrame(tab).to_csv(out / "nowcast_point.csv", index=False)


def run(origin, tag, pre150=False):
    if pre150:
        # The pre-#150 behaviour: weeks after the cut-off from the walk of
        # each sample's start-week nowcast.
        values, weeks = traj_pre150(m, origin, task)
    else:
        values, weeks = m.trajectories(origin, task, origin.point)
    np.save(out / f"values_{tag}.npy", values)
    (out / "weeks.txt").write_text("\n".join(weeks))
    return values, weeks


def traj_pre150(self, origin, task):
    import jax
    import jax.numpy as jnp

    from bdbvhubrenewal.forecasters.obs import sample_count

    snap = origin.snap
    point = origin.point
    weeks = rw._timeline(origin, task)
    n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
    n_loc = len(snap.locations)
    k_n = origin.size.shape[1]
    back = snap.cut_week(origin.o) - np.arange(n_data)
    k = np.clip(back, 0, k_n - 1)
    b = np.where(back < k_n, origin.removal[:, k], 0.0)
    key_x, key_walk, key_part, key_ahead = jax.random.split(rw._key(task), 4)
    shape = (n, *point.shape)
    reported = jnp.broadcast_to(jnp.asarray(origin.reported, jnp.float32), shape)
    nowcast = np.asarray(
        sample_count(
            key_x,
            reported,
            jnp.maximum(jnp.asarray(point, jnp.float32) - reported, 0.0),
            jnp.broadcast_to(jnp.asarray(origin.size[:, k], jnp.float32), shape),
            jnp.broadcast_to(jnp.asarray(b, jnp.float32), shape),
        ),
        np.float64,
    )
    cw = snap.cut_week(origin.o)
    w0 = rw._start_weeks(origin)
    quiet = rw._quiet(origin)
    rows = np.arange(n_loc)
    lam, y = rw._walks(key_walk, nowcast[:, rows, w0].T, origin, w0, n_all, quiet)
    cut = int(snap.cut[origin.o])
    miss = int(np.flatnonzero(snap.week == cw).max()) - cut
    at_cw = np.clip(cw - w0 - 1, 0, n_all - 1)
    part = np.asarray(
        jax.random.poisson(
            key_part, jnp.asarray(lam[rows, :, at_cw] * miss / 7, jnp.float32)
        ),
        dtype=np.float64,
    )
    values = np.zeros((n, n_loc, n_all))
    for loc in range(n_loc):
        s = int(w0[loc])
        if quiet[loc]:
            values[:, loc, : s + 1] = origin.reported[loc, : s + 1]
            values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
            values[:, loc, :n_data] = np.maximum(
                values[:, loc, :n_data], origin.reported[loc]
            )
            continue
        values[:, loc, : cw + 1] = nowcast[:, loc, : cw + 1]
        if miss > 0 and cw > s:
            values[:, loc, cw] += part[loc]
        values[:, loc, cw + 1 :] = y[loc, :, cw - s : n_all - s - 1]
    return values, weeks


for v in variants:
    o = copy.deepcopy(origin)
    pre = False
    for part in v.split("+"):
        if part == "main":
            pass
        elif part == "pre150":
            pre = True
        elif part == "nonb":  # nowcast without NB/removal noise
            o.size = np.full_like(o.size, 1e7)
            o.removal = np.zeros_like(o.removal)
        elif part == "nowalk":
            o.sigma = np.full_like(o.sigma, 1e-4)
        elif part == "noplus1":
            # point without apply_delay's +1: scale by 1/share
            from bdbvhubrenewal.data.snapshots import week_sum

            g = o.growth
            seen = np.broadcast_to(snap.observed[o.o], g.shape)
            x = week_sum(snap, np.where(seen, snap.counts[:, o.o, :], 0.0))
            nd = week_sum(snap, seen.astype(float))
            sh = week_sum(snap, np.where(seen, 1.0 / g, 0.0))
            share = np.where(nd > 0, sh / np.maximum(nd, 1), 1.0)
            o.point = np.where(nd > 0, x / share, 0.0)
        else:
            raise SystemExit(f"unknown {part}")
    run(o, v.replace("+", "_"), pre)
    print("done", v)
(out / "locs.txt").write_text("\n".join(locs))
