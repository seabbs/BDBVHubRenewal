# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check the lag covariates of real training blocks against raw counts.

Cuts blocks of one training simulation as the trainer does
(`finetune_stream.cut`), records each `training_block` call, and
recomputes every target and lag cell by brute force from the raw
reporting triangle: the target's context is the count of each step as
of the cut-off, its horizon the final count; lag `d` of step `t` is the
count reported by the release whose cut-off falls `d` steps after `t`,
NaN where no release had its cut-off then or that release is after the
cut. It then checks Chronos-2's validation slice of the padded block:
the context ends at the step before the nowcast, the targets of the
nowcast and forecast steps are only in `future_target`, and the
future covariates are the lag rows over the horizon.

    python scripts/covariate_alignment.py SIM_DIR SEED [N_BLOCKS]
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.forecasters import finetune as fin

ANY_RELEASE = False
"""See `brute`."""


def recorded_blocks(sim_dir: Path, seed: int, n: int) -> list[dict]:
    """Cut a simulation's blocks, keeping each call's arguments."""
    calls = []
    original = ft.training_block

    def spy(counts, source, places, cutoffs, max_lag, totals=False):
        block = original(
            counts, source, places, cutoffs, max_lag, totals=totals
        )
        if block is not None:
            calls.append(
                dict(
                    counts=counts,
                    source=source,
                    places=places,
                    cutoffs=list(cutoffs),
                    lag=max_lag,
                    totals=totals,
                    block=block,
                )
            )
        return block

    ft.training_block = spy
    fs.init_worker(
        min(ft.MAX_SERIES, ft.max_series(ft.MEMORY_GB)),
        ft.MAX_LAGS,
        ("weekly", "daily", "3day"),
        0.2,
        2,
        1,
    )
    sims = sorted(pd.read_csv(sim_dir / f"settings-{seed}.csv")["outbreak"])
    for sim in sims:
        fs.cut(sim_dir, [sim], 1, 32)
        sources = {c["source"] for c in calls}
        if len(calls) >= n and {"weekly", "daily", "3day"} <= sources:
            break
    ft.training_block = original
    # One of each source, multi-place first.
    out = []
    for source in ("weekly", "daily", "3day"):
        mine = [c for c in calls if c["source"] == source]
        mine.sort(key=lambda c: -len(c["places"]))
        out.append(mine[0])
    return out


def brute(call: dict) -> dict:
    """Recompute the block's targets and lags from raw counts."""
    counts = call["counts"].copy()
    source, places, lag = call["source"], call["places"], call["lag"]
    cutoffs = [dt.date.fromisoformat(c) for c in call["cutoffs"]]
    cut = cutoffs[-1]
    counts["reference_date"] = pd.to_datetime(counts["reference_date"]).dt.date
    counts["report_date"] = pd.to_datetime(counts["report_date"]).dt.date
    # Each place's rows, read independently of `releases`: the country is
    # every row, a total `P` its strata `P-*`, an age group `A<k>` its
    # strata `*-A<k>`.
    from bdbvhubrenewal.forecasters.base import NATIONAL

    loc = counts["location"].astype(str)

    def members(p: str) -> pd.Series:
        if p == NATIONAL:
            return pd.Series(True, index=counts.index)
        return (
            (loc == p) | loc.str.startswith(p + "-") | loc.str.endswith("-" + p)
        )

    rows = pd.concat(
        [counts.loc[members(p)].assign(location=p) for p in places]
    )
    width = {"weekly": 7, "daily": 1, "3day": 3}[source]
    end = epiweek_end(cut.isoformat()) + dt.timedelta(weeks=fin.AHEAD_WEEKS)
    if source == "weekly":
        step_end = lambda d: epiweek_end(d.isoformat())  # noqa: E731
        cut_step = step_end(cut)
        last_step = end
    elif source == "daily":
        step_end = lambda d: d  # noqa: E731
        cut_step, last_step = cut, end
    else:
        # Bins end on the cut-off day and every 3 days either side.
        def step_end(d):
            return d + dt.timedelta(days=(cut - d).days % 3)

        cut_step = cut
        last_step = cut + dt.timedelta(days=3 * ((end - cut).days // 3))
    rows["step"] = rows["reference_date"].map(step_end)
    # A step's lag-d cell is the release on the step's last day plus d
    # steps exactly (`aggregate_triangle` for 3-day bins). With
    # `ANY_RELEASE`, a 3-day bin instead takes any release whose cut-off
    # falls in the bin d steps on.
    if source == "3day" and not ANY_RELEASE:
        releases = {c: c for c in cutoffs}
    else:
        releases = {step_end(c): c for c in cutoffs}
    first = rows["step"].min()
    steps = []
    s = first
    while s <= last_step:
        steps.append(s)
        s += dt.timedelta(days=width)
    at = {s: i for i, s in enumerate(steps)}
    last = at[cut_step]
    start = last - lag + 1
    L, T = len(places), len(steps)
    target = np.full((L, T), np.nan)
    lagged = np.full((L, lag, T), np.nan)
    for i, p in enumerate(places):
        mine = rows[rows["location"] == p]
        for t, s in enumerate(steps):
            ref = mine[mine["step"] == s]
            if t < start:
                target[i, t] = ref.loc[ref["report_date"] <= cut, "count"].sum()
            else:
                target[i, t] = ref["count"].sum()
            for d in range(lag):
                rel_step = s + dt.timedelta(days=width * d)
                c = releases.get(rel_step)
                if c is not None and c <= cut:
                    lagged[i, d, t] = ref.loc[
                        ref["report_date"] <= c, "count"
                    ].sum()
    return dict(
        target=target,
        lagged=lagged.reshape(L * lag, T),
        start=start,
        last=last,
        steps=steps,
    )


def check(call: dict) -> dict:
    """Compare a recorded block with the brute-force one."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    block, want = call["block"], brute(call)
    L, lag = len(call["places"]), call["lag"]
    start, h = block["start"], block["horizon"]
    ctx = block["context"]
    T = ctx.shape[1]
    # Align the brute grid's end with the block's end.
    off = len(want["steps"]) - T
    wt, wl = want["target"][:, off:], want["lagged"][:, off:]
    got_t, got_l = ctx[:L], ctx[L:]
    same_t = np.allclose(got_t, wt, equal_nan=True)
    same_l = np.allclose(got_l, wl, equal_nan=True)
    # Chronos-2's validation slice of the padded block.
    padded = ft._pad(block, ft.CONTEXT)
    data = Chronos2Dataset(
        inputs=[fs.tensors(padded)],
        context_length=ft.CONTEXT,
        prediction_length=padded["prediction_length"],
        batch_size=1,
        output_patch_size=ft.OUTPUT_PATCH,
        mode=DatasetMode.VALIDATION,
    )
    b = next(iter(data))
    c = b["context"].numpy()
    ft_ = b["future_target"].numpy()
    fc = b["future_covariates"].numpy()
    ctx_last = c[:L, -1]
    keep = min(start, ft.CONTEXT)
    return {
        "source": call["source"],
        "places": L,
        "lag D": lag,
        "cut-off": call["cutoffs"][-1],
        "steps": T,
        "start": start,
        "horizon": h,
        "nowcast steps": want["last"] - off - start + 1,
        "targets match brute force": same_t,
        "lags match brute force": same_l,
        "brute start == block start": want["start"] - off == start,
        "context ends at step start-1": np.allclose(
            ctx_last, got_t[:, start - 1], equal_nan=True
        ),
        "context target cells == as-of-cut counts": np.allclose(
            c[:L, ft.CONTEXT - keep :],
            wt[:, start - keep : start],
            equal_nan=True,
        ),
        "nowcast/forecast truth only in future_target": np.allclose(
            ft_[:L, :h], wt[:, start : start + h], equal_nan=True
        ),
        "future_target covariate rows all NaN": bool(np.isnan(ft_[L:]).all()),
        "future_covariates target rows all NaN": bool(np.isnan(fc[:L]).all()),
        "future_covariates == lag rows over horizon": np.allclose(
            fc[L:, :h], wl[:, start : start + h], equal_nan=True
        ),
        "max non-NaN future lag cell index vs (last - d)": [
            int(np.max(np.where(~np.isnan(wl[d::lag][0]))[0], initial=-1))
            - (want["last"] - off - d)
            for d in range(lag)
        ],
        "covariate NaN share in history": float(
            np.isnan(got_l[:, :start]).mean()
        ),
        "covariate NaN share in horizon": float(
            np.isnan(got_l[:, start:]).mean()
        ),
        "example place 0 lags at nowcast steps": got_l[
            :lag, start : start + lag
        ]
        .round(0)
        .tolist(),
        "example place 0 truth at nowcast steps": got_t[0, start : start + lag]
        .round(0)
        .tolist(),
    }


def main() -> None:
    sim_dir, seed = Path(sys.argv[1]), int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    global ANY_RELEASE
    calls = recorded_blocks(sim_dir, seed, n)
    for call in calls:
        for k, v in check(call).items():
            if "example" not in k or call["source"] == "weekly":
                print(f"{k}: {v}")
        print()
    # 3-day blocks against the rule that every weekly release counts.
    ANY_RELEASE = True
    for call in calls:
        if call["source"] == "3day":
            want = brute(call)
            got = call["block"]["context"][len(call["places"]) :]
            wl = want["lagged"][:, len(want["steps"]) - got.shape[1] :]
            print(
                "3day, any release in the bin counts: non-NaN lag cells "
                f"{int((~np.isnan(wl)).sum())} vs block's "
                f"{int((~np.isnan(got)).sum())}; releases used by block "
                f"{len({c for c in call['cutoffs'] if (dt.date.fromisoformat(call['cutoffs'][-1]) - dt.date.fromisoformat(c)).days % 3 == 0})}"
                f" of {len(call['cutoffs'])}"
            )


if __name__ == "__main__":
    main()
