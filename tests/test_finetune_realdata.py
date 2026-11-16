"""Training blocks cut from the German data, for the German fine-tune."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

FIRST = dt.date(2021, 4, 6)
DAYS = 150
FILE_DATE = FIRST + dt.timedelta(days=DAYS - 1)
STATES = ("DE-BY", "DE-BE", "DE-HH")
AGES = ("00-04", "05-14", "15-34", "35-59", "60-79", "80+")


def _wide(seed: int = 1, late: bool = True) -> pd.DataFrame:
    """A hub triangle: states and ages that each sum to Germany.

    Changes are known up to `FILE_DATE`; `value_>80d` holds a large
    change when `late`, as the hub's file does for its oldest dates.
    """
    rng = np.random.default_rng(seed)
    cols = [f"value_{d}d" for d in range(81)]
    rows = []
    for i in range(DAYS):
        day = FIRST + dt.timedelta(days=i)
        known = min(80, (FILE_DATE - day).days)
        states = rng.poisson(3.0, (len(STATES), 81)).astype(float)
        # Corrections are kept as reported.
        states[:, 5] -= 1
        states[:, known + 1 :] = np.nan
        total = np.nansum(states, axis=0)
        total[known + 1 :] = np.nan
        # Ages split Germany's changes, so they sum to it too.
        ages = np.tile(np.floor(np.nan_to_num(total) / len(AGES)), (6, 1))
        ages[0] += np.nan_to_num(total) - ages.sum(axis=0)
        ages[:, known + 1 :] = np.nan
        for loc, age, values in (
            ("DE", "00+", total),
            *(("DE", a, v) for a, v in zip(AGES, ages, strict=True)),
            *((s, "00+", v) for s, v in zip(STATES, states, strict=True)),
        ):
            rows.append(
                {
                    "date": day.isoformat(),
                    "location": loc,
                    "age_group": age,
                    **dict(zip(cols, values, strict=True)),
                    "value_>80d": 1000.0 if late and known > 80 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_german_counts_hold_only_reports_by_the_date():
    from finetune_chronos import german_counts

    counts = german_counts(_wide(), "2021-07-31")
    assert counts["report_date"].max() <= "2021-07-31"
    assert set(counts.columns) >= {
        "location",
        "reference_date",
        "report_date",
        "count",
    }
    # Places are the tasks' codes, with no Germany row: it is their sum.
    assert "CD" not in set(counts["location"])
    assert {"CD-BY", "CD-A0"} <= set(counts["location"])
    # Negative changes stay as reported.
    assert (counts["count"] < 0).any()


def test_german_counts_leave_out_changes_after_80_days():
    from finetune_chronos import german_counts

    a = german_counts(_wide(late=True), FILE_DATE.isoformat())
    b = german_counts(_wide(late=False), FILE_DATE.isoformat())
    pd.testing.assert_frame_equal(a, b)


def test_german_blocks_cover_sources_places_and_a_40_day_lag():
    from finetune_chronos import GERMAN_LAGS, german_blocks, german_counts

    assert 40 in GERMAN_LAGS["daily"]
    counts = german_counts(_wide(), FILE_DATE.isoformat())
    cuts = [(FIRST + dt.timedelta(days=d)).isoformat() for d in range(60, 70)]
    blocks = german_blocks(counts, cuts, FILE_DATE.isoformat())
    assert {b["source"] for b in blocks} == {"weekly", "daily"}
    assert {b["places"] for b in blocks} == {"national", "states", "ages"}
    assert 40 in {b["lag"] for b in blocks if b["source"] == "daily"}
    states = next(b for b in blocks if b["places"] == "states")
    assert states["n_targets"] == 1 + len(STATES)
    assert states["levels"] == ("national", *["leaf"] * len(STATES))


def test_german_blocks_refuse_truth_after_the_truth_date():
    from finetune_chronos import german_blocks, german_counts

    counts = german_counts(_wide(), FILE_DATE.isoformat())
    cut = (FIRST + dt.timedelta(days=80)).isoformat()
    # The horizon ends four weeks after the cut, past this truth date.
    truth = (FIRST + dt.timedelta(days=100)).isoformat()
    with pytest.raises(ValueError, match="truth"):
        german_blocks(counts, [cut], truth)
    blocks = german_blocks(
        counts, [cut], (FIRST + dt.timedelta(108)).isoformat()
    )
    assert all(
        b["truth_end"] <= (FIRST + dt.timedelta(108)).isoformat()
        for b in blocks
    )


def test_german_blocks_use_no_report_after_the_data_date():
    from finetune_chronos import german_blocks, german_counts

    as_of = (FIRST + dt.timedelta(days=120)).isoformat()
    cuts = [(FIRST + dt.timedelta(days=70)).isoformat()]
    a = german_blocks(german_counts(_wide(1), as_of), cuts, as_of)
    later = _wide(1)
    # Changes reported after the data date differ.
    day = pd.to_datetime(later["date"])
    for d in range(81):
        after = (day + pd.Timedelta(days=d)).dt.date.astype(str) > as_of
        later.loc[after, f"value_{d}d"] += 7
    b = german_blocks(german_counts(later, as_of), cuts, as_of)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x["context"], y["context"], equal_nan=True)


class _Capture:
    """A sampler that keeps the group Chronos-2 would be given."""

    def __call__(self, groups, horizon, n, seed):  # noqa: ARG002
        self.group, self.horizon = groups[0], horizon
        rows = len(groups[0]["target"])
        return [np.zeros((n, rows, horizon))]


@pytest.mark.parametrize(("source", "lag"), [("daily", 28), ("weekly", 4)])
@pytest.mark.parametrize("task_index", [0, 1])
def test_german_blocks_see_what_the_ragged_model_sees(
    tmp_path, source, lag, task_index
):
    from finetune_chronos import CONTEXT, german_blocks, german_counts

    from bdbvhubrenewal.data import germany
    from bdbvhubrenewal.forecasters.foundation import FoundationModel

    wide = _wide(late=False)
    path = tmp_path / "triangle.csv"
    wide.to_csv(path, index=False)
    # A Wednesday, so dates move three days to a Saturday.
    day = "2021-07-28"
    task = germany.nowcast_tasks(
        germany.hub_triangle(path), day, n_samples=2, seed=1
    )[task_index]
    capture = _Capture()
    FoundationModel("m", source, "ragged", sample=capture).forecast(task)
    group = capture.group
    names = list(group["past_covariates"])
    blocks = german_blocks(
        germany_counts := german_counts(wide, FILE_DATE.isoformat()),
        [day],
        FILE_DATE.isoformat(),
        lags={"weekly": (lag,), "daily": (lag,)},
        sources=(source,),
    )
    assert len(germany_counts) > 0
    block = next(
        b for b in blocks if b["places"] == ("states", "ages")[task_index]
    )
    n = block["n_targets"]
    start = group["target"].shape[1]
    keep = min(start, CONTEXT)
    ctx = block["context"]
    # The targets' context and each lag series up to the reference date.
    assert np.array_equal(
        ctx[:n, CONTEXT - keep : CONTEXT],
        group["target"][:, start - keep :],
        equal_nan=True,
    )
    past = np.stack([group["past_covariates"][k] for k in names])
    future = np.stack([group["future_covariates"][k] for k in names])
    assert np.array_equal(
        ctx[n:, CONTEXT - keep : CONTEXT],
        past[:, start - keep :],
        equal_nan=True,
    )
    h = capture.horizon
    assert np.array_equal(
        ctx[n:, CONTEXT : CONTEXT + h], future, equal_nan=True
    )


def test_german_fine_tune_uses_only_the_development_half():
    from finetune_realdata import AS_OF, FIRST_CUT, cut_dates, truth_until

    # The development half ends the day before the held-out half starts.
    assert AS_OF == "2022-02-09"
    assert truth_until() == "2021-12-31"
    cuts = cut_dates(FIRST_CUT, truth_until())
    # The last cut's four weeks ahead end on the truth date.
    assert cuts[-1] == "2021-12-03"
    assert cuts[0] == FIRST_CUT
    assert len(set(cuts)) == len(cuts)


def test_the_latest_cuts_are_held_out_not_trained_on():
    from finetune_realdata import HELD_OUT_DAYS, split_held_out

    days = [f"2021-11-{d:02d}" for d in range(1, 31)]
    blocks = [{"cut": d, "source": "daily"} for d in days]
    train, held = split_held_out(blocks)
    assert len(held) == HELD_OUT_DAYS["germany"]
    assert max(b["cut"] for b in train) < min(b["cut"] for b in held)


def _bdbv_daily() -> pd.DataFrame:
    """Daily releases of two provinces and the country, their sum."""
    rng = np.random.default_rng(3)
    cutoffs = pd.date_range("2026-08-05", "2026-11-05", freq="7D")
    onsets = pd.date_range("2026-06-01", "2026-11-05", freq="D")
    rows = []
    base = {p: rng.poisson(4.0, len(onsets)) for p in ("CD-IT", "CD-NK")}
    for cut in cutoffs:
        for p, counts in base.items():
            for o, n in zip(onsets, counts, strict=True):
                if o > cut:
                    continue
                age = (cut - o).days
                seen = int(n * min(1.0, (age + 1) / 20))
                rows.append((cut, p, o, seen))
    df = pd.DataFrame(rows, columns=["cutoff", "location", "onset", "n"])
    nat = df.groupby(["cutoff", "onset"], as_index=False)["n"].sum()
    df = pd.concat([df, nat.assign(location="CD")], ignore_index=True)
    day = df["cutoff"].dt.date.astype(str)
    return pd.DataFrame(
        {
            "vintage": day,
            "location": df["location"],
            "onset_date": df["onset"].dt.date.astype(str),
            "observed": df["n"],
            "imputed": 0,
            "cutoff": day,
        }
    )


def test_bdbv_daily_blocks_see_what_the_ragged_model_sees():
    from finetune_chronos import CONTEXT, bdbv_daily_blocks

    from bdbvhubrenewal.forecasters.foundation import (
        hub_triangle,
        layout_block,
    )

    rows = _bdbv_daily()
    cut = "2026-09-30"
    block = next(
        b
        for b in bdbv_daily_blocks(rows, "2026-10-31", lags=(28,), cuts=[cut])
        if b["places"] == "provinces"
    )
    seen = rows.loc[rows["cutoff"] <= cut]
    places = ("CD", "CD-IT", "CD-NK")
    tri = hub_triangle(seen, "daily", places, cut, max_lag=28)
    group = layout_block(tri, "ragged", places, tri.n_context)
    start, n = tri.n_context, 3
    keep = min(start, CONTEXT)
    ctx = block["context"]
    assert np.array_equal(
        ctx[:n, CONTEXT - keep : CONTEXT], group["target"][:, start - keep :]
    )
    past = np.stack(list(group["past_covariates"].values()))
    assert np.array_equal(
        ctx[n:, CONTEXT - keep : CONTEXT],
        past[:, start - keep :],
        equal_nan=True,
    )


def test_bdbv_truth_after_the_truth_date_is_masked():
    from finetune_chronos import bdbv_daily_blocks

    rows = _bdbv_daily()
    truth_until = "2026-10-01"
    blocks = bdbv_daily_blocks(rows, truth_until)
    assert blocks
    for b in blocks:
        assert b["truth_end"] <= truth_until
        times = b["times"]
        n = b["n_targets"]
        late = [i for i, t in enumerate(times) if t > truth_until]
        assert np.isnan(b["context"][:n, late]).all()
    # A cut whose every truth step is too young gives no block; a long lag
    # reaches back to older steps.
    assert not bdbv_daily_blocks(
        rows, truth_until, lags=(14,), cuts=["2026-11-04"]
    )
    assert bdbv_daily_blocks(rows, truth_until, lags=(49,), cuts=["2026-11-04"])


def test_bdbv_weekly_blocks_take_every_lag_and_mask_young_truth():
    from finetune_chronos import GERMAN_LAGS, bdbv_weekly_blocks

    rows = []
    for week in pd.date_range("2026-03-07", "2026-10-31", freq="7D"):
        for place, n in (("CD-IT", 6), ("CD-NK", 3)):
            for k in range(n):
                rows.append((place, week.date().isoformat(), 2 + 3 * k))
    line_list = pd.DataFrame(rows, columns=["location", "onset_week", "delay"])
    blocks = bdbv_weekly_blocks(line_list, "2026-11-05", "2026-10-01")
    assert {b["lag"] for b in blocks} == set(GERMAN_LAGS["weekly"])
    assert {b["places"] for b in blocks} == {"national", "provinces"}
    assert {b["data"] for b in blocks} == {"bdbv"}
    assert max(b["truth_end"] for b in blocks) <= "2026-10-01"
    # Cases reported after the release's cut-off are not seen.
    assert all(b["cut"] <= "2026-11-05" for b in blocks)


def test_held_out_cuts_are_taken_per_dataset():
    from finetune_realdata import split_held_out

    german = [
        {"cut": f"2021-11-{d:02d}", "data": "germany"} for d in range(1, 31)
    ]
    bdbv = [{"cut": f"2026-10-{d:02d}", "data": "bdbv"} for d in (3, 10)]
    train, held = split_held_out(german + bdbv)
    assert {b["data"] for b in held} == {"germany"}
    assert len(held) == 14
    assert all(b in train for b in bdbv)
