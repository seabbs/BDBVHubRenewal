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
