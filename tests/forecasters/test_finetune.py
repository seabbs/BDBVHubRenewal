"""Training blocks for fine-tuning Chronos-2 on simulated outbreaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.finetune import (
    final_counts,
    releases,
    training_block,
    weekly_cutoffs,
)
from bdbvhubrenewal.forecasters.foundation import hub_triangle


def _counts() -> pd.DataFrame:
    """Two places' triangle: onsets 10-05 to 10-14, reported 0 to 9 days on."""
    rows = []
    for place, scale in (("A", 1), ("B", 2)):
        for onset in pd.date_range("2026-10-05", "2026-10-14"):
            for delay in (0, 3, 9):
                report = onset + pd.Timedelta(days=delay)
                rows.append((place, onset, report, scale))
    return pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )


def test_hub_triangle_takes_max_lag() -> None:
    rows = releases(_counts(), "weekly", ["2026-10-17", "2026-10-24"])
    tri = hub_triangle(rows, "weekly", (NATIONAL,), "2026-10-31", max_lag=2)
    assert tri.partial.shape == (1, 4, 2)
    assert tri.n_context == tri.last - 1


def test_releases_count_what_was_reported_by_each_cutoff() -> None:
    rows = releases(_counts(), "daily", ["2026-10-07", "2026-10-30"])
    first = rows.loc[
        (rows["vintage"] == "2026-10-07") & (rows["location"] == "A")
    ]
    # Onset 10-05 had its delay-0 report by 10-07, 10-07 its delay-0.
    assert first.set_index("onset_date")["observed"].to_dict() == {
        "2026-10-05": 1,
        "2026-10-06": 1,
        "2026-10-07": 1,
    }
    last = rows.loc[rows["vintage"] == "2026-10-30"]
    # The country is the sum of the places, every report in.
    country = last.loc[last["location"] == NATIONAL, "observed"]
    assert (country == 9).all()
    assert (rows["imputed"] == 0).all()


def test_weekly_releases_are_by_onset_week() -> None:
    rows = releases(_counts(), "weekly", ["2026-10-30"])
    a = rows.loc[rows["location"] == "A"].set_index("target_end_date")
    # Onsets 10-05 to 10-10 fall in the week ending Saturday 10-10.
    assert a["total"].to_dict() == {"2026-10-10": 18, "2026-10-17": 12}


def test_final_counts_line_up_with_the_grid() -> None:
    out = final_counts(
        _counts(), "daily", (NATIONAL, "A", "B"), ["2026-10-04", "2026-10-05"]
    )
    assert out.tolist() == [[0.0, 9.0], [0.0, 3.0], [0.0, 6.0]]


def test_cutoffs_fall_on_the_weekday() -> None:
    assert weekly_cutoffs("2026-10-05", "2026-10-22", 3) == [
        "2026-10-08",
        "2026-10-15",
        "2026-10-22",
    ]


def test_block_is_the_ragged_layout_with_final_counts_ahead() -> None:
    places = (NATIONAL, "A", "B")
    cutoffs = ["2026-10-07", "2026-10-09"]
    block = training_block(_counts(), "daily", places, cutoffs, max_lag=3)
    start, horizon = block["start"], block["horizon"]
    context = block["context"]
    assert block["n_targets"] == 3
    assert block["n_covariates"] == block["n_future_covariates"] == 9
    assert context.shape == (12, start + horizon)
    # Cut-off 10-09 with D = 3 gives context to 10-06 from onset 10-05.
    assert start == 2
    # The horizon runs to the Saturday four weeks after the cut-off's week.
    assert (
        horizon
        == (pd.Timestamp("2026-11-07") - pd.Timestamp("2026-10-07")).days + 1
    )
    # Targets past the context are the final counts.
    assert context[0, start : start + 8].tolist() == [9.0] * 8
    assert np.all(context[0, start + 8 :] == 0.0)
    # Lag 0 of the country at 10-09 is the cut-off's same-day report.
    assert context[3, 4] == 3.0
    # Lag 0 at 10-08 is missing: no release had its cut-off on 10-08.
    assert np.isnan(context[3, 3])
    assert np.array_equal(
        block["future_covariates"], context[:, start:], equal_nan=True
    )


def test_releases_add_each_place_total_of_its_strata() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
    rows = releases(counts, "daily", ["2026-10-30"], totals=True)
    by = rows.groupby("location")["observed"].sum()
    # P1 is the sum of its two strata, and the country counts them once.
    assert by["P1"] == by["P1-A1"] + by["P1-A2"]
    assert by[NATIONAL] == by["P1"]


def test_three_level_block_has_country_place_and_strata() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
    cutoffs = weekly_cutoffs("2026-10-05", "2026-10-21", 2)
    places = (NATIONAL, "P1", "P1-A1", "P1-A2")
    block = training_block(
        counts, "daily", places, cutoffs, max_lag=3, totals=True
    )
    assert block["n_targets"] == 4
    start = block["start"]
    target = block["context"][:4, start:]
    # The country and P1 are both the sum of the strata.
    np.testing.assert_array_equal(target[0], target[1])
    np.testing.assert_array_equal(target[1], target[2] + target[3])


def test_totals_sum_strata_by_place_and_by_age_exactly() -> None:
    counts = _counts()
    counts = pd.concat(
        [
            counts.assign(
                location=counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
            ),
            counts.assign(
                location=counts["location"].map({"A": "P2-A1", "B": "P2-A2"})
            ),
        ]
    )
    rows = releases(counts, "daily", ["2026-10-09", "2026-10-30"], totals=True)
    wide = rows.pivot_table(
        index=["vintage", "onset_date"],
        columns="location",
        values="observed",
        aggfunc="sum",
        fill_value=0,
    )
    leaves = ["P1-A1", "P1-A2", "P2-A1", "P2-A2"]
    assert (wide[NATIONAL] == wide[leaves].sum(axis=1)).all()
    assert (wide["P1"] == wide["P1-A1"] + wide["P1-A2"]).all()
    assert (wide["A1"] == wide["P1-A1"] + wide["P2-A1"]).all()


def test_regions_of_places_total_their_places_exactly() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "R1-P01", "B": "R1-P02"})
    rows = releases(counts, "weekly", ["2026-10-30"], totals=True)
    wide = rows.pivot_table(
        index="target_end_date",
        columns="location",
        values="total",
        aggfunc="sum",
        fill_value=0,
    )
    assert (wide["R1"] == wide["R1-P01"] + wide["R1-P02"]).all()
    # A region's places are not age groups: no total by their second part.
    assert "P01" not in wide.columns


def test_block_of_places_with_no_release_rows_is_none() -> None:
    # Only place A had cases, so its releases hold the country alone.
    counts = _counts().query("location == 'A'")
    cutoffs = ["2026-10-17", "2026-10-24"]
    assert training_block(counts, "daily", ("A",), cutoffs, max_lag=3) is None
    assert training_block(counts, "daily", ("B",), cutoffs, max_lag=3) is None


def _longer_counts() -> pd.DataFrame:
    """One place, the country, onsets over five weeks, reported 0, 2 and
    5 days on."""
    rows = []
    for onset in pd.date_range("2026-09-01", "2026-10-05"):
        for delay, n in ((0, 1), (2, 2), (5, 4)):
            rows.append(("A", onset, onset + pd.Timedelta(days=delay), n))
    return pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )


def test_three_day_triangle_sums_daily_reports_by_each_release() -> None:
    from bdbvhubrenewal.forecasters.finetune import aggregate_triangle

    counts = _longer_counts()
    cutoffs = ["2026-09-23", "2026-09-30"]
    rows = releases(counts, "daily", cutoffs)
    daily = hub_triangle(rows, "daily", (NATIONAL,), "2026-10-05", max_lag=6)
    three = aggregate_triangle(daily, 3, 2)
    ref = pd.to_datetime(counts["reference_date"])
    rep = pd.to_datetime(counts["report_date"])
    # Bins end on the latest cut-off and every 3 days either side of it.
    assert three.times[three.last] == "2026-09-30"
    for j, end in enumerate(three.times):
        e = pd.Timestamp(end)
        days = (ref > e - pd.Timedelta(days=3)) & (ref <= e)
        for k in range(2):
            release = e + pd.Timedelta(days=3 * k)
            got = three.partial[0, j, k]
            if release.date().isoformat() not in cutoffs:
                assert np.isnan(got)
                continue
            # Lag k: the release 3k days after the bin, for all its days.
            assert got == counts.loc[days & (rep <= release), "count"].sum()


def test_three_day_blocks_are_cut_like_daily_ones() -> None:
    counts = _longer_counts()
    cutoffs = ["2026-09-23", "2026-09-30"]
    block = training_block(counts, "3day", (NATIONAL,), cutoffs, max_lag=2)
    assert block["n_targets"] == 1
    assert block["n_covariates"] == 2
    target = block["context"][0]
    # Every full bin of three days holds 21 cases.
    assert target[block["start"] - 1] == 21.0


def test_three_day_block_with_no_whole_bin_by_its_cut_is_none() -> None:
    # Cases start the day before the cut-off: no 3-day bin ends there.
    counts = _longer_counts()
    cutoffs = ["2026-09-02"]
    assert (
        training_block(counts, "3day", (NATIONAL,), cutoffs, max_lag=2) is None
    )


def _two_places() -> pd.DataFrame:
    """Two places, onsets over five weeks, reported 0 and 3 days on."""
    rows = []
    for onset in pd.date_range("2026-09-01", "2026-10-05"):
        for place, n in (("A", 3), ("B", 5)):
            for delay in (0, 3):
                rows.append((place, onset, onset + pd.Timedelta(days=delay), n))
    return pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )


def test_a_parent_need_not_be_the_sum_of_its_children() -> None:
    from bdbvhubrenewal.forecasters.finetune import country_counts

    counts = _two_places()
    rng = np.random.default_rng(1)
    same, total = country_counts(counts, "exact", rng)
    assert same.equals(counts)
    assert total.equals(counts)
    # Unallocated cases are in the country's count but in no place's.
    places, total = country_counts(counts, "unallocated", rng)
    assert total.equals(counts)
    assert 0 < places["count"].sum() < counts["count"].sum()
    # The country reported through its own, later stream.
    places, total = country_counts(counts, "late_parent", rng)
    assert places.equals(counts)
    lag = pd.to_datetime(total["report_date"]) - pd.to_datetime(
        counts["report_date"]
    )
    assert lag.nunique() == 1
    assert lag.iloc[0] > pd.Timedelta(0)


def test_blocks_carry_a_parent_that_does_not_sum() -> None:
    counts = _two_places()
    cutoffs = ["2026-09-23", "2026-09-30"]
    places = (NATIONAL, "A", "B")
    exact = training_block(counts, "weekly", places, cutoffs, max_lag=2)
    loose = training_block(
        counts,
        "weekly",
        places,
        cutoffs,
        max_lag=2,
        parent="unallocated",
        rng=np.random.default_rng(1),
    )
    t = exact["start"] - 1
    assert exact["context"][0, t] == exact["context"][1:3, t].sum()
    assert loose["context"][0, t] > loose["context"][1:3, t].sum()


def test_a_late_country_keeps_its_late_reports_in_its_truth() -> None:
    counts = _two_places()
    country = counts.assign(
        report_date=pd.to_datetime(counts["report_date"]) + pd.Timedelta(days=9)
    )
    days = sorted(
        pd.to_datetime(counts["reference_date"])
        .dt.strftime("%Y-%m-%d")
        .unique()
    )
    truth = final_counts(counts, "daily", (NATIONAL,), days, False, country)
    assert truth.sum() == counts["count"].sum()


@pytest.mark.parametrize(
    ("source", "width"), [("daily+weekly", 7), ("daily+3day", 3)]
)
def test_a_mixed_block_adds_each_places_period_totals_on_daily_steps(
    source, width
) -> None:
    counts = _two_places()
    cutoffs = ["2026-09-23", "2026-09-30"]
    places = (NATIONAL, "A", "B")
    block = training_block(counts, source, places, cutoffs, max_lag=3)
    daily = training_block(counts, "daily", places, cutoffs, max_lag=3)
    n = len(places)
    # Daily counts, then one aggregate row per place, then daily lags.
    assert block["n_targets"] == 2 * n
    assert list(block["target_places"]) == [*range(n), *range(n)]
    assert block["places"] == places
    assert block["n_covariates"] == daily["n_covariates"]
    x, d = block["context"], daily["context"]
    assert np.array_equal(x[:n], d[:n], equal_nan=True)
    assert np.array_equal(x[2 * n :], d[n:], equal_nan=True)
    agg = x[n : 2 * n]
    ends = np.flatnonzero(~np.isnan(agg[0]))
    # A value on each period's last day only, the period's total.
    assert len(ends) > 1
    assert set(np.diff(ends)) == {width}
    assert np.isnan(agg[:, ends[0] + 1 : ends[1]]).all()
    for e in ends:
        assert np.allclose(agg[:, e], d[:n, e - width + 1 : e + 1].sum(1))


def test_a_mixed_blocks_periods_can_end_on_any_day() -> None:
    from bdbvhubrenewal.forecasters.finetune import period_totals

    times = [
        d.date().isoformat() for d in pd.date_range("2026-09-01", periods=30)
    ]
    days = np.ones((1, 30))
    for weekday in range(7):
        out = period_totals(days, times, "weekly", end=weekday)
        ends = [
            pd.Timestamp(times[e]).weekday()
            for e in np.flatnonzero(~np.isnan(out[0]))
        ]
        assert set(ends) == {weekday}
        assert np.nansum(out) == 7 * len(ends)
    offsets = set()
    for k in range(3):
        out = period_totals(days, times, "3day", end=k)
        ends = np.flatnonzero(~np.isnan(out[0]))
        assert set(np.diff(ends)) == {3}
        offsets.add(int(ends[0] % 3))
    assert offsets == {0, 1, 2}


def test_a_mixed_blocks_totals_in_the_context_are_the_releases_counts() -> None:
    # Late reports (20 days on) are past the release for recent onsets, so
    # the daily row's context is short of the truth there. The total row
    # must sum the same counts the daily row shows, not the truth.
    counts = _two_places()
    late = counts.assign(
        report_date=counts["report_date"] + pd.Timedelta(days=20)
    )
    counts = pd.concat([counts, late], ignore_index=True)
    cutoffs = ["2026-09-23", "2026-09-30"]
    places = (NATIONAL, "A", "B")
    block = training_block(counts, "daily+weekly", places, cutoffs, max_lag=3)
    n, start = len(places), block["start"]
    x = block["context"]
    daily, agg = x[:n], x[n : 2 * n]
    ends = [e for e in np.flatnonzero(~np.isnan(agg[0])) if e < start]
    assert ends
    for e in ends:
        assert np.allclose(agg[:, e], daily[:, e - 6 : e + 1].sum(1))
    truth = training_block(counts, "daily", places, ["2026-12-31"], max_lag=3)
    assert truth is not None


@pytest.mark.parametrize(
    ("source", "width"), [("daily+rolling7", 7), ("daily+rolling3", 3)]
)
def test_a_rolling_block_adds_each_places_rolling_sums_every_day(
    source, width
) -> None:
    counts = _two_places()
    cutoffs = ["2026-09-23", "2026-09-30"]
    places = (NATIONAL, "A", "B")
    block = training_block(counts, source, places, cutoffs, max_lag=3)
    daily = training_block(counts, "daily", places, cutoffs, max_lag=3)
    n = len(places)
    assert block["n_targets"] == 2 * n
    assert list(block["target_places"]) == [*range(n), *range(n)]
    x, d = block["context"], daily["context"]
    assert np.array_equal(x[:n], d[:n], equal_nan=True)
    assert np.array_equal(x[2 * n :], d[n:], equal_nan=True)
    agg = x[n : 2 * n]
    # Every day from the first whole window has its window's sum.
    assert np.isnan(agg[:, : width - 1]).all()
    assert np.isfinite(agg[:, width - 1 :]).all()
    for e in range(width - 1, agg.shape[1]):
        assert np.allclose(agg[:, e], d[:n, e - width + 1 : e + 1].sum(1))


def test_rolling_totals_sum_each_window_ending_on_each_day() -> None:
    from bdbvhubrenewal.forecasters.finetune import rolling_totals

    days = np.arange(10, dtype=float)[None, :]
    out = rolling_totals(days, 3)
    assert np.isnan(out[0, :2]).all()
    assert np.allclose(out[0, 2:], [3, 6, 9, 12, 15, 18, 21, 24])


def test_daily_releases_are_every_day_to_the_cut() -> None:
    from bdbvhubrenewal.forecasters.finetune import release_days

    rng = np.random.default_rng(0)
    out = release_days("2026-09-01", "2026-09-30", "daily", rng)
    assert out == [
        d.date().isoformat() for d in pd.date_range("2026-09-01", "2026-09-30")
    ]


@pytest.mark.parametrize(("cadence", "every"), [("3day", 3), ("weekly", 7)])
def test_periodic_releases_step_back_from_the_cut(cadence, every) -> None:
    from bdbvhubrenewal.forecasters.finetune import release_days

    rng = np.random.default_rng(0)
    out = release_days("2026-09-01", "2026-09-30", cadence, rng)
    assert out[-1] == "2026-09-30"
    gaps = np.diff(pd.to_datetime(out)).astype("timedelta64[D]").astype(int)
    assert set(gaps) == {every}
    assert pd.Timestamp(out[0]) - pd.Timestamp("2026-09-01") < pd.Timedelta(
        days=every
    )


@pytest.mark.parametrize("cadence", ["daily_gaps", "3day_gaps", "weekly_gaps"])
def test_releases_with_gaps_keep_the_cut_and_miss_some(cadence) -> None:
    from bdbvhubrenewal.forecasters.finetune import GAP_SHARE, release_days

    base = cadence.removesuffix("_gaps")
    rng = np.random.default_rng(1)
    full = release_days("2025-01-01", "2026-09-30", base, rng)
    kept, n = 0, 0
    for seed in range(20):
        out = release_days(
            "2025-01-01", "2026-09-30", cadence, np.random.default_rng(seed)
        )
        assert out[-1] == "2026-09-30"
        assert set(out) <= set(full)
        kept += len(out) - 1
        n += len(full) - 1
    assert 1 - kept / n == pytest.approx(GAP_SHARE, abs=0.03)


def test_weekday_releases_skip_weekends_but_keep_the_cut() -> None:
    from bdbvhubrenewal.forecasters.finetune import release_days

    rng = np.random.default_rng(0)
    out = release_days("2026-09-01", "2026-10-04", "weekdays", rng)
    # 2026-10-04 is a Sunday: the cut stays, the other weekends go.
    assert out[-1] == "2026-10-04"
    assert all(pd.Timestamp(d).weekday() < 5 for d in out[:-1])
    assert len(out) == 24 + 1


def test_an_unknown_cadence_is_refused() -> None:
    from bdbvhubrenewal.forecasters.finetune import release_days

    with pytest.raises(ValueError, match="cadence"):
        release_days("2026-09-01", "2026-09-30", "hourly", None)


def test_daily_releases_fill_every_lag_cell_weekly_ones_one_in_seven() -> None:
    from bdbvhubrenewal.forecasters.finetune import release_days

    counts = _two_places()
    places = (NATIONAL, "A", "B")
    rng = np.random.default_rng(0)
    fill = {}
    for cadence in ("daily", "weekly"):
        cuts = release_days("2026-09-01", "2026-09-30", cadence, rng)
        block = training_block(counts, "daily", places, cuts, max_lag=4)
        n, start = block["n_targets"], block["start"]
        # Lag d of day t needs a release on day t + d, all in the context.
        lags = block["context"][n:, 4:start]
        fill[cadence] = np.isfinite(lags).mean()
    assert fill["daily"] == 1
    assert fill["weekly"] == pytest.approx(1 / 7, abs=0.05)


def _releases_by_loop(counts: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """The release rows as summed one cut-off at a time (before r6)."""
    frames = []
    for cutoff in cutoffs:
        seen = counts.loc[counts["report_date"] <= pd.Timestamp(cutoff)]
        rows = seen.groupby(["location", "reference_date"], as_index=False)[
            "count"
        ].sum()
        rows.insert(0, "vintage", cutoff)
        rows.insert(1, "cutoff", cutoff)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("source", ["daily", "weekly"])
def test_releases_match_summing_each_cut_off_alone(source) -> None:
    from bdbvhubrenewal.forecasters.finetune import _release_rows

    rng = np.random.default_rng(3)
    n = 400
    counts = pd.DataFrame(
        {
            "location": rng.choice(["A", "B", "C"], n),
            "reference_date": pd.Timestamp("2026-09-01")
            + pd.to_timedelta(rng.integers(0, 30, n), "D"),
            "count": rng.integers(0, 5, n),
        }
    )
    counts["report_date"] = counts["reference_date"] + pd.to_timedelta(
        rng.integers(0, 20, n), "D"
    )
    days = pd.date_range("2026-08-30", "2026-10-25")
    cutoffs = [
        d.date().isoformat() for d in days[:: 1 if source == "daily" else 7]
    ]
    got = _release_rows(counts, cutoffs)
    want = _releases_by_loop(counts, cutoffs)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want, check_dtype=False
    )
