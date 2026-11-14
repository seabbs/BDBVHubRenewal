import pandas as pd

from paperfigs.dummy import dummy_split, dummy_study
from paperfigs.periods import PERIODS, STUDY_SPLITS, add_period


def test_add_period_splits_on_origin_date() -> None:
    df = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(
                ["2022-01-01", "2022-02-01", "2022-03-01"]
            )
        }
    )
    out = add_period(df, pd.Timestamp("2022-02-01"))
    assert out["period"].tolist() == ["development", "held out", "held out"]
    assert tuple(PERIODS) == ("development", "held out")


def test_no_split_adds_no_period() -> None:
    df = pd.DataFrame({"origin_date": pd.to_datetime(["2022-01-01"])})
    assert "period" not in add_period(df, None).columns


def test_splits_are_per_study() -> None:
    assert set(STUDY_SPLITS) <= {"synthetic", "germany", "ebola"}


def test_dummy_split_is_the_midpoint_of_origins() -> None:
    fc, _ = dummy_study("germany", seed=1, n_samples=5)
    split = dummy_split(fc)
    origins = sorted(fc["origin_date"].unique())
    early = sum(o < split for o in origins)
    assert abs(early - (len(origins) - early)) <= 1
