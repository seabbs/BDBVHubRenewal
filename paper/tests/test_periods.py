import json

import pandas as pd

from paperfigs.dummy import dummy_study
from paperfigs.periods import (
    PERIODS,
    STUDY_SPLITS,
    add_period,
    read_sidecar,
    study_split,
)


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


def test_germany_split_and_sidecar(tmp_path) -> None:
    assert STUDY_SPLITS["germany"] == pd.Timestamp("2022-02-10")
    assert study_split("germany") == pd.Timestamp("2022-02-10")
    assert study_split("ebola") is None
    side = tmp_path / "paper-forecasts.json"
    side.write_text(
        json.dumps({"study": "germany", "split_date": "2022-03-01"})
    )
    assert study_split("germany", read_sidecar(side)) == pd.Timestamp(
        "2022-03-01"
    )
    side.write_text(json.dumps({"study": "germany", "split_date": None}))
    assert study_split("germany", read_sidecar(side)) == pd.Timestamp(
        "2022-02-10"
    )


def test_dummy_germany_origins_straddle_the_split() -> None:
    fc, _ = dummy_study("germany", seed=1, n_samples=5)
    origins = pd.Series(sorted(fc["origin_date"].unique()))
    early = (origins < STUDY_SPLITS["germany"]).sum()
    assert 0 < early < len(origins)
