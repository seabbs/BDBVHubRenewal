"""Check the scores against R scoringutils on a fixed fixture.

`tests/scoringutils_reference.R` writes the expected scores from
`tests/fixtures/scoring-fixture.csv` with scoringutils' defaults.
"""

from pathlib import Path

import pandas as pd
import pytest

from paperfigs.schema import FORECAST_KEYS, validate_forecasts
from paperfigs.scores import score_forecasts

FIXTURES = Path(__file__).parent / "fixtures"
COLUMNS = ("score", "log_score", "bias", "coverage_50", "coverage_90")


@pytest.fixture(scope="module")
def compared() -> pd.DataFrame:
    fc = validate_forecasts(
        pd.read_csv(
            FIXTURES / "scoring-fixture.csv", dtype={"sample_id": "Int64"}
        )
    )
    want = pd.read_csv(
        FIXTURES / "scoringutils-expected.csv",
        parse_dates=["origin_date", "target_date"],
    )
    got = score_forecasts(fc)
    keys = [*FORECAST_KEYS, "metric"]
    out = got.merge(want, on=keys, suffixes=("", "_su"), validate="1:1")
    assert len(out) == len(want) == len(got)
    return out


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("metric", ["wis", "crps"])
def test_matches_scoringutils(compared, column, metric) -> None:
    rows = compared.loc[compared["metric"] == metric]
    assert len(rows) == 12
    pd.testing.assert_series_equal(
        rows[column].reset_index(drop=True),
        rows[f"{column}_su"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-12,
    )
