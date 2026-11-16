"""Fixtures shared by the paper tests."""

import pandas as pd
import pytest

TARGET = "wk inc confirmed cases"


def _hub_file(path, ref, location, weeks, values, forecast_date):
    """Write one hub sample file with `values[week][sample]`."""
    rows = [
        {
            "reference_date": ref,
            "forecast_date": forecast_date,
            "target": TARGET,
            "target_end_date": week,
            "location": location,
            "output_type": "sample",
            "output_type_id": s + 1,
            "value": v,
        }
        for week, draws in zip(weeks, values, strict=True)
        for s, v in enumerate(draws)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def hub(tmp_path):
    """Build a small hub with one weekly round and an old cumulative file."""
    root = tmp_path / "hub"
    draws = [list(range(1, 101)), list(range(1, 101))]
    _hub_file(
        root / "model-output/team-a/2026-10-31-team-a.csv",
        "2026-10-31",
        "CD",
        ["2026-10-24", "2026-11-07"],
        draws,
        "2026-10-27",
    )
    old = root / "model-output/team-a/2026-06-13-team-a.csv"
    pd.DataFrame(
        {
            "reference_date": ["2026-06-13"],
            "target": ["cumulative cases"],
            "location": ["CD"],
            "output_type": ["quantile"],
            "output_type_id": [0.5],
            "value": [100.0],
        }
    ).to_csv(old, index=False)
    (root / "target-data").mkdir(parents=True)
    releases = [
        ("2026-10-24", "2026-10-17", 3),
        ("2026-11-01", "2026-10-17", 8),
        ("2026-11-01", "2026-10-24", 10),
        ("2026-11-08", "2026-10-17", 9),
        ("2026-11-08", "2026-10-24", 30),
        ("2026-11-08", "2026-11-07", 50),
    ]
    pd.DataFrame(
        {
            "as_of": [r[0] for r in releases],
            "target_end_date": [r[1] for r in releases],
            "target": TARGET,
            "location": "CD",
            "location_name": "DRC",
            "location_type": "country",
            "observation": [r[2] for r in releases],
        }
    ).to_csv(root / "target-data/time-series.csv", index=False)
    return root
