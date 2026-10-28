import datetime as dt
import os
from pathlib import Path

import numpyro
import pandas as pd
import pytest

numpyro.set_host_device_count(2)

from bdbvhubrenewal.data import (  # noqa: E402
    PROVINCE_CENTRES,
    PROVINCE_POPULATIONS,
)
from bdbvhubrenewal.data.simulate import simulate_outbreak  # noqa: E402
from bdbvhubrenewal.model import FALLBACK, MAIN  # noqa: E402
from bdbvhubrenewal.model.inputs import vintage_from_tables  # noqa: E402

# Vintage dates as days after the simulation start (each a Saturday).
VINTAGE_DAYS = (166, 206)
MAIN_REPO_DATA = Path(
    os.environ.get(
        "BDBV_DATA",
        str(Path(__file__).resolve().parents[2] / "data"),
    )
)
# Two provinces with no cases, to exercise the importation component.
SIM_LOCATIONS = ("CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU", "CD-SK")


@pytest.fixture(scope="session")
def table():
    if (MAIN_REPO_DATA / "locations.csv").exists():
        return pd.read_csv(MAIN_REPO_DATA / "locations.csv")
    return pd.DataFrame(
        {
            "location": list(PROVINCE_CENTRES),
            "population": [PROVINCE_POPULATIONS[p] for p in PROVINCE_CENTRES],
            "latitude": [c[0] for c in PROVINCE_CENTRES.values()],
            "longitude": [c[1] for c in PROVINCE_CENTRES.values()],
        }
    )


@pytest.fixture(scope="session")
def outbreak():
    return simulate_outbreak()


def vintage_date(outbreak, day):
    return (outbreak.settings.start + dt.timedelta(days=day)).isoformat()


@pytest.fixture(scope="session")
def sim_tables(outbreak):
    return {
        d: outbreak.vintage(vintage_date(outbreak, d)) for d in VINTAGE_DAYS
    }


@pytest.fixture(scope="session")
def sim_vintage(outbreak, sim_tables, table):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    return vintage_from_tables(
        target, delays, vintage_date(outbreak, day), SIM_LOCATIONS, table
    )


@pytest.fixture(params=["main", "fallback"])
def config(request):
    return {"main": MAIN, "fallback": FALLBACK}[request.param]
