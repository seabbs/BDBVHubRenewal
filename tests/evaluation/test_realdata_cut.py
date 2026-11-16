"""The real-data model's BDBV training stops before any target it is
scored on."""

import datetime as dt
import tomllib
from pathlib import Path

from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.evaluation.backtest import load_config

ROOT = Path(__file__).parents[2]


def _cut() -> dict:
    text = (ROOT / "config" / "finetune.toml").read_text()
    return tomllib.loads(text)["realdata_bdbv"]


def test_the_cut_is_the_last_release_trained_on_with_its_truth_date():
    cut = _cut()
    assert cut["release"] == "2026-11-08"
    assert cut["cutoff"] == "2026-11-05"
    cutoff = dt.date.fromisoformat(cut["cutoff"])
    truth = cutoff - dt.timedelta(days=cut["truth_delay"])
    assert cut["truth_until"] == truth.isoformat()


def test_no_training_date_reaches_a_scored_bdbv_target():
    cut = _cut()
    config = load_config(ROOT / "config" / "backtest.toml")
    # A later origin's reference week is at least the cut-off's week, and
    # the backtest scores from `min_horizon` weeks before it.
    earliest = epiweek_end(cut["cutoff"]) + dt.timedelta(
        weeks=config.min_horizon
    )
    assert cut["truth_until"] < earliest.isoformat()
