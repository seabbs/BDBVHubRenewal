from pathlib import Path

import pandas as pd
import pytest
import yaml

from paperpipe.config import (
    MODES,
    check_no_heldout,
    run_settings,
    seed_of,
    synthetic_location,
)

CONFIG = Path(__file__).parents[1] / "workflow" / "config.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_modes() -> None:
    assert MODES == ("development", "paper")


def test_development_leaves_out_the_heldout_sets() -> None:
    cfg = _config()
    dev = run_settings(cfg, "development")
    held = set(cfg["heldout"]["synthetic_seeds"])
    assert dev.synthetic_seeds
    assert not held & set(dev.synthetic_seeds)
    assert dev.germany_before == pd.Timestamp(cfg["heldout"]["germany_from"])
    assert dev.germany_split is None


def test_paper_includes_the_heldout_sets() -> None:
    cfg = _config()
    paper = run_settings(cfg, "paper")
    assert paper.synthetic_seeds == tuple(cfg["heldout"]["synthetic_seeds"])
    assert paper.germany_before is None
    assert paper.germany_split == pd.Timestamp(cfg["heldout"]["germany_from"])


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        run_settings(_config(), "tuning")


def test_overlapping_seeds_raise() -> None:
    cfg = _config()
    cfg["development"]["synthetic_seeds"] = [9001]
    with pytest.raises(ValueError, match="9001"):
        run_settings(cfg, "development")


def test_missing_heldout_entry_raises() -> None:
    cfg = _config()
    del cfg["heldout"]["germany_from"]
    with pytest.raises(ValueError, match="germany_from"):
        run_settings(cfg, "development")


def test_synthetic_location_carries_its_seed() -> None:
    loc = synthetic_location(9001, "sim03")
    assert seed_of(loc) == 9001
    assert seed_of("DE-BY") is None


def _table(study: str, location: str, origin: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study": [study],
            "location": [location],
            "origin_date": [pd.Timestamp(origin)],
        }
    )


def test_guard_passes_development_rows() -> None:
    dev = run_settings(_config(), "development")
    ok = pd.concat(
        [
            _table("germany", "DE", "2022-02-09"),
            _table("synthetic", synthetic_location(7001, "sim01"), "2024-03-01"),
        ]
    )
    check_no_heldout(ok, dev)


def test_guard_rejects_heldout_german_dates() -> None:
    dev = run_settings(_config(), "development")
    with pytest.raises(ValueError, match="held-out"):
        check_no_heldout(_table("germany", "DE", "2022-02-10"), dev)


def test_guard_rejects_heldout_seeds() -> None:
    dev = run_settings(_config(), "development")
    bad = _table("synthetic", synthetic_location(9002, "sim01"), "2024-03-01")
    with pytest.raises(ValueError, match="held-out"):
        check_no_heldout(bad, dev)


def test_guard_allows_everything_in_paper_mode() -> None:
    paper = run_settings(_config(), "paper")
    check_no_heldout(_table("germany", "DE", "2022-04-01"), paper)
