import numpy as np
import pandas as pd
import pytest
import yaml

from paperpipe.evidence import (
    EVIDENCE,
    aggregation_table,
    copula_table,
    rho_medians,
    write_evidence,
)


def _copula() -> pd.DataFrame:
    rows = []
    for arm, es in (("rolling", 0.3), ("unpaired", 0.5), ("fixed", 0.31)):
        for origin in ("2026-10-04", "2026-10-11"):
            rows.append(
                {
                    "study": "ebola",
                    "strata": "provinces",
                    "model": "randomwalknowcast",
                    "arm": arm,
                    "origin_date": origin,
                    "energy_log": es,
                    "variogram_log": 40.0,
                    "cover50_weeks": 0.9,
                    "cover90_weeks": 0.95,
                    "cover50_places": 0.6,
                    "cover90_places": 0.7,
                    "wis_strata": 1.0,
                    "wis_national": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_copula_table_averages_and_orders_the_arms() -> None:
    t = copula_table(_copula(), "ebola")
    assert t["Pairing"].tolist() == ["Unpaired", "Fixed", "Rolling"]
    assert t["ES (log)"].tolist() == [0.5, 0.31, 0.3]
    assert t["Places 50% / 90%"].iloc[0] == "0.60 / 0.70"


def test_rho_medians_by_strata_and_model() -> None:
    rho = pd.DataFrame(
        {
            "study": "germany",
            "strata": "states",
            "model": "randomwalknowcast",
            "rho_t": [0.5, 0.6, 0.7],
            "rho_p": [0.0, 0.04, 0.1],
            "rho_n": np.nan,
        }
    )
    out = rho_medians(rho)
    assert out["germany"]["states"]["randomwalknowcast"] == {
        "rho_t": 0.6,
        "rho_p": 0.04,
    }


def test_aggregation_is_relative_to_the_reference_on_shared_targets() -> None:
    base = {
        "study": "ebola",
        "level": "national",
        "location": "CD",
        "origin_date": "2026-10-04",
        "cover50": 1,
        "cover95": 1,
    }
    rows = [
        {
            **base,
            "model": "randomwalknowcast",
            "option": "reference",
            "target_date": t,
            "wis": 10.0,
            "log_wis": 1.0,
        }
        for t in ("2026-09-27", "2026-10-04")
    ] + [
        {
            **base,
            "model": "fm",
            "option": "joint",
            "target_date": "2026-09-27",
            "wis": 5.0,
            "log_wis": 0.5,
        },
        {
            **base,
            "model": "fm",
            "option": "level",
            "target_date": "2026-09-27",
            "wis": 20.0,
            "log_wis": 2.0,
        },
    ]
    t = aggregation_table(pd.DataFrame(rows), "ebola")
    assert t["Option"].tolist() == ["Joint block", "Level by level"]
    assert t["Relative WIS (log)"].tolist() == ["0.50 (0.50)", "2.00 (2.00)"]
    assert t["n"].tolist() == [1, 1]


def test_missing_evidence_leaves_a_note_and_no_numbers(tmp_path) -> None:
    out = write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    assert "waits on" in (tmp_path / "si" / "copula-ebola.md").read_text()
    assert (tmp_path / "fig" / "sim-realism.png").is_file()
    doc = yaml.safe_load((tmp_path / "e.yml").read_text())
    rho = doc["evidence"]["rho"]["germany"]["states"]["randomwalknowcast"]
    assert rho["rho_t"] == "(not yet available)"
    assert len(out["evidence"]["waiting"]) == len(EVIDENCE)


def test_a_paper_run_fails_while_evidence_is_missing(tmp_path) -> None:
    from paperpipe.__main__ import main

    with pytest.raises(ValueError, match="waits on"):
        main(
            [
                "evidence",
                "--mode",
                "paper",
                "--si-dir",
                str(tmp_path),
                "--fig-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "e.yml"),
            ]
        )
