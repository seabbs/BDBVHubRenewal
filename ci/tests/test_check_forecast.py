from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import AS_OF, HUB, REF, make_samples, weeks, write_samples

from submission_ci.check_forecast import main, run_checks
from submission_ci.common import read_json
from submission_ci.hubdata import read_observed

OBS = read_observed(HUB / "time-series.csv", AS_OF)
TASKS = read_json(HUB / "tasks.json")
LOCS = ["CD", "CD-IT", "CD-NK", "CD-HU"]


def statuses(path: Path, **kwargs: Any) -> dict[str, str]:
    checks = run_checks(
        path,
        reference_date=REF,
        observed=OBS,
        expected_locations=LOCS,
        tasks=TASKS,
        **kwargs,
    )
    return {c.name: c.status for c in checks}


def sub(tmp_path: Path, **kwargs: Any) -> Path:
    return write_samples(
        tmp_path / f"{REF}-epiforecasts-renewal.csv", make_samples(**kwargs)
    )


def test_plausible_file_passes(tmp_path: Path) -> None:
    s = statuses(sub(tmp_path))
    assert s.pop("baseline") == "skip" and s.pop("previous") == "skip"
    assert set(s.values()) == {"pass"}, s


def test_negative_and_nan_values_fail(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][6] = -3.0
        rows[1][6] = float("nan")

    assert statuses(sub(tmp_path, mutate=mutate))["values"] == "fail"


def test_unparsable_value_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][6] = "abc"

    assert statuses(sub(tmp_path, mutate=mutate))["values"] == "fail"


def test_wrong_sample_count_fails(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path, n=99))["samples"] == "fail"


def test_ragged_trajectory_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        del rows[3]

    s = statuses(sub(tmp_path, mutate=mutate))
    assert s["samples"] == "fail"


def test_missing_location_and_horizon_fail(tmp_path: Path) -> None:
    last = weeks()[-1]

    def mutate(rows: list[list[Any]]) -> None:
        rows[:] = [r for r in rows if r[2] != last]

    s = statuses(
        sub(tmp_path, levels={"CD-IT": 200.0, "CD-NK": 80.0}, mutate=mutate)
    )
    assert s["coverage"] == "fail"


def test_national_not_sum_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == "CD":
                r[6] += 1
                break

    assert statuses(sub(tmp_path, mutate=mutate))["national_sum"] == "fail"


def test_schema_failures(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        rows[0][4] = "quantile"
        rows[1][2] = "2026-11-06"  # a Friday

    assert statuses(sub(tmp_path, mutate=mutate))["schema"] == "fail"
    extra = write_samples(
        tmp_path / "x" / f"{REF}-m.csv",
        [[*r, "x"] for r in make_samples()],
        header=[
            "reference_date",
            "target",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
            "extra",
        ],
    )
    assert statuses(extra)["schema"] == "fail"


def test_location_not_in_tasks_fails(tmp_path: Path) -> None:
    s = statuses(sub(tmp_path, levels={"CD-XX": 10.0}))
    assert s["schema"] == "fail"


@pytest.mark.parametrize(
    ("growth", "status"), [(1.5, "pass"), (2.5, "warn"), (5.0, "fail")]
)
def test_growth_levels(tmp_path: Path, growth: float, status: str) -> None:
    assert statuses(sub(tmp_path, growth=growth))["growth"] == status


def test_bounds_fail_on_explosion(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path, growth=8.0))["bounds"] == "fail"


@pytest.mark.parametrize(
    ("scale", "status"), [(1.0, "pass"), (3.0, "warn"), (8.0, "fail")]
)
def test_observed_levels(tmp_path: Path, scale: float, status: str) -> None:
    levels = {
        k: v * 1.2 * scale
        for k, v in {"CD-IT": 200.0, "CD-NK": 80.0, "CD-HU": 15.0}.items()
    }
    assert statuses(sub(tmp_path, levels=levels))["observed"] == status


def test_baseline_and_previous_only_warn(tmp_path: Path) -> None:
    base = write_samples(tmp_path / "b" / "base.csv", make_samples(seed=2))
    far = write_samples(
        tmp_path / "p" / "prev.csv",
        make_samples(levels={"CD-IT": 2000.0, "CD-NK": 800.0, "CD-HU": 150.0}),
    )
    s = statuses(sub(tmp_path), baseline=base, previous=far)
    assert s["baseline"] == "pass" and s["previous"] == "warn"
    s = statuses(sub(tmp_path), baseline=far)
    assert s["baseline"] == "warn"


def test_persistence_skips_baseline(tmp_path: Path) -> None:
    assert statuses(sub(tmp_path), kind="persistence")["baseline"] == "skip"


def test_cli_writes_report(tmp_path: Path) -> None:
    path = sub(tmp_path, growth=5.0)
    args = [
        "--id",
        "main",
        "--submission",
        str(path),
        "--reference-date",
        REF,
        "--observed",
        str(HUB / "time-series.csv"),
        "--tasks",
        str(HUB / "tasks.json"),
        "--out",
        str(tmp_path / "o"),
        "--locations",
        ",".join(LOCS),
    ]
    assert main(args) == 1
    assert main([*args, "--no-fail"]) == 0
    data = read_json(tmp_path / "o" / "checks.json")
    assert data["status"] == "fail"
    assert (
        "Forecast checks for `main`: FAIL"
        in (tmp_path / "o" / "checks.md").read_text()
    )


def test_cli_missing_file_is_error(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--id",
                "m",
                "--submission",
                str(tmp_path / "none.csv"),
                "--reference-date",
                REF,
                "--observed",
                str(HUB / "time-series.csv"),
                "--out",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert read_json(tmp_path / "checks.json")["status"] == "error"
