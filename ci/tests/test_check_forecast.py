from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import AS_OF, HUB, REF, make_samples, weeks, write_samples

from submission_ci.check_forecast import main, repair_national, run_checks
from submission_ci.common import read_json
from submission_ci.hubdata import read_observed, read_samples

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
        tmp_path / f"{REF}-epiforecasts-seabbsrenewal.csv",
        make_samples(**kwargs),
    )


def test_plausible_file_passes(tmp_path: Path) -> None:
    s = statuses(sub(tmp_path))
    assert s.pop("baseline") == "skip" and s.pop("previous") == "skip"
    # Zones are required only for seabbsbaseline.
    assert s.pop("zones") == "skip"
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


def test_missing_location_and_horizon_warn(tmp_path: Path) -> None:
    last = weeks()[-1]

    def mutate(rows: list[list[Any]]) -> None:
        rows[:] = [r for r in rows if r[2] != last]

    s = statuses(
        sub(tmp_path, levels={"CD-IT": 200.0, "CD-NK": 80.0}, mutate=mutate)
    )
    assert s["coverage"] == "warn"


def test_national_not_sum_fails(tmp_path: Path) -> None:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == "CD":
                r[6] += 1
                break

    assert statuses(sub(tmp_path, mutate=mutate))["national_sum"] == "fail"


def _checks(path: Path, **kwargs: Any) -> dict[str, Any]:
    checks = run_checks(
        path,
        reference_date=REF,
        observed=OBS,
        expected_locations=LOCS,
        tasks=TASKS,
        **kwargs,
    )
    return {c.name: c for c in checks}


def _sidecar(**kw: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "national": "CD",
        "modelled_provinces": ["CD-HU", "CD-IT", "CD-NK"],
        "unassigned": [],
        **kw,
    }


def _bump_national(week: str, sample: int, by: float) -> Any:
    def mutate(rows: list[list[Any]]) -> None:
        for r in rows:
            if r[3] == "CD" and r[2] == week and r[5] == sample:
                r[6] += by

    return mutate


def test_national_without_sidecar_says_so(tmp_path: Path) -> None:
    c = _checks(sub(tmp_path))["national_sum"]
    assert c.status == "pass"
    assert "no national.json" in c.summary


def test_national_exact_with_sidecar(tmp_path: Path) -> None:
    c = _checks(sub(tmp_path), national_sidecar=_sidecar())["national_sum"]
    assert c.status == "pass"
    assert "3 modelled provinces" in c.summary


def test_national_allows_only_float_noise(tmp_path: Path) -> None:
    w = weeks()[-1]
    noisy = sub(tmp_path, mutate=_bump_national(w, 1, 1e-12))
    assert statuses(noisy)["national_sum"] == "pass"
    off = sub(tmp_path, mutate=_bump_national(w, 1, 1e-6))
    assert statuses(off)["national_sum"] == "fail"


def test_national_off_by_one_fails(tmp_path: Path) -> None:
    path = sub(tmp_path, mutate=_bump_national(weeks()[0], 7, 1.0))
    c = _checks(path, national_sidecar=_sidecar())["national_sum"]
    assert c.status == "fail"
    assert "sample 7" in c.summary


def test_national_missing_modelled_province_warns(tmp_path: Path) -> None:
    side = _sidecar(modelled_provinces=["CD-HU", "CD-IT", "CD-NK", "CD-TO"])
    c = _checks(sub(tmp_path), national_sidecar=side)["national_sum"]
    assert c.status == "warn"
    assert "CD-TO" in c.summary


def test_national_unassigned_component_adds_up(tmp_path: Path) -> None:
    w = weeks()[-1]
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    assert statuses(path)["national_sum"] == "fail"
    side = _sidecar(
        unassigned=[{"target_end_date": w, "sample": 3, "value": 4.0}]
    )
    c = _checks(path, national_sidecar=side)["national_sum"]
    assert c.status == "pass"
    assert "unassigned" in c.summary
    short = _sidecar(
        unassigned=[{"target_end_date": w, "sample": 3, "value": 3.0}]
    )
    assert (
        _checks(path, national_sidecar=short)["national_sum"].status == "fail"
    )


def test_cli_reads_national_sidecar(tmp_path: Path) -> None:
    w = weeks()[-1]
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    side = tmp_path / "national.json"
    side.write_text(
        json.dumps(
            _sidecar(
                unassigned=[{"target_end_date": w, "sample": 3, "value": 4.0}]
            )
        )
    )
    args = [
        "--id",
        "main",
        "--submission",
        str(path),
        "--reference-date",
        REF,
        "--observed",
        str(HUB / "time-series.csv"),
        "--out",
        str(tmp_path / "o"),
        "--national-sidecar",
    ]
    main([*args, str(side), "--no-fail"])
    got = read_json(tmp_path / "o" / "checks.json")["checks"]
    assert {c["name"]: c["status"] for c in got}["national_sum"] == "pass"
    assert not read_json(tmp_path / "o" / "checks.json")["handled"]
    # Without the sidecar the national rows are rebuilt from the
    # provinces (no data gap given), so the check passes and says so.
    main([*args, str(tmp_path / "absent.json"), "--no-fail"])
    out = read_json(tmp_path / "o" / "checks.json")
    nat = {c["name"]: c for c in out["checks"]}["national_sum"]
    assert nat["status"] == "pass"
    assert "no national.json" in nat["summary"]
    assert out["handled"] == [
        "national (CD) rebuilt as the sum of the provinces plus nothing "
        "unassigned"
    ]


def _national(path: Path) -> dict[tuple[str, str], float]:
    s = read_samples(path)
    return {
        (sid, w): v for sid, t in s.values["CD"].items() for w, v in t.items()
    }


def test_repair_national_by_construction(tmp_path: Path) -> None:
    w = weeks()[-1]
    clean = sub(tmp_path)
    assert repair_national(clean, "CD") == ""
    before = _national(clean)
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    note = repair_national(path, "CD")
    assert note.startswith("national (CD) rebuilt")
    assert _national(path) == before
    assert statuses(path)["national_sum"] == "pass"
    # The model's unassigned component is kept.
    path = sub(tmp_path, mutate=_bump_national(w, 3, 4.0))
    side = _sidecar(
        unassigned=[{"target_end_date": w, "sample": 3, "value": 4.0}]
    )
    assert repair_national(path, "CD", side) == ""
    # Without national.json the data's gap is the unassigned part.
    path = sub(tmp_path)
    note = repair_national(path, "CD", None, {w: 2.0})
    assert "gap between the national data" in note
    after = _national(path)
    assert after[("1", w)] == before[("1", w)] + 2.0
    assert after[("1", weeks()[0])] == before[("1", weeks()[0])]


def test_repair_national_adds_a_missing_national(tmp_path: Path) -> None:
    path = sub(tmp_path, national=False)
    assert "added" in repair_national(path, "CD")
    assert statuses(path)["national_sum"] == "pass"


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


def test_fitted_national_is_reported_not_rebuilt(tmp_path: Path) -> None:
    # renewalsp fits the nation on its own, so national need not be the sum.
    w = weeks()[-1]
    path = sub(tmp_path, mutate=_bump_national(w, 3, -4.0))
    side = _sidecar(national_source="fitted")
    before = _national(path)
    assert repair_national(path, "CD", side) == ""
    assert _national(path) == before
    c = _checks(path, national_sidecar=side)["national_sum"]
    assert c.status == "pass"
    assert "own fit" in c.summary
    assert "report only" in c.summary


def with_zones(rows: list[list[Any]], skip: str = "", off: float = 0.0) -> None:
    """Split each province's samples evenly over two zones."""
    for r in list(rows):
        loc = r[3]
        if loc == "CD" or loc == skip:
            continue
        for k in (1, 2):
            rows.append([*r[:3], f"{loc}-0{k}", r[4], r[5], r[6] / 2 + off])


def test_zones_required_and_present(tmp_path: Path) -> None:
    f = sub(tmp_path, mutate=with_zones)
    checks = {c.name: c for c in run_checks(
        f, reference_date=REF, observed=read_observed(HUB / "time-series.csv"),
        require_zones=True,
    )}  # fmt: skip
    assert checks["zones"].status == "pass"
    # Zones do not count as provinces in the national sum.
    assert checks["national_sum"].status == "pass"


def test_missing_zones_fail_when_required(tmp_path: Path) -> None:
    obs = read_observed(HUB / "time-series.csv")
    none = sub(tmp_path)
    check = {c.name: c for c in run_checks(
        none, reference_date=REF, observed=obs, require_zones=True
    )}["zones"]  # fmt: skip
    assert check.status == "fail"
    part = sub(tmp_path / "p", mutate=lambda r: with_zones(r, skip="CD-NK"))
    check = {c.name: c for c in run_checks(
        part, reference_date=REF, observed=obs, require_zones=True
    )}["zones"]  # fmt: skip
    assert check.status == "fail"
    assert "CD-NK" in check.summary
    # Not required: no zones is fine.
    check = {c.name: c for c in run_checks(
        none, reference_date=REF, observed=obs
    )}["zones"]  # fmt: skip
    assert check.status == "skip"


def test_zones_that_do_not_add_up_warn(tmp_path: Path) -> None:
    f = sub(tmp_path, mutate=lambda r: with_zones(r, off=1.0))
    check = {c.name: c for c in run_checks(
        f, reference_date=REF, observed=read_observed(HUB / "time-series.csv"),
        require_zones=True,
    )}["zones"]  # fmt: skip
    assert check.status == "warn"
