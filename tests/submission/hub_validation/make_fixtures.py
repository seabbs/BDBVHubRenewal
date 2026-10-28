"""Generate the constructed submission and metadata fixtures.

Each invalid fixture differs from the valid base file by one change, so it
should trigger one specific check. Run from anywhere:

    python make_fixtures.py

The output is deterministic (fixed seed), so re-running gives identical
files. Expected outcomes live in ``fixtures/manifest.json``, which is
maintained by hand and checked by the tests.
"""

from __future__ import annotations

import csv
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fixtures" / "constructed"
MODEL_ID = "testteam-testmodel"
REF = "2026-10-31"
COLS = [
    "reference_date",
    "target",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
]
TARGET = "wk inc confirmed cases"
PROVINCES = ["CD-IT", "CD-NK"]

Row = dict[str, str]


def weeks(start: str, end: str) -> list[str]:
    """Saturdays from start to end inclusive, as ISO strings."""
    d, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= stop:
        out.append(d.isoformat())
        d += timedelta(days=7)
    return out


BASE_WEEKS = weeks("2026-10-03", "2026-11-21")


def trajectories(
    locs: dict[str, list[str]],
    n: int = 100,
    seed: int = 1,
    national: bool = True,
) -> list[Row]:
    """Simulate joint sample trajectories.

    ``locs`` maps a sub-national location to the weeks it covers. If
    ``national`` is true, ``CD`` is added as the sum of the provinces by
    sample index over the weeks they all share.
    """
    rng = random.Random(seed)
    rows: list[Row] = []
    vals: dict[tuple[str, int, str], int] = {}
    for loc, wks in locs.items():
        for i in range(1, n + 1):
            level = rng.randint(5, 200)
            for w in wks:
                level = max(0, level + rng.randint(-20, 20))
                vals[(loc, i, w)] = level
    if national:
        shared = sorted(set.intersection(*(set(w) for w in locs.values())))
        for i in range(1, n + 1):
            for w in shared:
                vals[("CD", i, w)] = sum(vals[(p, i, w)] for p in locs)
    for (loc, i, w), v in sorted(vals.items()):
        rows.append(
            {
                "reference_date": REF,
                "target": TARGET,
                "target_end_date": w,
                "location": loc,
                "output_type": "sample",
                "output_type_id": str(i),
                "value": str(v),
            }
        )
    return rows


def base_rows() -> list[Row]:
    return trajectories(dict.fromkeys(PROVINCES, BASE_WEEKS))


def write(
    case: str,
    rows: list[Row],
    cols: list[str] | None = None,
    model_dir: str = MODEL_ID,
    file_name: str | None = None,
) -> None:
    d = OUT / case / model_dir
    d.mkdir(parents=True, exist_ok=True)
    name = file_name or f"{REF}-{MODEL_ID}.csv"
    cols = cols or COLS
    with (d / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def drop(rows: list[Row], **match: str) -> list[Row]:
    return [r for r in rows if not all(r[k] == v for k, v in match.items())]


def with_week(rows: list[Row], week: str) -> list[Row]:
    """Append ``week`` to every trajectory (copying the last week's value)."""
    last = BASE_WEEKS[-1]
    extra = [
        dict(r, target_end_date=week)
        for r in rows
        if r["target_end_date"] == last
    ]
    return rows + extra


def submissions() -> None:
    b = base_rows()
    write("valid_basic", b)
    loc_off = {loc: k * 100 for k, loc in enumerate(["CD", *PROVINCES])}
    write(
        "valid_unique_ids",
        [
            dict(
                r,
                output_type_id=str(
                    int(r["output_type_id"]) + loc_off[r["location"]]
                ),
            )
            for r in b
        ],
    )
    write("valid_reordered_columns", b, cols=list(reversed(COLS)))
    write(
        "valid_noninteger_values",
        [dict(r, value=f"{int(r['value']) + 0.5}") for r in b],
    )
    write(
        "valid_health_zone",
        b
        + [
            dict(r, location="CD-IT-22", value=str(int(r["value"]) // 3))
            for r in b
            if r["location"] == "CD-IT"
        ],
    )
    write(
        "valid_forecast_weeks_only",
        [r for r in b if r["target_end_date"] >= REF],
    )
    write("valid_zero_values", [dict(r, value="0") for r in b])
    # Different week sets per location, sample ids reused across locations.
    write(
        "location_week_sets_differ",
        trajectories(
            {
                "CD-IT": weeks("2026-02-14", "2026-11-21"),
                "CD-NK": weeks("2026-10-24", "2026-11-21"),
            },
            national=False,
        ),
    )
    # Same, but sample ids unique across locations.
    diff = trajectories(
        {
            "CD-IT": weeks("2026-02-14", "2026-11-21"),
            "CD-NK": weeks("2026-10-24", "2026-11-21"),
        },
        national=False,
    )
    write(
        "location_week_sets_differ_unique_ids",
        [
            dict(
                r,
                output_type_id=str(
                    int(r["output_type_id"])
                    + (100 if r["location"] == "CD-NK" else 0)
                ),
            )
            for r in diff
        ],
    )

    write("invalid_samples_99", drop(b, location="CD-NK", output_type_id="100"))
    write(
        "invalid_samples_101",
        b
        + [
            dict(r, output_type_id="101")
            for r in b
            if r["location"] == "CD-NK" and r["output_type_id"] == "1"
        ],
    )
    write(
        "invalid_ragged_trajectory",
        drop(
            b,
            location="CD-IT",
            output_type_id="7",
            target_end_date="2026-10-10",
        ),
    )
    write(
        "invalid_ted_not_saturday",
        [
            dict(r, target_end_date="2026-10-30")
            if r["target_end_date"] == "2026-10-31"
            else r
            for r in b
        ],
    )
    write("invalid_ted_beyond_horizon", with_week(b, "2026-11-28"))
    write("invalid_ted_before_start", with_week(b, "2026-02-07"))
    write(
        "invalid_bad_location",
        [
            dict(r, location="CD-XX") if r["location"] == "CD-NK" else r
            for r in b
        ],
    )
    rows = [dict(r) for r in b]
    rows[5]["value"] = "-3"
    write("invalid_negative_value", rows)
    rows = [dict(r) for r in b]
    rows[5]["value"] = ""
    write("invalid_missing_value", rows)
    rows = [dict(r) for r in b]
    rows[5]["value"] = "many"
    write("invalid_non_numeric_value", rows)
    write(
        "invalid_wrong_output_type",
        [dict(r, output_type="quantile") for r in b],
    )
    write("invalid_bad_target", [dict(r, target="wk inc cases") for r in b])
    write(
        "invalid_extra_column",
        [dict(r, horizon="0") for r in b],
        cols=[*COLS, "horizon"],
    )
    write(
        "invalid_missing_column",
        b,
        cols=[c for c in COLS if c != "target_end_date"],
    )
    write(
        "invalid_ref_date_mismatch",
        [dict(r, reference_date="2026-10-24") for r in b],
    )
    unknown = [dict(r, reference_date="2026-11-07") for r in b]
    write(
        "invalid_unknown_round", unknown, file_name=f"2026-11-07-{MODEL_ID}.csv"
    )
    rows = [dict(r) for r in b]
    rows[0]["reference_date"] = "2026-10-24"
    write("invalid_mixed_ref_dates", rows)
    write("invalid_duplicate_row", [*b, dict(b[0])])
    write(
        "invalid_noninteger_sample_id",
        [
            dict(r, output_type_id="100.5")
            if r["output_type_id"] == "100"
            else r
            for r in b
        ],
    )
    write("invalid_bad_file_name", b, file_name=f"{REF}_{MODEL_ID}.csv")
    write("invalid_dir_mismatch", b, model_dir="otherteam-testmodel")
    write(
        "invalid_no_metadata",
        b,
        model_dir="nometa-testmodel",
        file_name=f"{REF}-nometa-testmodel.csv",
    )


VALID_META = """team_name: "Test Team"
team_abbr: "testteam"
model_name: "Test model"
model_abbr: "testmodel"
model_contributors: [
  {
    "name": "A Person",
    "affiliation": "Somewhere",
    "email": "a.person@example.com",
    "orcid": "0000-0001-8057-8037"
  }
]
license: "CC-BY-4.0"
model_details:
  data_inputs: "Hub target data."
  methods: "A short description."
  methods_long: "A longer description."
model_version: "0.1.0"
repo_url: "https://github.com/example/example"
ensemble_of_hub_models: false
"""


def metadata() -> None:
    d = OUT.parent / "constructed-metadata"
    cases: dict[str, tuple[str, str]] = {
        "valid": (MODEL_ID, VALID_META),
        # README says email is required; the schema does not require it.
        "valid_no_email": (
            MODEL_ID,
            VALID_META.replace('    "email": "a.person@example.com",\n', ""),
        ),
        "invalid_missing_methods": (
            MODEL_ID,
            VALID_META.replace('  methods: "A short description."\n', ""),
        ),
        "invalid_methods_too_long": (
            MODEL_ID,
            VALID_META.replace("A short description.", "x" * 201),
        ),
        "invalid_license_lowercase": (
            MODEL_ID,
            VALID_META.replace("CC-BY-4.0", "cc-by-4.0"),
        ),
        "invalid_model_abbr_too_long": (
            "testteam-abcdefghijklmnopq",
            VALID_META.replace(
                'model_abbr: "testmodel"', 'model_abbr: "abcdefghijklmnopq"'
            ),
        ),
        "invalid_bad_orcid": (
            MODEL_ID,
            VALID_META.replace("0000-0001-8057-8037", "0000-0001-8057"),
        ),
        "invalid_missing_contributors": (
            MODEL_ID,
            VALID_META.split("model_contributors")[0]
            + "license"
            + VALID_META.split("license", 1)[1],
        ),
        "invalid_name_mismatch": ("testteam-othermodel", VALID_META),
        "invalid_bad_email": (
            MODEL_ID,
            VALID_META.replace("a.person@example.com", "not-an-email"),
        ),
    }
    for case, (mid, text) in cases.items():
        p = d / case
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{mid}.yml").write_text(text)
    # Metadata used when validating constructed submissions.
    md = OUT / "model-metadata"
    md.mkdir(parents=True, exist_ok=True)
    (md / f"{MODEL_ID}.yml").write_text(VALID_META)
    (md / "otherteam-testmodel.yml").write_text(
        VALID_META.replace('"testteam"', '"otherteam"')
    )


def main() -> None:
    for p in (OUT, OUT.parent / "constructed-metadata"):
        if p.exists():
            shutil.rmtree(p)
    submissions()
    metadata()


if __name__ == "__main__":
    main()
