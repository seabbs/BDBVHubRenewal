"""Regenerate the fixture hub data in fixtures/hub/ (deterministic)."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
HUB = HERE / "hub"
LEVELS = {"CD-IT": 200, "CD-NK": 80, "CD-HU": 15}
NAMES = {
    "CD": "DRC",
    "CD-IT": "Ituri",
    "CD-NK": "Nord Kivu",
    "CD-HU": "Haut Uele",
}
TRUNCATION = [0.2, 0.5, 0.8]  # last three weeks, most recent first


def weeks(end: dt.date) -> list[dt.date]:
    start = dt.date(2026, 8, 1)
    out = []
    while start <= end:
        out.append(start)
        start += dt.timedelta(days=7)
    return out


def time_series() -> None:
    rows = []
    for as_of in ("2026-10-24", "2026-10-31"):
        a = dt.date.fromisoformat(as_of)
        ws = weeks(a)
        for loc, level in LEVELS.items():
            for i, w in enumerate(ws):
                back = len(ws) - 1 - i
                f = TRUNCATION[back] if back < 3 else 1.0
                rows.append((as_of, w, loc, round(level * (1 + 0.02 * i) * f)))
        for w in ws:
            total = sum(r[3] for r in rows if r[0] == as_of and r[1] == w)
            rows.append((as_of, w, "CD", total))
    with (HUB / "time-series.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "as_of",
                "target_end_date",
                "target",
                "location",
                "location_name",
                "location_type",
                "observation",
            ]
        )
        for as_of, w, loc, v in rows:
            kind = "country" if loc == "CD" else "province"
            wr.writerow(
                [
                    as_of,
                    w.isoformat(),
                    "wk inc confirmed cases",
                    loc,
                    NAMES[loc],
                    kind,
                    v,
                ]
            )


def tasks() -> None:
    ends = [
        (dt.date(2026, 2, 14) + dt.timedelta(days=7 * k)).isoformat()
        for k in range(42)
    ]
    data = {
        "schema_version": "https://raw.githubusercontent.com/hubverse-org/"
        "schemas/main/v6.0.0/tasks-schema.json",
        "rounds": [
            {
                "round_id_from_variable": True,
                "round_id": "reference_date",
                "model_tasks": [
                    {
                        "task_ids": {
                            "reference_date": {
                                "required": None,
                                "optional": ["2026-10-31", "2026-11-07"],
                            },
                            "location": {
                                "required": None,
                                # Two health zones per province.
                                "optional": sorted(
                                    [
                                        *NAMES,
                                        *(
                                            f"{p}-0{k}"
                                            for p in NAMES
                                            if p != "CD"
                                            for k in (1, 2)
                                        ),
                                    ]
                                ),
                            },
                            "target": {
                                "required": None,
                                "optional": ["wk inc confirmed cases"],
                            },
                            "target_end_date": {
                                "required": None,
                                "optional": ends,
                            },
                        },
                        "output_type": {
                            "sample": {
                                "output_type_id_params": {
                                    "type": "integer",
                                    "min_samples_per_task": 100,
                                    "max_samples_per_task": 100,
                                },
                                "value": {"type": "double", "minimum": 0},
                                "is_required": True,
                            }
                        },
                    }
                ],
                "submissions_due": {
                    "relative_to": "reference_date",
                    "start": -6,
                    "end": -4,
                },
            }
        ],
    }
    (HUB / "tasks.json").write_text(json.dumps(data, indent=2) + "\n")


def previous() -> None:
    out = HERE / "tmp-previous"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "fake_bdbvhubrenewal.py"),
            "nowcast",
            "--model",
            "renewal",
            "--vintage",
            "2026-10-24",
            "--out",
            str(out),
        ],
        check=True,
        env={**os.environ, "CAND_ID": "main"},
    )
    src = next(out.rglob("*.csv"))
    (HUB / "previous.csv").write_text(src.read_text())
    for p in sorted(out.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    out.rmdir()


if __name__ == "__main__":
    HUB.mkdir(exist_ok=True)
    time_series()
    tasks()
    previous()
    (HUB / "version.json").write_text(
        json.dumps(
            {
                "data_commit": "d" * 40,
                "committed_at": "2026-10-31T20:04:06Z",
                "hub_ref": "h" * 40,
                "as_of": "2026-10-31",
            },
            indent=2,
        )
        + "\n"
    )
    (HUB / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rounds": {
                    "2026-10-31": {
                        "data_commit": "c" * 40,
                        "as_of": "2026-10-24",
                        "status": "submitted",
                        "produced_by": "main model",
                        "pr_url": "",
                        "run": "",
                        "updated": "2026-10-26T09:00:00+00:00",
                        "history": [],
                    }
                },
            },
            indent=2,
        )
        + "\n"
    )
    (HUB / "seabbs-renewal.yml").write_text(
        "team_abbr: seabbs\nmodel_abbr: renewal\n"
    )
