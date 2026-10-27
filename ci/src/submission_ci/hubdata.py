"""Read the hub's target data and hub-format sample files."""

from __future__ import annotations

import csv
import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

TARGET = "wk inc confirmed cases"
SUBMISSION_COLUMNS = frozenset(
    {
        "reference_date",
        "target",
        "target_end_date",
        "location",
        "output_type",
        "output_type_id",
        "value",
    }
)


def is_province(location: str) -> bool:
    """Whether a location code is a province (`CD-XX`)."""
    parts = location.split("-")
    return len(parts) == 2 and parts[0] == "CD" and len(parts[1]) == 2


@dataclass
class Samples:
    """Sample trajectories keyed by location, then sample, then week."""

    columns: list[str]
    rows: int
    reference_dates: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)
    output_types: set[str] = field(default_factory=set)
    # location -> sample id -> target_end_date -> value
    values: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    duplicates: int = 0
    unparsable: list[str] = field(default_factory=list)

    @property
    def locations(self) -> list[str]:
        """Locations in the file, sorted."""
        return sorted(self.values)

    def weeks(self, location: str) -> list[str]:
        """All target end dates for a location, sorted."""
        out: set[str] = set()
        for traj in self.values[location].values():
            out.update(traj)
        return sorted(out)

    def week_values(self, location: str, week: str) -> list[float]:
        """Sample values for one location and week."""
        return [
            traj[week]
            for traj in self.values[location].values()
            if week in traj
        ]


def read_samples(path: Path | str) -> Samples:
    """Read a hub-format sample CSV."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = Samples(columns=list(reader.fieldnames or []), rows=0)
        for line, row in enumerate(reader, start=2):
            out.rows += 1
            out.reference_dates.add(row.get("reference_date", ""))
            out.targets.add(row.get("target", ""))
            out.output_types.add(row.get("output_type", ""))
            try:
                value = float(row.get("value", ""))
            except ValueError:
                out.unparsable.append(f"line {line}: {row.get('value')!r}")
                value = math.nan
            traj = out.values[row.get("location", "")][
                row.get("output_type_id", "")
            ]
            week = row.get("target_end_date", "")
            if week in traj:
                out.duplicates += 1
            traj[week] = value
    return out


@dataclass(frozen=True)
class Observed:
    """One vintage of the hub's weekly target data."""

    as_of: str
    # location -> target_end_date -> count
    counts: dict[str, dict[str, float]]


def read_observed(path: Path | str, as_of: str | None = None) -> Observed:
    """Read one vintage of `target-data/time-series.csv` (latest default)."""
    by_vintage: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["target"] != TARGET:
                continue
            by_vintage[row["as_of"]][row["location"]][
                row["target_end_date"]
            ] = float(row["observation"])
    if not by_vintage:
        raise ValueError(f"{path} holds no `{TARGET}` rows")
    vintage = as_of or max(by_vintage)
    if vintage not in by_vintage:
        raise ValueError(f"vintage {vintage} is not in {path}")
    return Observed(
        vintage, {k: dict(v) for k, v in by_vintage[vintage].items()}
    )


def max_as_of(lines: Iterable[str]) -> str:
    """Latest `as_of` in the lines of a `time-series.csv`."""
    reader = csv.DictReader(lines)
    best = ""
    for row in reader:
        best = max(best, row.get("as_of", ""))
    return best


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile of a list (NaN when empty)."""
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    pos = q * (len(clean) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(clean) - 1)
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def is_saturday(value: str) -> bool:
    """Whether an ISO date string is a Saturday."""
    try:
        return dt.date.fromisoformat(value).weekday() == 5
    except ValueError:
        return False
