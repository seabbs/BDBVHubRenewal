"""Validate BDBV Modeling Hub submissions against the hub's own config.

The hub's rules are read at run time from a local clone of
InsightNet-US/BDBV-Modeling-Hub (``hub-config/tasks.json``,
``hub-config/admin.json`` and ``hub-config/model-metadata-schema.json``), so
the validator follows config changes such as new reference dates.

Checks mirror the names used by the R package hubValidations so results
can be compared one to one. Two checks are Python only and stricter than
hubValidations 3.0.0: ``spl_id_type`` (sample ids must be integers when the
config says so) and ``readme_horizon`` (a warning when a target_end_date is
not a Saturday within the README's window).

Dependencies: the standard library only for model output; PyYAML for
metadata files.

Usage::

    python hub_validate.py --hub PATH [--metadata-dir DIR]
        [--now ISO8601 | --skip-window] [--json OUT] FILE...

Exit status is 1 if any file has a failed check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

STD_COLS = ("output_type", "output_type_id", "value")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FILE_RE = re.compile(
    r"^(?P<round_id>\d{4}-\d{2}-\d{2}|[^-]+)"
    r"-(?P<team>[A-Za-z0-9_+]+)-(?P<model>[A-Za-z0-9_+]+)"
    r"\.(?P<ext>[A-Za-z0-9]+)$"
)
NA_STRINGS = {"", "NA", "NaN", "nan", "null", "NULL"}

Row = dict[str, str]


# Results -------------------------------------------------------------------


@dataclass
class Check:
    """Outcome of one check.

    ``status`` is one of ``pass``, ``fail`` (non-halting), ``error``
    (halting, as for hubValidations' ``check_error``), ``skip`` or ``warn``.
    """

    check: str
    status: str
    message: str


@dataclass
class Report:
    file: str
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        check: str,
        ok: bool,
        message: str,
        halt: bool = False,
        fail_message: str | None = None,
    ) -> bool:
        status = "pass" if ok else ("error" if halt else "fail")
        self.checks.append(
            Check(check, status, message if ok else fail_message or message)
        )
        return ok

    def skip(self, check: str, message: str) -> None:
        self.checks.append(Check(check, "skip", message))

    def warn(self, check: str, message: str) -> None:
        self.checks.append(Check(check, "warn", message))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status in ("fail", "error")]

    @property
    def valid(self) -> bool:
        return not self.failed


# Hub config ----------------------------------------------------------------


def norm(v: Any) -> str:
    """Canonical string for a config or file value.

    Numbers compare numerically (``"1"``, ``"1.0"`` and ``1`` match), so
    quantile levels such as ``0.025`` match however they are written.
    """
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(v).upper()
    s = str(v).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    if math.isfinite(f) and f == int(f):
        return str(int(f))
    return repr(f)


@dataclass
class TaskId:
    required: set[str] | None
    optional: set[str] | None

    @property
    def allowed(self) -> set[str]:
        return (self.required or set()) | (self.optional or set())


@dataclass
class OutputType:
    name: str
    is_required: bool
    required_ids: set[str] | None
    optional_ids: set[str] | None
    sample_params: dict[str, Any] | None
    value: dict[str, Any]


@dataclass
class ModelTask:
    task_ids: dict[str, TaskId]
    output_types: dict[str, OutputType]


@dataclass
class Round:
    index: int
    name: str
    round_id_from_variable: bool
    round_id: str
    model_tasks: list[ModelTask]
    submissions_due: dict[str, Any]

    @property
    def round_ids(self) -> set[str]:
        if not self.round_id_from_variable:
            return {self.round_id}
        out: set[str] = set()
        for mt in self.model_tasks:
            tid = mt.task_ids.get(self.round_id)
            if tid:
                out |= tid.allowed
        return out

    @property
    def task_id_names(self) -> list[str]:
        names: list[str] = []
        for mt in self.model_tasks:
            for n in mt.task_ids:
                if n not in names:
                    names.append(n)
        return names


def _vals(x: Any) -> set[str] | None:
    if x is None:
        return None
    return {norm(v) for v in x}


def _parse_output_type(name: str, cfg: dict[str, Any]) -> OutputType:
    oti = cfg.get("output_type_id", {})
    return OutputType(
        name=name,
        is_required=bool(cfg.get("is_required", False)),
        required_ids=_vals(oti.get("required")) if oti else None,
        optional_ids=_vals(oti.get("optional")) if oti else None,
        sample_params=cfg.get("output_type_id_params"),
        value=cfg.get("value", {}),
    )


@dataclass
class Hub:
    path: Path
    rounds: list[Round]
    admin: dict[str, Any]
    metadata_schema: dict[str, Any] | None

    @classmethod
    def load(cls, path: str | Path) -> Hub:
        p = Path(path)
        tasks = json.loads((p / "hub-config" / "tasks.json").read_text())
        admin = json.loads((p / "hub-config" / "admin.json").read_text())
        schema_p = p / "hub-config" / "model-metadata-schema.json"
        schema = json.loads(schema_p.read_text()) if schema_p.exists() else None
        rounds = []
        for i, r in enumerate(tasks["rounds"]):
            mts = []
            for mt in r["model_tasks"]:
                tids = {
                    k: TaskId(
                        _vals(v.get("required")), _vals(v.get("optional"))
                    )
                    for k, v in mt["task_ids"].items()
                }
                ots = {
                    k: _parse_output_type(k, v)
                    for k, v in mt["output_type"].items()
                }
                mts.append(ModelTask(tids, ots))
            rounds.append(
                Round(
                    index=i,
                    name=r.get("round_name", f"round {i + 1}"),
                    round_id_from_variable=bool(r["round_id_from_variable"]),
                    round_id=r["round_id"],
                    model_tasks=mts,
                    submissions_due=r["submissions_due"],
                )
            )
        return cls(p, rounds, admin, schema)

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.admin.get("timezone", "UTC"))

    @property
    def file_formats(self) -> list[str]:
        return list(self.admin.get("file_format", ["csv"]))

    def find_round(self, round_id: str) -> Round | None:
        for r in self.rounds:
            if round_id in r.round_ids:
                return r
        return None

    def all_round_ids(self) -> list[str]:
        out: list[str] = []
        for r in self.rounds:
            out += sorted(r.round_ids)
        return out

    def submission_window(
        self, rnd: Round, round_id: str
    ) -> tuple[datetime, datetime]:
        """Open and close times of the submission window, in hub time.

        Matches hubValidations: the window opens at 00:00:00 on the start
        day and closes at 23:59:59 on the end day.
        """
        due = rnd.submissions_due
        tz = self.timezone
        if "relative_to" in due:
            ref = date.fromisoformat(round_id)
            start_d = ref + timedelta(days=int(due["start"]))
            end_d = ref + timedelta(days=int(due["end"]))
        else:
            start_d = date.fromisoformat(str(due["start"]))
            end_d = date.fromisoformat(str(due["end"]))
        return (
            datetime.combine(start_d, time(0, 0, 0), tz),
            datetime.combine(end_d, time(23, 59, 59), tz),
        )


# Model output --------------------------------------------------------------


def parse_file_name(name: str) -> dict[str, str] | None:
    m = FILE_RE.match(name)
    return m.groupdict() if m else None


def find_metadata(model_id: str, dirs: Iterable[Path]) -> Path | None:
    for d in dirs:
        for ext in ("yml", "yaml"):
            p = d / f"{model_id}.{ext}"
            if p.exists():
                return p
    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _date_task_ids(rnd: Round) -> set[str]:
    """Task ids whose config values are all ISO dates."""
    out = set()
    for mt in rnd.model_tasks:
        for k, t in mt.task_ids.items():
            vals = t.allowed
            if vals and all(DATE_RE.match(v) for v in vals):
                out.add(k)
    return out


def _row_model_task(row: Row, rnd: Round) -> int | None:
    """Index of the first model task a row is valid for, else None."""
    for i, mt in enumerate(rnd.model_tasks):
        if all(norm(row.get(k)) in t.allowed for k, t in mt.task_ids.items()):
            ot = mt.output_types.get(row["output_type"])
            if ot is None:
                continue
            if ot.sample_params is not None:
                return i
            if ot.required_ids is None and ot.optional_ids is None:
                # e.g. mean/median: output_type_id must be NA.
                if row["output_type_id"] in NA_STRINGS:
                    return i
                continue
            ids = (ot.required_ids or set()) | (ot.optional_ids or set())
            if norm(row["output_type_id"]) in ids:
                return i
    return None


def _invalid_values(rows: list[Row], rnd: Round, cols: list[str]) -> str:
    """Describe invalid task id / output type values column by column."""
    msgs = []
    for c in cols:
        allowed: set[str] = set()
        for mt in rnd.model_tasks:
            if c in mt.task_ids:
                allowed |= mt.task_ids[c].allowed
            elif c == "output_type":
                allowed |= set(mt.output_types)
        if not allowed:
            continue
        bad = sorted({r[c] for r in rows if norm(r[c]) not in allowed})
        if bad:
            shown = ", ".join(f'"{b}"' for b in bad[:5])
            more = "" if len(bad) <= 5 else f" and {len(bad) - 5} more"
            msgs.append(
                f"Column `{c}` contains invalid value(s) {shown}{more}."
            )
    return " ".join(msgs) or "Invalid value combinations present."


def validate_submission(
    file: str | Path,
    hub: Hub,
    metadata_dirs: Iterable[Path] = (),
    now: datetime | None = None,
    skip_window: bool = False,
) -> Report:
    """Run all model-output checks on one file."""
    fp = Path(file)
    rep = Report(str(file))

    # File-level checks.
    meta = parse_file_name(fp.name)
    if not rep.add(
        "file_name",
        meta is not None,
        f'File name "{fp.name}" is valid.',
        halt=True,
        fail_message=f'File name "{fp.name}" must be '
        "<round_id>-<team>-<model>.<ext>.",
    ):
        return rep
    assert meta is not None
    model_id = f"{meta['team']}-{meta['model']}"
    round_id = meta["round_id"]
    rep.add(
        "file_location",
        fp.parent.name == model_id,
        "File directory name matches `model_id` in file name.",
        fail_message=f'File should be in directory "{model_id}", '
        f'not "{fp.parent.name}".',
    )
    rnd = hub.find_round(round_id)
    if not rep.add(
        "round_id_valid",
        rnd is not None,
        "`round_id` is valid.",
        halt=True,
        fail_message=f'`round_id` "{round_id}" is not in '
        "hub-config/tasks.json.",
    ):
        return rep
    assert rnd is not None
    if not rep.add(
        "file_format",
        meta["ext"] in hub.file_formats,
        "File is accepted hub format.",
        halt=True,
        fail_message=f'Extension "{meta["ext"]}" not in {hub.file_formats}.',
    ):
        return rep
    dirs = [*metadata_dirs, hub.path / "model-metadata"]
    rep.add(
        "metadata_exists",
        find_metadata(model_id, dirs) is not None,
        "Metadata file exists.",
        fail_message=f"No metadata file {model_id}.yml or .yaml.",
    )
    if skip_window:
        rep.skip("submission_time", "Submission window check skipped.")
    else:
        start, end = hub.submission_window(rnd, round_id)
        t = now or datetime.now(hub.timezone)
        if t.tzinfo is None:
            t = t.replace(tzinfo=hub.timezone)
        rep.add(
            "submission_time",
            start <= t <= end,
            "Submission time is within the submission window.",
            fail_message=f"Time {t.isoformat()} is outside window "
            f"{start.isoformat()} to {end.isoformat()}.",
        )

    # Read.
    try:
        with fp.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = list(reader.fieldnames or [])
            rows: list[Row] = [dict(r) for r in reader]
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        rep.add(
            "file_read",
            False,
            "",
            halt=True,
            fail_message=f"File could not be read: {e}",
        )
        return rep
    date_ids = _date_task_ids(rnd)
    bad_read = []
    for i, r in enumerate(rows, start=2):
        if None in r or any(v is None for v in r.values()):
            bad_read.append(f"line {i}: wrong number of fields")
            continue
        v = r.get("value", "")
        if v not in NA_STRINGS and not _is_number(v):
            bad_read.append(f'line {i}: `value` "{v}" is not numeric')
        for c in date_ids & set(cols):
            if r[c] not in NA_STRINGS and not DATE_RE.match(r[c]):
                bad_read.append(f'line {i}: `{c}` "{r[c]}" is not a date')
    if not rep.add(
        "file_read",
        not bad_read,
        "File could be read.",
        halt=True,
        fail_message="File could not be read with the hub schema: "
        + "; ".join(bad_read[:5]),
    ):
        return rep

    # Round id column.
    ric = rnd.round_id
    if rnd.round_id_from_variable:
        if not rep.add(
            "valid_round_id_col",
            ric in cols,
            "`round_id_col` name is valid.",
            halt=True,
            fail_message=f'Column "{ric}" missing.',
        ):
            return rep
        ids = sorted({r[ric] for r in rows})
        if not rep.add(
            "unique_round_id",
            len(ids) == 1,
            f'`round_id` column "{ric}" has a single value.',
            halt=True,
            fail_message=f'Column "{ric}" contains {len(ids)} '
            f"values: {', '.join(ids)}.",
        ):
            return rep
        if not rep.add(
            "match_round_id",
            ids[0] == round_id,
            "`round_id` column matches the file name.",
            halt=True,
            fail_message=f"`round_id` value {ids[0]} does not "
            f'match file name "{round_id}".',
        ):
            return rep

    # Columns.
    expected = [*rnd.task_id_names, *STD_COLS]
    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected]
    msg = []
    if missing:
        msg.append(f"Missing column(s): {', '.join(missing)}.")
    if extra:
        msg.append(f"Unexpected column(s): {', '.join(extra)}.")
    if not rep.add(
        "colnames",
        not msg,
        "Column names are valid.",
        halt=True,
        fail_message=" ".join(msg),
    ):
        return rep

    # Valid values and combinations.
    mt_idx = [_row_model_task(r, rnd) for r in rows]
    bad_rows = [r for r, m in zip(rows, mt_idx, strict=True) if m is None]
    if not rep.add(
        "valid_vals",
        not bad_rows,
        "`tbl` contains valid values/value combinations.",
        halt=True,
        fail_message=_invalid_values(
            bad_rows, rnd, [*rnd.task_id_names, "output_type"]
        ),
    ):
        return rep

    # Unique rows.
    key_cols = [*rnd.task_id_names, "output_type", "output_type_id"]
    keys = Counter(tuple(norm(r[c]) for c in key_cols) for r in rows)
    dups = sum(n - 1 for n in keys.values() if n > 1)
    rep.add(
        "rows_unique",
        dups == 0,
        "All task id/output type/output type id combinations are unique.",
        fail_message=f"{dups} duplicated row(s).",
    )

    _check_req_vals(rep, rows, rnd, mt_idx)
    _check_values(rep, rows, rnd, mt_idx)
    _check_non_desc(rep, rows, rnd)
    if any(r["output_type"] == "sample" for r in rows):
        _check_samples(rep, rows, rnd, mt_idx)
    else:
        for c in (
            "spl_mt_unique",
            "spl_compound_taskid_set",
            "spl_compound_tid",
            "spl_non_compound_tid",
            "spl_n",
        ):
            rep.skip(c, "No samples found.")
    _check_readme_horizon(rep, rows, round_id)
    return rep


def _check_req_vals(
    rep: Report, rows: list[Row], rnd: Round, mt_idx: list[int | None]
) -> None:
    """Check required task id values, output types and output type ids.

    A simplified version of hubValidations' ``req_vals``: for each model
    task the file uses, every required task id value must appear, every
    required output type must appear, and each task id combination with a
    given output type must carry all required output type ids.
    """
    problems = []
    used = {m for m in mt_idx if m is not None}
    for m in used:
        mt = rnd.model_tasks[m]
        mrows = [r for r, i in zip(rows, mt_idx, strict=True) if i == m]
        for k, t in mt.task_ids.items():
            for v in sorted(t.required or set()):
                if not any(norm(r[k]) == v for r in mrows):
                    problems.append(f'`{k}` "{v}" missing')
        for name, ot in mt.output_types.items():
            orows = [r for r in mrows if r["output_type"] == name]
            if ot.is_required and not orows:
                problems.append(f'output type "{name}" missing')
            if ot.required_ids and orows and ot.sample_params is None:
                combos: dict[tuple[str, ...], set[str]] = defaultdict(set)
                for r in orows:
                    combos[tuple(r[k] for k in mt.task_ids)].add(
                        norm(r["output_type_id"])
                    )
                for combo, got in combos.items():
                    lack = ot.required_ids - got
                    if lack:
                        problems.append(
                            f'{combo}: "{name}" ids {sorted(lack)} missing'
                        )
    rep.add(
        "req_vals",
        not problems,
        "Required task id/output type/output type id values present.",
        fail_message="; ".join(problems[:5]),
    )


def _check_values(
    rep: Report, rows: list[Row], rnd: Round, mt_idx: list[int | None]
) -> None:
    problems = []
    for i, (r, m) in enumerate(zip(rows, mt_idx, strict=True), start=2):
        if m is None:
            continue
        spec = rnd.model_tasks[m].output_types[r["output_type"]].value
        v = r["value"]
        if v in NA_STRINGS:
            problems.append(f"line {i}: value is missing")
            continue
        x = float(v)
        if not math.isfinite(x):
            problems.append(f"line {i}: value {v} is not finite")
        if spec.get("type") == "integer" and x != int(x):
            problems.append(f"line {i}: value {v} is not an integer")
        if "minimum" in spec and x < spec["minimum"]:
            problems.append(
                f"line {i}: value {v} is below minimum {spec['minimum']}"
            )
        if "maximum" in spec and x > spec["maximum"]:
            problems.append(
                f"line {i}: value {v} is above maximum {spec['maximum']}"
            )
    rep.add(
        "value_col_valid",
        not problems,
        "Values in column `value` are valid.",
        fail_message="; ".join(problems[:5]),
    )


def _check_non_desc(rep: Report, rows: list[Row], rnd: Round) -> None:
    qrows = [r for r in rows if r["output_type"] in ("quantile", "cdf")]
    if not qrows:
        rep.skip("value_col_non_desc", "No quantile or cdf output types.")
        return
    groups: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for r in qrows:
        k = (*(r[c] for c in rnd.task_id_names), r["output_type"])
        groups[k].append((float(r["output_type_id"]), float(r["value"])))
    bad = [
        k
        for k, g in groups.items()
        if any(
            b[1] < a[1] for a, b in zip(sorted(g), sorted(g)[1:], strict=False)
        )
    ]
    rep.add(
        "value_col_non_desc",
        not bad,
        "Quantile/cdf values are non-decreasing.",
        fail_message=f"Decreasing values for {bad[:3]}.",
    )


def _check_samples(
    rep: Report, rows: list[Row], rnd: Round, mt_idx: list[int | None]
) -> None:
    """Sample checks following hubValidations' semantics.

    A sample is all rows sharing an ``output_type_id``. hubValidations
    infers the compound task id set from the data: a configured compound
    task id that varies within a sample is dropped. So reusing ids
    ``1..100`` in every location makes each id a draw that is joint across
    locations (compound set coarser than configured), while unique ids per
    location keep the configured set.
    """
    srows = [
        (r, m)
        for r, m in zip(rows, mt_idx, strict=True)
        if r["output_type"] == "sample"
    ]
    by_id: dict[str, list[Row]] = defaultdict(list)
    mts_by_id: dict[str, set[int]] = defaultdict(set)
    for r, m in srows:
        by_id[r["output_type_id"]].append(r)
        if m is not None:
            mts_by_id[r["output_type_id"]].add(m)

    multi = sorted(k for k, v in mts_by_id.items() if len(v) > 1)
    if not rep.add(
        "spl_mt_unique",
        not multi,
        "Sample ids each belong to one modelling task.",
        halt=True,
        fail_message=f"Samples {multi[:5]} span modelling tasks.",
    ):
        return
    m = next(iter(mts_by_id[next(iter(by_id))]))
    mt = rnd.model_tasks[m]
    ot = mt.output_types["sample"]
    params = ot.sample_params or {}
    task_ids = list(mt.task_ids)
    cfg_set = list(params.get("compound_taskid_set") or task_ids)

    inferred = {
        sid: tuple(t for t in cfg_set if len({r[t] for r in g}) == 1)
        for sid, g in by_id.items()
    }
    sets = Counter(inferred.values())
    if not rep.add(
        "spl_compound_taskid_set",
        len(sets) == 1,
        "All samples share one compound task id set that matches "
        "or is coarser than the configured set.",
        halt=True,
        fail_message=f"{len(sets)} different compound task id "
        "sets across samples: "
        + "; ".join(f"{list(k)} ({n} samples)" for k, n in sets.most_common(3)),
    ):
        return
    comp = next(iter(sets))
    non_comp = [t for t in task_ids if t not in comp]

    multi_idx = sorted(
        sid
        for sid, g in by_id.items()
        if len({tuple(r[t] for t in comp) for r in g}) > 1
    )
    if not rep.add(
        "spl_compound_tid",
        not multi_idx,
        "Each sample has a single compound task id value.",
        halt=True,
        fail_message=f"Samples {multi_idx[:5]} span compound task id values.",
    ):
        return

    hashes = {
        sid: tuple(sorted(tuple(r[t] for t in non_comp) for r in g))
        for sid, g in by_id.items()
    }
    mode, _ = Counter(hashes.values()).most_common(1)[0]
    odd = sorted((s for s, h in hashes.items() if h != mode), key=_id_key)
    if not rep.add(
        "spl_non_compound_tid",
        not odd,
        "Non compound task id combinations are consistent across samples.",
        halt=True,
        fail_message=f"Samples {odd[:10]} do not match the most "
        f"frequent combination of {non_comp} in their task.",
    ):
        return

    per_idx: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for sid, g in by_id.items():
        per_idx[tuple(g[0][t] for t in comp)].add(sid)
    counts = {k: len(v) for k, v in per_idx.items()}
    lo = params.get("min_samples_per_task", 1)
    hi = params.get("max_samples_per_task", math.inf)
    distinct = sorted(set(counts.values()))
    out = {k: n for k, n in counts.items() if not lo <= n <= hi}
    if len(distinct) > 1:
        rep.add(
            "spl_n",
            False,
            "",
            fail_message=f"Samples per compound idx vary: {distinct}.",
        )
    else:
        rep.add(
            "spl_n",
            not out,
            "Required samples per compound idx present.",
            fail_message=f"{distinct[0]} samples per compound idx, "
            f"expected between {lo} and {hi}.",
        )

    if params.get("type") == "integer":
        bad = sorted({s for s in by_id if not re.fullmatch(r"-?\d+", s)})
        rep.add(
            "spl_id_type",
            not bad,
            "Sample ids are integers.",
            fail_message=f"Non-integer sample ids {bad[:5]} (config "
            "output_type_id_params.type is integer).",
        )


def _id_key(s: str) -> tuple[int, float, str]:
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)


def _check_readme_horizon(rep: Report, rows: list[Row], round_id: str) -> None:
    """Warn when a target_end_date is not a Saturday up to ref + 21 days."""
    if not rows or "target_end_date" not in rows[0]:
        return
    if not DATE_RE.match(round_id):
        return
    ref = date.fromisoformat(round_id)
    bad = sorted(
        {
            r["target_end_date"]
            for r in rows
            if DATE_RE.match(r["target_end_date"])
            and (
                date.fromisoformat(r["target_end_date"]).weekday() != 5
                or date.fromisoformat(r["target_end_date"])
                > ref + timedelta(days=21)
            )
        }
    )
    if bad:
        rep.warn(
            "readme_horizon",
            f"target_end_date(s) {bad[:5]} are not Saturdays up to "
            "three weeks after reference_date (README rule).",
        )
    else:
        rep.checks.append(
            Check(
                "readme_horizon",
                "pass",
                "target_end_dates follow the README rule.",
            )
        )


# Model metadata ------------------------------------------------------------


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _schema_errors(x: Any, s: Any, path: str = "") -> list[str]:
    """Validate ``x`` against the subset of JSON Schema the hub uses."""
    if s is True:
        return []
    if s is False:
        return [f"{path or '/'} is not allowed"]
    errs: list[str] = []
    types: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "object": dict,
        "array": list,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
    }
    t = s.get("type")
    if t:
        tt = t if isinstance(t, list) else [t]
        if not any(
            isinstance(x, types[k])
            and not (k in ("integer", "number") and isinstance(x, bool))
            for k in tt
        ):
            return [f"{path or '/'} must be of type {t}"]
    if "enum" in s and x not in s["enum"]:
        errs.append(f"{path or '/'} must be one of {s['enum']}")
    if isinstance(x, str):
        if "maxLength" in s and len(x) > s["maxLength"]:
            errs.append(
                f"{path} must not have more than {s['maxLength']} characters"
            )
        if "minLength" in s and len(x) < s["minLength"]:
            errs.append(
                f"{path} must have at least {s['minLength']} characters"
            )
        if "pattern" in s and not re.search(s["pattern"], x):
            errs.append(f'{path} must match pattern "{s["pattern"]}"')
        fmt = s.get("format")
        if fmt == "email" and not EMAIL_RE.match(x):
            errs.append(f'{path} must match format "email"')
        if fmt == "uri":
            u = urlparse(x)
            if not (u.scheme and (u.netloc or u.path)):
                errs.append(f'{path} must match format "uri"')
    if isinstance(x, dict):
        props = s.get("properties", {})
        for req in s.get("required", []):
            if req not in x:
                errs.append(
                    f"{path or '/'} must have required property '{req}'"
                )
        for k, v in x.items():
            if k in props:
                errs += _schema_errors(v, props[k], f"{path}/{k}")
            elif s.get("additionalProperties") is False:
                errs.append(
                    f"{path or '/'} must not have additional property '{k}'"
                )
    if isinstance(x, list) and "items" in s:
        for i, v in enumerate(x):
            errs += _schema_errors(v, s["items"], f"{path}/{i}")
    return errs


def validate_metadata(file: str | Path, hub: Hub) -> Report:
    """Run the model metadata checks on one YAML file."""
    fp = Path(file)
    rep = Report(str(file))
    if not rep.add(
        "metadata_file_ext",
        fp.suffix in (".yml", ".yaml"),
        'Metadata file extension is "yml" or "yaml".',
        halt=True,
    ):
        return rep
    try:
        import yaml
    except ImportError:
        rep.add(
            "metadata_file_read",
            False,
            "",
            halt=True,
            fail_message="PyYAML is needed to read metadata files.",
        )
        return rep
    try:
        data = yaml.safe_load(fp.read_text())
    except (OSError, yaml.YAMLError) as e:
        rep.add(
            "metadata_file_read",
            False,
            "",
            halt=True,
            fail_message=f"Could not read YAML: {e}",
        )
        return rep
    if hub.metadata_schema is None:
        rep.skip("metadata_matches_schema", "No metadata schema in hub.")
    else:
        errs = _schema_errors(data, hub.metadata_schema)
        if not rep.add(
            "metadata_matches_schema",
            not errs,
            "Metadata file contents are consistent with schema.",
            halt=True,
            fail_message="; ".join(errs),
        ):
            return rep
    mid = f"{data.get('team_abbr')}-{data.get('model_abbr')}"
    rep.add(
        "metadata_file_name",
        fp.stem == mid,
        "Metadata file name matches `model_id`.",
        halt=True,
        fail_message=f'File name "{fp.name}" but `model_id` is "{mid}".',
    )
    return rep


# CLI -----------------------------------------------------------------------


def validate_file(file: str | Path, hub: Hub, **kw: Any) -> Report:
    if Path(file).suffix in (".yml", ".yaml"):
        return validate_metadata(file, hub)
    return validate_submission(file, hub, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument(
        "--hub",
        required=True,
        type=Path,
        help="Local clone of BDBV-Modeling-Hub.",
    )
    ap.add_argument(
        "--metadata-dir",
        type=Path,
        action="append",
        default=[],
        help="Extra directory to search for <model_id>.yml.",
    )
    ap.add_argument("--now", help="ISO 8601 time for the window check.")
    ap.add_argument("--skip-window", action="store_true")
    ap.add_argument("--json", type=Path, help="Write results as JSON.")
    a = ap.parse_args(argv)
    hub = Hub.load(a.hub)
    now = datetime.fromisoformat(a.now) if a.now else None
    reports = [
        validate_file(
            f,
            hub,
            metadata_dirs=a.metadata_dir,
            now=now,
            skip_window=a.skip_window,
        )
        for f in a.files
    ]
    for r in reports:
        print(f"{'VALID' if r.valid else 'INVALID'}: {r.file}")
        for c in r.checks:
            if c.status in ("fail", "error", "warn"):
                print(f"  [{c.status}] {c.check}: {c.message}")
    if a.json:
        a.json.write_text(
            json.dumps(
                [
                    dict(file=r.file, **asdict(c))
                    for r in reports
                    for c in r.checks
                ],
                indent=2,
            )
        )
    return 0 if all(r.valid for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
