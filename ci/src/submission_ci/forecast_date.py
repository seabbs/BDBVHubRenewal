"""The hub's `forecast_date` rules, for the gate and the submit step.

Every model-output file has a `forecast_date` column. For a new file all
rows share one date. A resubmission in the same week keeps the file now on
the hub's main branch unchanged and appends rows with exactly one new
date, so the file keeps its name and its history. The rules follow
`src/validate-submission.R` in InsightNet-US/BDBV-Modeling-Hub, which
decides when they differ from this module.

`fetch` reads the hub's current file (public, read only). `append` joins
it to a new forecast. The model's own validators (`hub_validate.py`,
hubValidations) then see the same appended file the PR will hold.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

FORECAST_DATE = "forecast_date"
RECENCY_DAYS = 1
HUB_REPO = "InsightNet-US/BDBV-Modeling-Hub"

Row = dict[str, str]


def parse_date(value: str) -> dt.date | None:
    """Parse a strict `YYYY-MM-DD` date, or None for `NA` and timestamps."""
    if len(value) != 10 or value.startswith("0000"):
        return None
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def read_rows(path: Path | str) -> tuple[list[str], list[Row]]:
    """Read a CSV as text."""
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def write_rows(path: Path | str, cols: list[str], rows: list[Row]) -> None:
    """Write rows with the given columns."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def problems(
    cols: list[str],
    rows: list[Row],
    previous: tuple[list[str], list[Row]] | None = None,
    today: dt.date | None = None,
) -> list[str]:
    """List why a file would fail the hub's `forecast_date` checks.

    Without `previous` every row is new. With a `today`, the new date must
    be within a day of it. The hub stops at its first failure, so this
    returns at most one message.
    """
    if cols.count(FORECAST_DATE) != 1:
        return [f"`{FORECAST_DATE}` must be a column, exactly once"]
    dates = [r.get(FORECAST_DATE, "") for r in rows]
    for i, value in enumerate(dates):
        if parse_date(value) is None:
            return [
                f"`{FORECAST_DATE}` must be a nonblank YYYY-MM-DD date; "
                f'CSV line {i + 2} has "{value}"'
            ]
    for i, r in enumerate(rows):
        ref = r.get("reference_date", "")
        if parse_date(ref) is None:
            return [f'`reference_date` "{ref}" is not a YYYY-MM-DD date']
        if dates[i] > ref:
            return [
                f"`{FORECAST_DATE}` must be on or before `reference_date`; "
                f"CSV line {i + 2} has {dates[i]} after {ref}"
            ]
    order = sorted(cols)
    if previous is None:
        new = dates
    else:
        pcols, prows = previous
        if FORECAST_DATE not in pcols:
            return [
                "the hub's copy has no `forecast_date` column, so it "
                "cannot be appended to"
            ]
        if sorted(pcols) != order:
            return ["the appended file must keep the existing columns"]
        old = {r[FORECAST_DATE] for r in prows}
        kept = Counter(
            tuple(r[c] for c in order) for r in rows if r[FORECAST_DATE] in old
        )
        if kept != Counter(tuple(r[c] for c in order) for r in prows):
            return [
                "every existing row must be kept unchanged, duplicates "
                "included, and new rows need a new `forecast_date`"
            ]
        new = [d for d in dates if d not in old]
    distinct = sorted(set(new))
    if len(distinct) != 1:
        return [
            "a submission must add exactly one new `forecast_date`, "
            f"found {distinct or 'none'}"
        ]
    if today is not None:
        gap = abs((dt.date.fromisoformat(distinct[0]) - today).days)
        if gap > RECENCY_DAYS:
            return [
                f"the new `forecast_date` {distinct[0]} must be within "
                f"{RECENCY_DAYS} day of today ({today.isoformat()})"
            ]
    return []


def append(
    previous: tuple[list[str], list[Row]],
    new: tuple[list[str], list[Row]],
) -> tuple[list[str], list[Row]]:
    """Return the hub's rows unchanged, then the new forecast's rows.

    Raises
    ------
    ValueError
        If the result would fail `problems`.
    """
    pcols, prows = previous
    ncols, nrows = new
    if sorted(pcols) != sorted(ncols):
        raise ValueError(
            "the new file's columns differ from the hub's copy: "
            f"{ncols} against {pcols}"
        )
    joined = [*prows, *nrows]
    found = problems(ncols, joined, previous)
    if found:
        raise ValueError(found[0])
    return pcols, joined


def previous_url(repo: str, model_id: str, reference_date: str) -> str:
    """Address of a model's file for a round on the hub's main branch."""
    return (
        f"https://raw.githubusercontent.com/{repo}/main/model-output/"
        f"{model_id}/{reference_date}-{model_id}.csv"
    )


def fetch(
    model_id: str,
    reference_date: str,
    out: Path,
    *,
    repo: str = HUB_REPO,
    checkout: Path | None = None,
    timeout: float = 30.0,
) -> bool:
    """Write the hub's file for the round to `out`, if it has one.

    The file is read from `checkout` when given (a local copy of the hub,
    used by the dry run) and from raw GitHub otherwise. Nothing is sent.
    """
    name = f"{reference_date}-{model_id}.csv"
    if checkout is not None:
        src = checkout / "model-output" / model_id / name
        if not src.exists():
            return False
        out.write_bytes(src.read_bytes())
        return True
    url = previous_url(repo, model_id, reference_date)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            out.write_bytes(resp.read())
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="Read the hub's file for the round.")
    f.add_argument("--model-id", required=True)
    f.add_argument("--reference-date", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--repo", default=HUB_REPO)
    f.add_argument("--checkout", type=Path, default=None)
    a = sub.add_parser("append", help="Append a new forecast to that file.")
    a.add_argument("--previous", type=Path, required=True)
    a.add_argument("--new", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check", help="Apply the hub's forecast_date rules.")
    c.add_argument("file", type=Path)
    c.add_argument("--previous", type=Path, default=None)
    c.add_argument("--today", default=None, help="YYYY-MM-DD, hub timezone.")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        found = fetch(
            args.model_id,
            args.reference_date,
            args.out,
            repo=args.repo,
            checkout=args.checkout,
        )
        print("found" if found else "absent")
        return 0
    if args.cmd == "append":
        try:
            cols, rows = append(read_rows(args.previous), read_rows(args.new))
        except ValueError as err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        write_rows(args.out, cols, rows)
        print(f"Appended {args.new} to {args.previous}: {len(rows)} rows.")
        return 0
    cols, rows = read_rows(args.file)
    prev = read_rows(args.previous) if args.previous else None
    today = dt.date.fromisoformat(args.today) if args.today else None
    found_problems = problems(cols, rows, prev, today)
    for msg in found_problems:
        print(f"ERROR: {msg}", file=sys.stderr)
    return 1 if found_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
