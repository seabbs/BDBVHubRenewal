"""Check that the hub's national data add up to its provinces.

`bdbv-hub-renewal coherence` writes `coherence.json` for the data vintage
(see INTEGRATION.md): for each stream (`observed`, `imputed`, `total`)
the weeks where the national count is not the sum of the provinces.
This is a check on the data, not on a candidate, so it runs once, in
`prepare`. A gap is handled in the model, through an unassigned national
component, and noted, so it never warns or blocks:

- `pass` when national equals the sum of provinces in every week, and
  also when it does not, with `gap` set and a note of how it is handled;
- `warn` only when the file cannot be read;
- `n/a` (`skip`) when there is no file.

It writes `coherence-summary.{md,json}`, which `candidate.sh` copies to
each candidate so the PR body can report it once.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    STATUS_LABEL,
    append_summary,
    fmt,
    markdown_table,
    read_json,
    set_outputs,
    write_json,
    write_text,
)

STREAMS = ("observed", "imputed", "total")
MAX_ROWS = 30


@dataclass
class Coherence:
    """The verdict on the data's national coherence."""

    status: str
    summary: str
    as_of: str = ""
    national: str = "CD"
    streams: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    handling: str = ""
    gap: bool = False

    def handled(self) -> str:
        """A `Handled:` note on the gap, or empty when there is none."""
        if not self.gap:
            return ""
        return f"{self.summary}; modelled as an unassigned national component"


def assess(data: Mapping[str, Any] | None, error: str = "") -> Coherence:
    """Judge one `coherence.json` payload."""
    if error:
        return Coherence("warn", f"coherence.json could not be read: {error}")
    if data is None:
        return Coherence(
            "skip", "no coherence.json, so the data were not checked"
        )
    national = str(data.get("national", "CD"))
    streams = {k: dict(v) for k, v in (data.get("streams") or {}).items() if v}
    rows = [dict(r) for r in data.get("rows") or []]
    out = Coherence(
        "pass",
        f"national ({national}) equals the sum of the provinces in every week",
        as_of=str(data.get("as_of", "")),
        national=national,
        streams=streams,
        rows=rows,
        handling=str(data.get("handling", "")),
    )
    gapped = {st: v for st, v in streams.items() if v.get("weeks_with_gap", 0)}
    if not (rows or gapped or data.get("coherent") is False):
        return out
    out.gap = True
    st = "total" if "total" in gapped else next(iter(gapped), "")
    if st:
        v = gapped[st]
        n = int(v["weeks_with_gap"])
        out.summary = (
            f"national ({national}) differs from the sum of the provinces "
            f"in {n} week{'s' if n != 1 else ''} ({st} stream, largest gap "
            f"{fmt(v.get('max_abs_gap'))})"
        )
    else:
        out.summary = (
            f"national ({national}) differs from the sum of the provinces"
        )
    return out


def gap_by_week(c: Coherence, stream: str = "total") -> dict[str, float]:
    """National minus the provinces, by week, for one stream."""
    return {
        str(r["target_end_date"]): float(r.get("gap") or 0.0)
        for r in c.rows
        if r.get("stream") == stream
    }


def details(c: Coherence) -> str:
    """The stream table, the weeks with a gap and the handling note."""
    lines: list[str] = []
    if c.streams:
        lines.append(
            markdown_table(
                ["stream", "weeks with a gap", "largest gap", "total gap"],
                [
                    [
                        st,
                        fmt(c.streams[st].get("weeks_with_gap")),
                        fmt(c.streams[st].get("max_abs_gap")),
                        fmt(c.streams[st].get("total_gap")),
                    ]
                    for st in STREAMS
                    if st in c.streams
                ],
            )
        )
        lines.append("")
    if c.rows:
        shown = sorted(
            c.rows, key=lambda r: (str(r["target_end_date"]), r["stream"])
        )
        lines.append("<details><summary>Weeks with a gap</summary>\n")
        lines.append(
            markdown_table(
                ["week ending", "stream", "national", "provinces", "gap"],
                [
                    [
                        str(r["target_end_date"]),
                        str(r["stream"]),
                        fmt(r.get("national")),
                        fmt(r.get("provinces")),
                        fmt(r.get("gap")),
                    ]
                    for r in shown[:MAX_ROWS]
                ],
            )
        )
        if len(shown) > MAX_ROWS:
            lines.append(f"\n... and {len(shown) - MAX_ROWS} more weeks.")
        lines.append("\n</details>\n")
    if c.handling:
        lines.append(f"How the model handles it: {c.handling}\n")
    return "\n".join(lines)


def markdown(c: Coherence) -> str:
    """The data coherence report as a markdown section."""
    head = f"## Data coherence: {STATUS_LABEL[c.status]}\n"
    vintage = f" (data vintage {c.as_of})" if c.as_of else ""
    return f"{head}\n{c.summary[0].upper()}{c.summary[1:]}{vintage}.\n\n" + (
        details(c)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point. Never fails the step."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coherence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    data, error = None, ""
    if args.coherence.exists():
        try:
            data = read_json(args.coherence)
        except ValueError as err:
            error = str(err)
    c = assess(data, error)
    md = markdown(c)
    write_text(args.out / "coherence-summary.md", md)
    write_json(args.out / "coherence-summary.json", asdict(c))
    append_summary(md)
    set_outputs({"coherence": c.status})
    print(f"data coherence: {c.status} ({c.summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
