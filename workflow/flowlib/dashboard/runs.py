"""The runs and references a dashboard shows, from configuration.

Nothing about which runs exist is in the code. Runs come in three layers,
each replacing a run of the same name from the one before: the
committed `dashboard.runs` of workflow/config.yaml, then `DASHBOARD_RUNS`
in config/local.env (specs separated by `;`), then `--run` on the
command line. A run spec is

    name=SOURCE[,label=L][,colour=C][,live][,base=OTHER][,prefix=P]
        [,hashes=H1+H2][,n=H1:250+H2:500]

A spec splits only at a comma that starts a known field, so a value may
hold commas not followed by a field name (a regex such as `a{1,2}`).
Config entries are read as they are, so their labels may hold anything.

SOURCE is a training output folder (its `train/n<N>/settings.json` name
the run's checkpoints) or a release prefix ending in `-`, such as
`realdata-ft-`. `hashes` and `n` name checkpoints a run has no folder
for. `base` names the run whose step-0 losses are the base model's.
One run is `live`, the one the status follows; by default the first.

References (zero-shot ragged, complete-data only, randomwalknowcast)
match score and forecast rows by a model regex and, optionally, a
checkpoint value, in the same layers: `dashboard.references`, then
`--ref` specs
`name=MODEL_REGEX[,checkpoint=C][,label=L][,colour=C][,role=R]`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

PALETTE = ("#2563eb", "#db2777", "#ea580c", "#0d9488", "#7c3aed", "#65a30d")
"""Colours of runs that name none, in order."""

RUN_FIELDS = {"label", "colour", "live", "base", "prefix", "hashes", "n"}
REF_FIELDS = {"label", "colour", "checkpoint", "role"}
ROLES = {"baseline", "start", "other"}
"""A reference's role on the score panels: the series scores are relative
to (not drawn), the one each run's curve starts from (dashed), or other
(dotted)."""


@dataclass(frozen=True)
class Run:
    """One training run the dashboard shows."""

    key: str
    label: str
    colour: str = ""
    folder: Path | None = None
    prefix: str | None = None
    hashes: tuple[str, ...] = ()
    n: Mapping[str, int] = field(default_factory=dict)
    base: str | None = None
    live: bool = False


@dataclass(frozen=True)
class Reference:
    """A reference series, matched on rows by model and checkpoint."""

    key: str
    label: str
    colour: str
    model: str
    checkpoint: str | None = None
    role: str = "other"

    def matches(self, row: Mapping[str, str]) -> bool:
        """Whether a score or forecast row is this reference's."""
        if not re.search(self.model, row.get("model", "")):
            return False
        return (
            self.checkpoint is None or row.get("checkpoint") == self.checkpoint
        )


def _split(spec: str, fields: set[str]) -> tuple[str, str, dict[str, str]]:
    """Split a spec at commas that start a known field.

    So a value may hold commas (a regex such as `a{1,2}`) as long as they
    are not followed by a field name; `,word=` with an unknown word is an
    error.
    """
    known = "|".join(sorted(fields))
    head, *rest = re.split(rf",(?=(?:{known})(?:=|,|$))", spec)
    if "=" not in head:
        raise ValueError(f"a spec is name=SOURCE[,key=value...], not {spec!r}")
    name, value = (x.strip() for x in head.split("=", 1))
    stray = re.search(r",([A-Za-z_]+)=", value)
    if stray:
        raise ValueError(f"unknown field {stray.group(1)!r} in {spec!r}")
    opts = {}
    for item in rest:
        k, sep, v = item.partition("=")
        opts[k.strip()] = v.strip() if sep else "1"
    return name, value, opts


def make_run(
    key: str,
    source: str,
    *,
    label: str | None = None,
    colour: str = "",
    live: bool = False,
    base: str | None = None,
    prefix: str | None = None,
    hashes: Sequence[str] = (),
    n: Mapping[str, int] | None = None,
) -> Run:
    """Return a run; SOURCE is a folder, or a release prefix ending in `-`."""
    is_prefix = source.endswith("-") and "/" not in source
    return Run(
        key=key,
        label=label or key,
        colour=colour or "",
        folder=None if is_prefix else Path(source).expanduser(),
        prefix=source if is_prefix else prefix,
        hashes=tuple(hashes),
        n=dict(n or {}),
        base=base,
        live=bool(live),
    )


def parse_run(spec: str) -> Run:
    """Return the run of a `name=SOURCE[,...]` spec."""
    name, source, o = _split(spec, RUN_FIELDS)
    sizes = {}
    for pair in filter(None, o.get("n", "").split("+")):
        h, _, size = pair.partition(":")
        sizes[h] = int(size)
    return make_run(
        name,
        source,
        label=o.get("label"),
        colour=o.get("colour", ""),
        live="live" in o,
        base=o.get("base"),
        prefix=o.get("prefix"),
        hashes=tuple(filter(None, o.get("hashes", "").split("+"))),
        n=sizes,
    )


def _from_config(entry: Mapping) -> Run:
    return make_run(
        entry["key"],
        entry["source"],
        label=entry.get("label"),
        colour=entry.get("colour", ""),
        live=bool(entry.get("live")),
        base=entry.get("base"),
        prefix=entry.get("prefix"),
        hashes=tuple(entry.get("hashes") or ()),
        n={h: int(v) for h, v in (entry.get("n") or {}).items()},
    )


def load_runs(
    config: Iterable[Mapping], machine: str, cli: Sequence[str]
) -> list[Run]:
    """Return the runs of the three layers, in order, one live."""
    out: dict[str, Run] = {}
    layers = [
        [_from_config(e) for e in config],
        [parse_run(s) for s in machine.split(";") if s.strip()],
        [parse_run(s) for s in cli],
    ]
    # A later layer replaces a run in place, keeping its order.
    for layer in layers:
        for r in layer:
            out[r.key] = r
    runs = list(out.values())
    live = [r.key for r in runs if r.live]
    first = live[-1] if live else (runs[0].key if runs else None)
    return [
        replace(
            r,
            live=r.key == first,
            colour=r.colour or PALETTE[i % len(PALETTE)],
        )
        for i, r in enumerate(runs)
    ]


def _role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {sorted(ROLES)}")
    return role


def make_reference(
    key: str,
    model: str,
    *,
    label: str | None = None,
    colour: str | None = None,
    checkpoint: str | None = None,
    role: str | None = None,
) -> Reference:
    """Return a reference matched on rows by `model` and `checkpoint`."""
    return Reference(
        key=key,
        label=label or key,
        colour=colour or "#6b7280",
        model=model,
        checkpoint=checkpoint,
        role=_role(role or "other"),
    )


def parse_reference(spec: str) -> Reference:
    """Return the reference of a `name=MODEL_REGEX[,...]` spec."""
    name, model, o = _split(spec, REF_FIELDS)
    return make_reference(
        name,
        model,
        label=o.get("label"),
        colour=o.get("colour"),
        checkpoint=o.get("checkpoint"),
        role=o.get("role"),
    )


def load_references(
    config: Iterable[Mapping], cli: Sequence[str]
) -> list[Reference]:
    """Return the references of config then `--ref`, later ones replacing."""
    out: dict[str, Reference] = {}
    for e in config:
        out[e["key"]] = make_reference(
            e["key"],
            e["model"],
            label=e.get("label"),
            colour=e.get("colour"),
            checkpoint=e.get("checkpoint"),
            role=e.get("role"),
        )
    for s in cli:
        r = parse_reference(s)
        out[r.key] = r
    return list(out.values())


HASH = re.compile(r"[0-9a-f]{6,}")


def split_label(name: str) -> tuple[str, str]:
    """Return a checkpoint name's hash and the prefix before it.

    `chronos-slots-abc123`, `slots-abc123` and `abc123` give
    (`abc123`, `chronos-slots-`), (`abc123`, `slots-`) and (`abc123`, ``).
    """
    m = HASH.search(name)
    if not m or m.end() != len(name):
        return name, ""
    return m.group(0), name[: m.start()]


def folder_hashes(folder: Path | None) -> set[str]:
    """Return the checkpoint hashes a training output folder has saved."""
    if folder is None:
        return set()
    out = set()
    for s in Path(folder).glob("train/n*/settings.json"):
        try:
            key = json.loads(s.read_text()).get("hash")
        except (OSError, ValueError):
            continue
        if key:
            out.add(str(key))
    return out


class Classifier:
    """Which run or reference a checkpoint or row belongs to."""

    def __init__(
        self, runs: Sequence[Run], references: Sequence[Reference] = ()
    ) -> None:
        self.runs = list(runs)
        self.references = list(references)
        self._saved = {r.key: folder_hashes(r.folder) for r in self.runs}

    def run_of(self, name: str) -> str | None:
        """Return the run of a checkpoint name, or None if none claims it."""
        key, prefix = split_label(name)
        bare = prefix.removeprefix("chronos-")
        for r in self.runs:
            if key in r.hashes or key in self._saved[r.key]:
                return r.key
        for r in self.runs:
            if r.prefix and r.prefix in (prefix, bare):
                return r.key
        return None

    def series(self, row: Mapping[str, str]) -> str | None:
        """Return a score or forecast row's series: a reference, else a run."""
        for ref in self.references:
            if ref.matches(row):
                return ref.key
        checkpoint = row.get("checkpoint", "")
        if not checkpoint or checkpoint == "zero-shot":
            return None
        return self.run_of(checkpoint)
