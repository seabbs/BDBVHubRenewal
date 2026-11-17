"""Assemble a dashboard from workflow config, the machine and the CLI.

`workflow/config.yaml`'s `dashboard` section holds what is the same on
every machine (references, release-prefix runs, the scoring rule, the
mailbox filter, the datasets); `config/local.env` holds this machine's
runs (`DASHBOARD_RUNS`), score source (`REMOTE`, `SCORE_CACHE`) and
mailbox; `--run` and `--ref` add or replace runs and references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from . import paperstyle
from . import runs as spec
from .mirror import Source
from .page import Context

ROOT = Path(__file__).parents[3]


def build(
    config: Mapping[str, Any],
    local: Mapping[str, str],
    runs: Sequence[str] = (),
    refs: Sequence[str] = (),
    root: Path = ROOT,
    once: bool = False,
) -> Context:
    """Return the dashboard's context.

    Raises
    ------
    SystemExit
        If no layer names a run.
    """
    ft = config["finetune"]
    extra = local_settings(local, root)
    dash = {
        **config["dashboard"],
        **{
            k: v
            for k, v in extra.items()
            if k not in ("runs", "exclude", "datasets")
        },
    }
    dash["datasets"] = merge_datasets(
        config["dashboard"].get("datasets") or [], extra.get("datasets") or []
    )
    run_list = spec.load_runs(
        [*(dash.get("runs") or []), *(extra.get("runs") or [])],
        local.get("DASHBOARD_RUNS", ""),
        runs,
    )
    if not run_list:
        raise SystemExit(
            "no runs to show: set dashboard.runs, DASHBOARD_RUNS in "
            "config/local.env, or --run name=<folder|prefix>"
        )
    # A reference may take its colour from the paper figures' model.
    references = [
        {
            **e,
            "colour": e.get("colour")
            or paperstyle.model_colour(e["paper_model"]),
        }
        if e.get("paper_model")
        else e
        for e in dash.get("references") or []
    ]
    source = Source(
        local,
        root / dash["mirror"],
        datasets=dash.get("datasets") or [],
        scores=dash.get("scores"),
        mailbox=dash.get("mailbox"),
        root=root,
        extras=[
            e["path"]
            for e in [*(dash.get("tables") or []), *(dash.get("notes") or [])]
        ],
    )
    excluded = exclusions(
        [config["dashboard"].get("exclude") or {}, extra.get("exclude") or {}],
        root,
    )
    return Context(
        runs=run_list,
        references=spec.load_references(references, refs),
        source=source,
        rule=dash["rule"],
        outbreaks=int(ft["outbreaks"]),
        learning_rate=float(ft["learning_rate"]),
        datasets=dash.get("datasets") or [],
        mailbox=source.mailbox,
        mail_filter=dash.get("mail_filter", ""),
        mail_lines=source.mailbox_lines,
        once=once,
        view={
            "studies": dash.get("studies") or {},
            "levels": dash.get("levels") or [],
            "defaults": dash.get("score_defaults") or {},
        },
        tables=[
            {**t, "path": str(source.extra(i))}
            for i, t in enumerate(dash.get("tables") or [])
        ],
        notes=[
            {**t, "path": str(source.extra(len(dash.get("tables") or []) + i))}
            for i, t in enumerate(dash.get("notes") or [])
        ],
        excluded=excluded,
    )


def merge_datasets(
    committed: Sequence[Mapping[str, Any]], local: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return the datasets, local entries updating committed ones by key."""
    out = {d["key"]: dict(d) for d in committed}
    for d in local:
        out.setdefault(d["key"], {}).update(d)
    return list(out.values())


def local_settings(local: Mapping[str, str], root: Path) -> dict[str, Any]:
    """Return the uncommitted dashboard settings, if any.

    `DASHBOARD_CONFIG` in config/local.env names the file; by default it
    is config/dashboard.local.yaml (git ignores it; see
    config/dashboard.example.yaml). Its keys replace those of the
    committed `dashboard` config, except `runs` (added after the
    committed ones) and `exclude` (joined with the committed one).
    """
    path = Path(
        local.get("DASHBOARD_CONFIG")
        or root / "config" / "dashboard.local.yaml"
    ).expanduser()
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def exclusions(parts: Sequence[Mapping[str, Any]], root: Path) -> set[str]:
    """Return the checkpoint hashes no panel shows.

    Each part lists `hashes`, and `dirs`, training output folders whose
    saved checkpoints are all excluded (relative to the repository).
    """
    out: set[str] = set()
    for part in parts:
        out |= {str(h) for h in part.get("hashes") or ()}
        for d in part.get("dirs") or ():
            out |= spec.folder_hashes(root / Path(d).expanduser())
    return out
