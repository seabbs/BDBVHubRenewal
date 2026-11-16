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
    dash = config["dashboard"]
    ft = config["finetune"]
    run_list = spec.load_runs(
        dash.get("runs") or [], local.get("DASHBOARD_RUNS", ""), runs
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
    source = Source(local, root / dash["mirror"])
    return Context(
        runs=run_list,
        references=spec.load_references(references, refs),
        source=source,
        rule=dash["rule"],
        outbreaks=int(ft["outbreaks"]),
        learning_rate=float(ft["learning_rate"]),
        lr_steps=int(ft["lr_steps"]),
        style=paperstyle.page_style(),
        datasets=dash.get("datasets") or [],
        mailbox=source.mailbox,
        mail_filter=dash.get("mail_filter", ""),
        once=once,
    )
