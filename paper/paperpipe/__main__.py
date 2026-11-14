"""Command line for the workflow's steps: `python -m paperpipe <step>`.

Steps, in workflow order:

- `data`: a study's observations and per-location table;
- `forecasts`: every model's forecasts for a study;
- `table`: the forecasts checked against the schema and the held-out
  guard;
- `truth`: the rolling truth, D days after each target date;
- `scores`: the forecasts scored against the truth;
- `figures` and `score-tables`: main-text and SI outputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from paperfigs.schema import validate_forecasts, validate_observations
from paperfigs.scores import score_forecasts
from paperfigs.truth import attach_truth, rolling_truth
from paperpipe.config import check_no_heldout, run_settings
from paperpipe.outputs import score_tables, write_figures
from paperpipe.studies import load_data, run_forecasts

_DATES = ("origin_date", "target_date", "as_of_date")


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in _DATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "stratum" in df.columns:
        df["stratum"] = df["stratum"].astype(str)
    return df


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _date(value: str | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def _delays(locations: pd.DataFrame) -> pd.Series:
    return locations.set_index("location")["delay_days"].astype(int)


def _data(a: argparse.Namespace) -> None:
    obs, locations = load_data(
        a.study,
        seeds=a.seeds,
        before=_date(a.before),
        delay_days=a.delay,
        hub=a.hub,
    )
    _write(validate_observations(obs), a.observations)
    _write(locations, a.locations)


def _forecasts(a: argparse.Namespace) -> None:
    fc = run_forecasts(
        a.study,
        seeds=a.seeds,
        before=_date(a.before),
        n_samples=a.n_samples,
        hub=a.hub,
        candidates=a.candidates,
        store=a.store,
    )
    _write(fc, a.out)


def _table(a: argparse.Namespace) -> None:
    fc = validate_forecasts(_read(a.raw))
    check_no_heldout(fc, run_settings(json.loads(a.config), a.mode))
    _write(fc, a.out)


def _truth(a: argparse.Namespace) -> None:
    obs = validate_observations(_read(a.observations))
    _write(rolling_truth(obs, _delays(_read(a.locations))), a.out)


def _scores(a: argparse.Namespace) -> None:
    fc = validate_forecasts(_read(a.forecasts))
    _write(score_forecasts(attach_truth(fc, _read(a.truth))), a.out)


def _figures(a: argparse.Namespace) -> None:
    write_figures(
        a.study,
        validate_forecasts(_read(a.forecasts)),
        validate_observations(_read(a.observations)),
        _read(a.scores),
        _read(a.locations),
        a.main_dir,
        a.si_dir,
        split=_date(a.split),
    )


def _score_tables(a: argparse.Namespace) -> None:
    score_tables(
        a.study,
        _read(a.scores),
        _read(a.locations),
        a.main_dir,
        a.si_dir,
        split=_date(a.split),
    )


def _limits(p: argparse.ArgumentParser) -> None:
    p.add_argument("study")
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    p.add_argument("--before", default=None)
    p.add_argument("--hub", type=Path, default=None)
    p.add_argument("--candidates", type=Path, nargs="*", default=[])
    p.add_argument("--store", type=Path, default=None)


def _outputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("study")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--locations", type=Path, required=True)
    p.add_argument("--main-dir", type=Path, required=True)
    p.add_argument("--si-dir", type=Path, required=True)
    p.add_argument("--split", default=None)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of every step."""
    parser = argparse.ArgumentParser(prog="paperpipe", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)

    p = sub.add_parser("data")
    _limits(p)
    p.add_argument("--delay", type=int, default=None)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--locations", type=Path, required=True)
    p.set_defaults(run=_data)

    p = sub.add_parser("forecasts")
    _limits(p)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_forecasts)

    p = sub.add_parser("table")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--config", required=True, help="Workflow config as JSON.")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_table)

    p = sub.add_parser("truth")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--locations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_truth)

    p = sub.add_parser("scores")
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_scores)

    p = sub.add_parser("figures")
    _outputs(p)
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.set_defaults(run=_figures)

    p = sub.add_parser("score-tables")
    _outputs(p)
    p.set_defaults(run=_score_tables)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one step."""
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
