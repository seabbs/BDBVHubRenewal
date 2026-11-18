"""Command line for the workflow's steps: `python -m paperpipe <step>`.

Steps, in workflow order:

- `data`: a study's observations, per-location table and run notes;
- `forecasts`: every model's forecasts for a study;
- `table`: the forecasts checked against the schema and the held-out
  guard;
- `truth`: the rolling truth, D days after each target date;
- `scores`: the forecasts scored against the truth;
- `joint-scores`: sample trajectories scored jointly (energy score);
- `figures` and `score-tables`: main-text and SI outputs;
- `evidence`: SI tables, figures and numbers from evidence branches;
- `ebola-extra`: further Ebola samples, kept to some models;
- `germany-extra`: further German forecasts as a paper table;
- `summary`: the numbers the paper quotes, as Quarto metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from bdbvhubrenewal.reporting.live_checkpoint import LIVE_RUN

from paperdata import germany
from paperfigs.joint import joint_scores
from paperfigs.schema import validate_forecasts, validate_observations
from paperfigs.scores import score_forecasts
from paperfigs.truth import attach_truth, rolling_truth
from paperpipe.config import (
    check_no_heldout,
    check_no_heldout_releases,
    keep_finetuned,
    run_settings,
)
from paperpipe.evidence import EVIDENCE, write_evidence
from paperpipe.meta import baseline_of, read_meta, write_meta
from paperpipe.outputs import (
    joint_table,
    real_finetune_combined,
    score_tables,
    to_markdown,
    write_figures,
)
from paperpipe.studies import (
    NO_SOURCES,
    Sources,
    load_data,
    missing,
    run_forecasts,
    study_baseline,
)
from paperpipe.summary import study_summary, write_summary

log = logging.getLogger(__name__)

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


def _sources(a: argparse.Namespace) -> Sources:
    return Sources(
        synthetic_forecasts=a.synthetic_forecasts,
        germany_forecasts=tuple(a.germany_forecasts),
        germany_triangle=a.germany_triangle,
        hub=a.hub,
        ebola_data=a.ebola_data,
        ebola_config=a.ebola_config,
        ebola_backtest=a.ebola_backtest,
        ebola_extra=tuple(a.ebola_extra),
        ebola_draws=a.ebola_draws,
        ebola_draws_models=tuple(a.ebola_draws_models),
        candidates=tuple(a.candidates),
        store=a.store,
        store_ref=a.store_ref,
    )


def _data(a: argparse.Namespace) -> None:
    sources = _sources(a)
    reason = missing(a.study, sources, a.delay)
    if reason and a.mode == "paper":
        raise ValueError(
            f"The paper run cannot use dummy data for the {a.study} study: "
            f"{reason}."
        )
    if reason:
        log.warning("The %s study uses dummy data: %s.", a.study, reason)
        sources = NO_SOURCES
    obs, locations = load_data(
        a.study,
        seeds=a.seeds,
        before=_date(a.before),
        delay_days=a.delay,
        sources=sources,
    )
    if a.study == "germany":
        settings = run_settings(json.loads(a.config), a.mode)
        check_no_heldout_releases(obs, settings, a.delay)
    _write(validate_observations(obs), a.observations)
    _write(locations, a.locations)
    write_meta(
        a.meta,
        a.study,
        "dummy" if reason else "real",
        baseline=None if reason else study_baseline(a.study, sources, a.delay),
        note="" if reason else a.note,
    )


def _forecasts(a: argparse.Namespace) -> None:
    fc = run_forecasts(
        a.study,
        seeds=a.seeds,
        before=_date(a.before),
        n_samples=a.n_samples,
        sources=_sources(a),
        source=read_meta(a.meta)["source"],
        baseline=read_meta(a.meta).get("baseline"),
        split=_date(a.split),
    )
    _write(fc, a.out)


def _table(a: argparse.Namespace) -> None:
    config = json.loads(a.config)
    fc = keep_finetuned(validate_forecasts(_read(a.raw)), config)
    check_no_heldout(fc, run_settings(config, a.mode))
    _write(fc, a.out)


def _truth(a: argparse.Namespace) -> None:
    obs = validate_observations(_read(a.observations))
    truth = rolling_truth(
        obs, _delays(_read(a.locations)), first_after=a.first_after
    )
    _write(truth, a.out)


def _scores(a: argparse.Namespace) -> None:
    fc = attach_truth(validate_forecasts(_read(a.forecasts)), _read(a.truth))
    _write(score_forecasts(fc), a.out)
    # Sample forecasts by the fair CRPS too, for the SI.
    _write(score_forecasts(fc, samples="crps"), a.crps_out)


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
        crps_scores=_read(a.crps_scores),
        dummy=read_meta(a.meta)["source"] == "dummy",
        truth=_read(a.truth),
        nowcasts=json.loads(a.config)["nowcasts"],
        baseline=baseline_of(read_meta(a.meta)),
    )


def _joint(a: argparse.Namespace) -> None:
    fc = validate_forecasts(_read(a.forecasts))
    scores = joint_scores(attach_truth(fc, _read(a.truth)))
    _write(scores, a.out)
    joint_table(scores, a.table, baseline_of(read_meta(a.meta)))


def _evidence(a: argparse.Namespace) -> None:
    paths = {name: getattr(a, name) for name in EVIDENCE}
    numbers = write_evidence(paths, a.si_dir, a.fig_dir, a.out, run=a.live_run)
    if a.mode == "paper" and numbers["evidence"]["waiting"]:
        raise ValueError(
            "The paper run cannot leave SI results out; it waits on "
            + "; ".join(numbers["evidence"]["waiting"])
            + "."
        )


def _ebola_extra(a: argparse.Namespace) -> None:
    raw = pd.read_csv(a.input)
    _write(raw.loc[raw["model"].str.match(a.models)], a.out)


def _germany_extra(a: argparse.Namespace) -> None:
    table = germany.extra_table(
        a.input, models=a.models, suffix=a.suffix, backtest=a.backtest
    )
    _write(table, a.out)


def _summary(a: argparse.Namespace) -> None:
    studies = {}
    for m, t in zip(a.meta, a.scores, strict=True):
        meta = read_meta(m)
        studies[meta["study"]] = study_summary(meta, pd.read_csv(t), a.mode)
    write_summary(
        a.out,
        a.mode,
        studies,
        finetuned=a.finetuned,
        realdata_after=a.realdata_after,
    )


def _real_finetune(a: argparse.Namespace) -> None:
    def read(path: Path) -> pd.DataFrame | None:
        frame = pd.read_csv(path) if path.stat().st_size else None
        return None if frame is None or frame.empty else frame

    table = real_finetune_combined(read(a.germany), read(a.ebola))
    a.out.write_text(to_markdown(table))


def _score_tables(a: argparse.Namespace) -> None:
    score_tables(
        a.study,
        _read(a.scores),
        _read(a.locations),
        a.main_dir,
        a.si_dir,
        split=_date(a.split),
        crps_scores=_read(a.crps_scores),
        baseline=baseline_of(read_meta(a.meta)),
        after=a.realdata_after,
    )


def _limits(p: argparse.ArgumentParser) -> None:
    p.add_argument("study")
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    p.add_argument("--before", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--synthetic-forecasts", type=Path, default=None)
    p.add_argument("--germany-forecasts", type=Path, nargs="*", default=[])
    p.add_argument("--germany-triangle", type=Path, default=None)
    p.add_argument("--hub", type=Path, default=None)
    p.add_argument("--ebola-data", type=Path, default=None)
    p.add_argument("--ebola-config", type=Path, default=None)
    p.add_argument("--ebola-backtest", type=Path, default=None)
    p.add_argument("--ebola-extra", type=Path, nargs="*", default=[])
    p.add_argument("--ebola-draws", type=Path, default=None)
    p.add_argument("--ebola-draws-models", nargs="*", default=[])
    p.add_argument("--candidates", type=Path, nargs="*", default=[])
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--store-ref", default="origin/fit-store")


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
    p.add_argument("--note", default="", help="What is provisional.")
    p.add_argument("--mode", required=True)
    p.add_argument("--config", required=True, help="Workflow config as JSON.")
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
    p.add_argument(
        "--first-after",
        action="store_true",
        help="Take the first release on or after D days, not the last before.",
    )
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--locations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_truth)

    p = sub.add_parser("scores")
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--crps-out", type=Path, required=True)
    p.set_defaults(run=_scores)

    p = sub.add_parser("figures")
    _outputs(p)
    p.add_argument("--crps-scores", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--config", required=True, help="Workflow config as JSON.")
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.set_defaults(run=_figures)

    p = sub.add_parser("score-tables")
    _outputs(p)
    p.add_argument("--meta", type=Path, required=True)
    # The last BDBV release the real-data models trained on.
    p.add_argument("--realdata-after", default=None)
    p.set_defaults(run=_score_tables)

    p = sub.add_parser("joint-scores")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.set_defaults(run=_joint)

    p = sub.add_parser("evidence")
    for name in EVIDENCE:
        p.add_argument(f"--{name.replace('_', '-')}", type=Path, default=None)
    p.add_argument("--mode", required=True)
    p.add_argument("--si-dir", type=Path, required=True)
    p.add_argument("--fig-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--live-run",
        default=LIVE_RUN,
        help="The live fine-tuning run, by the checkpoint scores' `run`.",
    )
    p.set_defaults(run=_evidence)

    p = sub.add_parser("ebola-extra")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--models", default=".")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_ebola_extra)

    p = sub.add_parser("germany-extra")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--models", default=None)
    p.add_argument("--suffix", default="")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_germany_extra)

    p = sub.add_parser("summary")
    p.add_argument("--mode", required=True)
    p.add_argument("--meta", type=Path, nargs="+", required=True)
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--finetuned", default=None)
    p.add_argument(
        "--realdata-after",
        default=None,
        help="The last BDBV release the real-data model trained on.",
    )
    p.set_defaults(run=_summary)

    p = sub.add_parser("real-finetune")
    p.add_argument("--germany", type=Path, required=True)
    p.add_argument("--ebola", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_real_finetune)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one step."""
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
