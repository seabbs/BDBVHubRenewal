"""Training segments: one per checkpoint prefix, resumed in turn.

The continuous run (`scripts/finetune_stream.py`) checkpoints after the
first N outbreaks of its seed list, for each prefix N. The workflow runs
it once per prefix: segment N resumes from the trainer's state after the
prefix before and stops (`--until N`) once prefix N is saved. The
learning-rate schedule, step and batch position are in that state, so
the segments in turn are the one continuous run.

A batch (`scripts/finetune_simulate.R <n> <seed>`) holds the outbreaks
`<seed>-<index>` its settings file lists, less those the sim QC
excluded, so how many batches a prefix needs is known only once they
are simulated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path


def segments(prefixes: Iterable[int]) -> list[tuple[int | None, int]]:
    """Return each prefix with the prefix it resumes from (None: base)."""
    ordered = sorted(set(prefixes))
    return list(zip([None, *ordered[:-1]], ordered, strict=True))


def check_chunks(prefixes: Iterable[int], chunk: int) -> None:
    """Raise if a prefix does not end a training chunk.

    The trainer takes outbreaks in chunks of `chunk`, so it can stop
    cleanly only at a chunk's end.
    """
    bad = [p for p in prefixes if p % chunk]
    if bad:
        raise ValueError(
            f"prefixes {bad} do not end a chunk of {chunk} outbreaks"
        )


def round_seeds(
    round_: int, base: int, train: int, held: int
) -> tuple[list[int], list[int]]:
    """Return a round's training and held-out batch seeds.

    As `scripts/sim_rounds.sh`: round r takes the seeds from
    `base + 100 r`, its `train` training batches then `held` held-out.
    """
    first = base + 100 * round_
    seeds = list(range(first, first + train + held))
    return seeds[:train], seeds[train:]


def seeds_for(
    n: int, seeds: Sequence[int], kept: Callable[[int], int]
) -> list[int]:
    """Return the first seeds of `seeds` whose batches hold `n` outbreaks.

    `kept(seed)` gives a batch's outbreaks after exclusions. It is asked
    in order and only until there are enough, so in a Snakemake input
    function it can wait on each batch through a checkpoint.
    """
    total = 0
    out = []
    for seed in seeds:
        if total >= n:
            break
        total += kept(seed)
        out.append(seed)
    if total < n:
        raise ValueError(f"the seed list holds only {total} outbreaks, not {n}")
    return out


def excluded(path: Path | str) -> set[str]:
    """Return the simulations a sim-QC exclusion file lists.

    As the trainer reads it: one a line, or a CSV with a `simulation`
    column first; no file is no exclusions.
    """
    path = Path(path)
    if not path.exists():
        return set()
    lines = [x.strip() for x in path.read_text().splitlines()]
    if lines and lines[0].startswith("simulation"):
        lines = [x.split(",")[0] for x in lines[1:]]
    return {x for x in lines if x}


def kept_outbreaks(settings: Path | str, skip: set[str]) -> int:
    """Return how many outbreaks of a batch's settings file are kept."""
    rows = Path(settings).read_text().splitlines()[1:]
    ids = {r.split(",")[0] for r in rows if r.strip()}
    return len(ids - skip)


def _done(state: Path) -> int:
    import json

    return int(json.loads((state / "state.json").read_text())["done"])


def resume_state(
    run: Path | str, prev: Path | str | None, prev_n: int | None
) -> str:
    """Put the state a segment resumes from in the trainer's `run` folder.

    The trainer resumes from `run` (its `OUT/state`). That is kept if it
    has reached the prefix before, `prev_n`, as a segment that stopped
    part way, or a live run whose checkpoints kept no state, resumes
    from its own saves. Otherwise the state kept with that prefix
    (`prev`, `n<P>/state`) is copied in.

    Returns
    -------
    str
        "base" (first segment), "kept" or "copied".

    Raises
    ------
    ValueError
        If there is no state at or past the prefix before.
    """
    import shutil

    run = Path(run)
    if prev_n is None:
        return "base"
    if (run / "state.json").exists() and _done(run) >= prev_n:
        return "kept"
    if prev is None or not (Path(prev) / "state.json").exists():
        raise ValueError(
            f"no state at or past n{prev_n} to resume from: neither {run} "
            f"nor {prev}"
        )
    shutil.rmtree(run, ignore_errors=True)
    shutil.copytree(prev, run)
    return "copied"


def reachable(n: int, seeds: Sequence[int], kept: Callable[[int], int]) -> bool:
    """Whether the seed list can hold `n` outbreaks."""
    try:
        seeds_for(n, seeds, kept)
    except ValueError:
        return False
    return True


def gather_exclusions(paths: Iterable[Path | str], out: Path | str) -> None:
    """Write the sim-QC exclusions of several rounds as one CSV."""
    rows = sorted(set().union(*(excluded(p) for p in paths)))
    Path(out).write_text("".join(["simulation\n", *(f"{r}\n" for r in rows)]))
