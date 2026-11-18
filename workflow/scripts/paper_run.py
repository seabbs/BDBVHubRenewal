"""Run the paper on this machine, niced and reported (`task paper`).

    python workflow/scripts/paper_run.py [development|paper]
    python workflow/scripts/paper_run.py --targets TARGET...

The first form runs `scripts/paper.sh` (the paper workflow and render,
as Pages CI does) at nice 19 with this machine's cores less its headroom
(`flowlib.machine.budget`). For `paper` it first checks the held-out
inputs (the paper run's simulations with their READY file, and the German
held-out forecasts): while any is missing it says what it waits for and
exits 0, so a scheduled run tries again later; then it pins the final
fine-tuning run's newest checkpoint (`paper/scripts/follow_checkpoint.py`).
The second form runs those targets of the paper workflow alone. Each run
logs to PAPER_LOGS and posts start, done or FAILED with the log's first
error line (mailbox lines when NOTIFY is set, `flowlib.notify`).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import localconfig, machine, notify  # noqa: E402

RENDERS = ("development", "paper")


def say(text: str, cfg: dict[str, str]) -> None:
    """Print a line and post it as a paper line."""
    print(text, flush=True)
    notify.post(f"paper: {text}", cfg)


def missing() -> list[str]:
    """Return the paper's held-out inputs not yet here."""
    import yaml

    sys.path.insert(0, str(ROOT / "paper"))
    from paperpipe.config import run_settings

    paper = ROOT / "paper"
    config = yaml.safe_load((paper / "workflow" / "config.yaml").read_text())
    sims = paper / localconfig.expand(
        config["synthetic"]["simulations"]["paper"]
    )
    seeds = run_settings(config, "paper").synthetic_seeds
    # The simulation set writes READY beside sims/ once every batch is
    # simulated, recorded and reviewed.
    need = [sims.parent / "READY"]
    need += [
        sims / f"{kind}-{s}.{ext}"
        for s in seeds
        for kind, ext in (("counts", "csv.gz"), ("settings", "csv"))
    ]
    need.append(ROOT / "results/germany/held-out/paper-forecasts.csv.gz")
    return [str(p.relative_to(ROOT)) for p in need if not p.exists()]


def pin() -> str:
    """Pin the final run's newest checkpoint; return it (empty: pending)."""
    import yaml

    paper = ROOT / "paper"
    config = yaml.safe_load((paper / "workflow" / "config.yaml").read_text())
    got = subprocess.run(
        [
            sys.executable,
            "scripts/follow_checkpoint.py",
            config["finetune_run"],
            "--write",
            "workflow/config.yaml",
        ],
        cwd=paper,
        capture_output=True,
        text=True,
        check=True,
    )
    return got.stdout.strip()


def first_error(log: Path) -> str:
    """Return the log's first error line, cut short."""
    for line in log.read_text(errors="replace").splitlines():
        if re.search(r"error|exception", line, re.I):
            return line.strip()[:250]
    return ""


def main(argv: list[str]) -> int:
    """Run one paper job; return its exit code."""
    cfg = localconfig.read()
    cores = str(machine.budget(cfg)["cores"])
    if argv[:1] == ["--targets"]:
        name, targets = "targets", argv[1:]
        cmd = [
            "uv",
            "run",
            "--group",
            "paper",
            "--group",
            "chronos",
            "snakemake",
            "-s",
            "workflow/Snakefile",
            *targets,
            "--cores",
            cores,
            "--resources",
            "heavy=1",
            "--rerun-triggers",
            "mtime",
        ]
        where = ROOT / "paper"
    else:
        name = argv[0] if argv else "development"
        if name not in RENDERS:
            raise SystemExit(f"give one of {RENDERS} or --targets, not {name}")
        if name == "paper":
            waiting = missing()
            if waiting:
                say(f"waiting for: {', '.join(waiting)}", cfg)
                return 0
        cmd, where = ["scripts/paper.sh", name], ROOT
    logs = Path(cfg["PAPER_LOGS"]).expanduser()
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"paper-{name}.log"
    pinned = (
        f", fine-tuned checkpoint {pin() or 'pending'}"
        if name == "paper"
        else ""
    )
    say(f"started {name}{pinned} (log {log})", cfg)
    with log.open("w") as out:
        rc = subprocess.run(
            ["nice", "-n", "19", *cmd],
            cwd=where,
            env={**os.environ, "CORES": cores},
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode
    if rc:
        say(f"FAILED {name} (exit {rc}): {first_error(log)}", cfg)
    else:
        say(f"done {name}", cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
