"""German development-half quantiles at nowcast and forecast horizons.

The German hub scores the 7-day sums ending 0 to 28 days before each
nowcast date. This adds the sums ending 7 to 28 days after it, so a
checkpoint's nowcasts and forecasts can be compared with zero-shot on
the same dates. It runs the scorer's own `germany_forecasts` and
`with_truth` with `data.germany.HORIZONS` widened, in a task store of its
own, one checkpoint per process.

    python scripts/ftpractice/german_horizons.py LABEL OUT_DIR [N_DATES]

LABEL is `reference` (zero-shot), a checkpoint label (`slots-<hash>`), or
`NAME=ADAPTER_DIR` for an unpublished adapter with the series bias, which
loads from ADAPTER_DIR as checkpoint `slots-NAME` and writes under NAME.
Writes OUT_DIR/<LABEL>/<date>.csv.gz, resuming from the dates written.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

HORIZONS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)


def main() -> None:
    """Forecast every development date at the widened horizons."""
    label, out = sys.argv[1], Path(sys.argv[2])
    from bdbvhubrenewal.data import germany as dg
    from bdbvhubrenewal.forecasters import foundation as fd

    if "=" in label:
        label, adapter = label.split("=", 1)
        local = Path(adapter).expanduser()
        spec = f"{fd.SLOTS_PREFIX}{label}"
        sums = tuple(
            (name, hashlib.sha256((local / name).read_bytes()).hexdigest())
            for name in fd.ADAPTER_FILES
        )
        fd.resolve_checkpoint = lambda *_a, **_k: fd.Checkpoint(spec, sums)
        fd.finetuned_adapter = lambda _checkpoint: local
        pin = spec
    else:
        pin = None

    dg.HORIZONS = HORIZONS
    from bdbvhubrenewal.evaluation import checkpoint as ck
    from bdbvhubrenewal.evaluation import germany as de

    os.environ["BDBV_CHRONOS_FT"] = pin or (
        ck.REFERENCE_PIN if label == ck.REFERENCE else label
    )
    models = list(ck.ZERO_SHOT if label == ck.REFERENCE else ck.FINETUNED)
    config = de.load_germany_config("config/germany.toml")
    path = Path("data/raw/germany") / Path(config.triangle).name
    triangle = dg.hub_triangle(path)
    root = out / label
    root.mkdir(parents=True, exist_ok=True)
    dates = de.nowcast_dates(config, held_out=False)
    if len(sys.argv) > 3:
        dates = dates[: int(sys.argv[3])]
    todo = [d for d in dates if not (root / f"{d}.csv.gz").exists()]
    for i in range(0, len(todo), 20):
        chunk = todo[i : i + 20]
        start = time.monotonic()
        fc = de.germany_forecasts(
            models,
            triangle,
            chunk,
            config.n_samples,
            config.seed,
            out / "tasks",
        )
        fc = de.with_truth(fc, triangle, config.truth_delay)
        for date in chunk:
            mine = fc.loc[fc["origin"] == date].reset_index(drop=True)
            ck._write(mine, root / f"{date}.csv.gz")
        ck._free()
        print(
            f"{label} {len(chunk)} dates to {chunk[-1]}: "
            f"{time.monotonic() - start:.0f} s",
            flush=True,
        )
    print(f"{label} done", flush=True)


if __name__ == "__main__":
    main()
