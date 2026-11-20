"""Publish a verified checkpoint with the trainer's own `publish`.

    python workflow/scripts/publish_checkpoint.py CHECKPOINT OUT

CHECKPOINT is a run's `n<N>/` folder. A checkpoint with the series bias
is released as `chronos-slots-<hash>`, else `chronos-ft-<hash>`, as
scripts/finetune_stream.py did with `--publish`. Safe to run again: a
release whose sums match is left as it is. An offloaded checkpoint's
held-out losses (`held-out.json`, rules/evaluate.smk) are uploaded to its
release beside its settings. OUT records the tag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_stream as fs  # noqa: E402

from nowcastfoundation.forecaster import SLOTS_PREFIX, release_tag  # noqa: E402
from nowcastfoundation.sidecar import SIDECAR  # noqa: E402


def main() -> None:
    """Publish the checkpoint in `sys.argv[1]`, recording it in argv[2]."""
    where, out = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads((where / "settings.json").read_text())
    key = settings["hash"]
    spec = f"{SLOTS_PREFIX}{key}" if settings.get("series_bias") else key
    fs.publish(where, spec)
    # Uploaded on every run, so a release made before its sidecar gets it.
    sidecar = where / SIDECAR
    if sidecar.exists():
        subprocess.run(
            ["gh", "release", "upload", release_tag(spec), str(sidecar),
             "--clobber", "-R", fs.REPO],
            check=True,
        )  # fmt: skip
    out.write_text(json.dumps({"tag": release_tag(spec), "hash": key}) + "\n")


if __name__ == "__main__":
    main()
