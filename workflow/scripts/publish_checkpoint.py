"""Publish a verified checkpoint with the trainer's own `publish`.

    python workflow/scripts/publish_checkpoint.py CHECKPOINT OUT

CHECKPOINT is a run's `n<N>/` folder. A checkpoint with per-series slots
is released as `chronos-slots-<hash>`, else `chronos-ft-<hash>`, as
scripts/finetune_stream.py did with `--publish`. Safe to run again: a
release whose sums match is left as it is. OUT records the tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_stream as fs  # noqa: E402

from bdbvhubrenewal.forecasters.foundation import (  # noqa: E402
    SLOTS_PREFIX,
    release_tag,
)


def main() -> None:
    """Publish the checkpoint in `sys.argv[1]`, recording it in argv[2]."""
    where, out = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads((where / "settings.json").read_text())
    key = settings["hash"]
    spec = f"{SLOTS_PREFIX}{key}" if settings.get("series_slots") else key
    fs.publish(where, spec)
    out.write_text(json.dumps({"tag": release_tag(spec), "hash": key}) + "\n")


if __name__ == "__main__":
    main()
