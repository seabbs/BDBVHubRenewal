"""Ready the trainer's folder for one training segment.

    prepare_segment.py TRAIN RUN ROUNDS_DIR [--prev DIR --prev-n N]
        [--release-rounds R ... --repo REPO]

Puts the state the segment resumes from in TRAIN/state
(`flowlib.segments.resume_state`), fetches the released QC exclusions
of the rounds it trains on when its batches come from releases
(`flowlib.exclusions.fetch_released`), and writes every known exclusion
to RUN/excluded.csv, the trainer's `--exclude`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import exclusions, segments


def main() -> None:
    """Ready the segment's folder."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("rounds_dir", type=Path)
    p.add_argument("--prev", type=Path)
    p.add_argument("--prev-n", type=int)
    p.add_argument("--release-rounds", type=int, nargs="*", default=[])
    p.add_argument("--repo")
    a = p.parse_args()
    print("state:", segments.resume_state(a.train / "state", a.prev, a.prev_n))
    released = a.run / "excluded-releases"
    if a.release_rounds:
        got = exclusions.fetch_released(a.release_rounds, a.repo, released)
        print("released exclusions:", got)
    paths = [*a.rounds_dir.glob("r*/qc/excluded.csv"), *released.glob("*.csv")]
    segments.gather_exclusions(paths, a.run / "excluded.csv")
    print(f"exclusions: {len(segments.excluded(a.run / 'excluded.csv'))}")


if __name__ == "__main__":
    main()
