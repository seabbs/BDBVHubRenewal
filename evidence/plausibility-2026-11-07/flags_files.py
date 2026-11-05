"""Plausibility flags for the 2026-11-07 round, per model and place."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(sys.argv[1])
ART = S / "art"
FILES = {
    "randomwalk": ART
    / "inputs/candidates/randomwalk/submission/model-output/seabbs-randomwalk"
    / "2026-11-07-seabbs-randomwalk.csv",
    "randomwalknowcast": ART
    / "inputs/candidates/randomwalknowcast/submission/model-output"
    / "seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv",
    "pprandomwalknowcast": S
    / "pprw/model-output/seabbs-pprw/2026-11-07-seabbs-pprw.csv",
    "renewal": ART
    / "candidate-main/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
    "renewalsp": ART
    / "candidate-renewalsp/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
    "renewalbvd": ART
    / "candidate-bvd/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
}
AS_OF = "2026-11-01"
COMPLETE = ["2026-10-10", "2026-10-17"]  # reported at lag of 2+ weeks
RECENT = ["2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10",
          "2026-10-17", "2026-10-24", "2026-10-31"]
H0, H1, H3 = "2026-11-07", "2026-11-14", "2026-11-28"
FORECAST = ["2026-11-07", "2026-11-14", "2026-11-21", "2026-11-28"]

