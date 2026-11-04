# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Prototype fit of the main renewal model at 2026-11-01 with a given lag.

`uniform` is the current model (uniform lag on 0..3 days). `tail` plugs
in uniform 0..3 with a weight of 0.105 spread over 4..14 days, the
release-pair estimate (scripts/publication_lag.py), through the `empirical`
path. A plug-in, not the joint in-model version.

Run as `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run python
scripts/publication_lag_fit.py tail <out dir>`.
"""

import dataclasses
import json
import sys

import numpy as np

from bdbvhubrenewal import cli
from bdbvhubrenewal.data import releases
from bdbvhubrenewal.forecasters import renewal

lag, out = sys.argv[1], sys.argv[2]
if lag == "tail":
    pmf = np.zeros(15)
    pmf[:4] = (1 - 0.105) / 4
    pmf[4:] = 0.105 / 11
    releases.publication_lag_pmf = lambda *_, **__: pmf
    renewal.VARIANTS["renewal"] = dataclasses.replace(
        renewal.VARIANTS["renewal"], overrides={"publication_lag": "empirical"}
    )
sampler = {
    "num_chains": 2,
    "num_warmup": 500,
    "num_samples": 500,
    "target_accept": 0.9,
    "max_tree_depth": 10,
    "seed": 1,
    "chain_method": "parallel",
}
sys.exit(
    cli.main(
        [
            "fit",
            "--config",
            "main",
            "--vintage",
            "2026-11-01",
            "--sampler-json",
            json.dumps(sampler),
            "--out",
            out,
        ]
    )
)
