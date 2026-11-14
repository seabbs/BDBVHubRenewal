"""Check hubValidations (R) gives the results recorded in the manifest.

Skipped when Rscript or hubValidations is not installed. Set
``R_CACHE_ROOTPATH`` if R cannot write to its default cache.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from hv_fixtures import BUNDLE, FIXTURES, METADATA_DIRS, R_VALIDATOR


def _have_hubvalidations() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = !requireNamespace("
            "'hubValidations', quietly = TRUE))",
        ],
        check=False,
        capture_output=True,
    )
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _have_hubvalidations(), reason="Rscript/hubValidations not available"
)


@pytest.fixture(scope="module")
def r_results(hub_path: Path, tmp_path_factory: Any) -> dict[str, set[str]]:
    out = tmp_path_factory.mktemp("r") / "r.json"
    md = [a for d in METADATA_DIRS for a in ("--metadata-dir", str(d))]
    cmd = [
        "Rscript",
        str(R_VALIDATOR),
        "--hub",
        str(hub_path),
        *md,
        "--skip-window",
        "--json",
        str(out),
        *[str(BUNDLE / f["path"]) for f in FIXTURES],
    ]
    subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=900
    )
    res: dict[str, set[str]] = defaultdict(set)
    for x in json.loads(out.read_text()):
        res.setdefault(x["file"], set())
        if x["status"] in ("fail", "error"):
            res[x["file"]].add(x["check"])
    return res


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda f: f["path"])
def test_r_matches_manifest(
    fx: dict[str, Any], r_results: dict[str, set[str]]
) -> None:
    got = r_results[str(BUNDLE / fx["path"])]
    assert sorted(got) == fx["r_failures"]
