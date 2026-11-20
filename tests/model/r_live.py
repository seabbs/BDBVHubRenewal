"""Run R code on JSON inputs and read its JSON output, for live checks.

Tests compare our components with R packages (epinowcast, primarycensored,
EpiNow2, the simulator's R engine) by calling them, not through committed
fixtures. A test skips, with the reason, only where R or a package is
missing; CI installs them (`.github/workflows/test.yml`, job `r-live`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = """
lib <- Sys.getenv("ENW_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::read_json(args[1], simplifyVector = TRUE)
out <- local({
%s
})
jsonlite::write_json(out, args[2], digits = NA, auto_unbox = TRUE,
  matrix = "rowmajor")
"""


def require_r(packages: Sequence[str]) -> None:
    """Skip the calling test unless Rscript and `packages` are available.

    Parameters
    ----------
    packages : sequence of str
        R packages the check calls.
    """
    if shutil.which("Rscript") is None:
        pytest.skip("needs R (Rscript is not on the PATH)")
    names = ", ".join(f'"{p}"' for p in ["jsonlite", *packages])
    code = (
        'lib <- Sys.getenv("ENW_LIB"); '
        "if (nzchar(lib)) .libPaths(c(lib, .libPaths())); "
        "miss <- Filter(function(p) !requireNamespace(p, quietly = TRUE), "
        f"c({names})); "
        'cat(paste(miss, collapse = " "))'
    )
    out = subprocess.run(
        ["Rscript", "-e", code], capture_output=True, text=True, check=False
    )
    missing = out.stdout.strip()
    if out.returncode != 0 or missing:
        pytest.skip(f"needs the R packages {missing or packages}")


def run_r(
    code: str,
    inputs: Any,
    packages: Sequence[str] = (),
    timeout: float = 1800,
) -> Any:
    """Run R `code` on `inputs` and return its value, through JSON.

    In the code, `input` holds the inputs (`jsonlite::read_json`,
    simplified); the value of its last expression is returned.

    Parameters
    ----------
    code : str
        R code, the body of a `local` block.
    inputs : object
        JSON-serialisable inputs.
    packages : sequence of str
        R packages the code needs; the test skips without them.
    timeout : float
        Seconds before the run is stopped.

    Returns
    -------
    object
        The code's value, read back from JSON.
    """
    require_r(packages)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, script = (
            Path(tmp) / f for f in ("in.json", "out.json", "run.R")
        )
        src.write_text(json.dumps(inputs))
        script.write_text(_PRELUDE % code)
        env = {**os.environ}
        out = subprocess.run(
            ["Rscript", str(script), str(src), str(dst)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        assert out.returncode == 0, out.stderr[-4000:]
        return json.loads(dst.read_text())
