"""The fine-tune simulator runs designs whose report effects are a walk."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LIB = ROOT / "output" / "finetune" / "rlib"

# Loads the simulator's functions without running a batch.
SOURCE = r"""
ex <- parse("scripts/finetune_simulate.R")
for (e in ex) {
  txt <- paste(deparse(e), collapse = "")
  if (grepl("commandArgs|args\\[\\[|^if \\(sys.nframe", txt)) next
  if (grepl("^lib <-", txt)) { lib <- "output/finetune/rlib"; next }
  eval(e, globalenv())
}
"""


@pytest.mark.skipif(
    not LIB.exists() or shutil.which("Rscript") is None,
    reason="needs R and epinowcast's simulate branch in output/finetune/rlib",
)
@pytest.mark.parametrize("family", ["negbin", "negbin1d", "poisson"])
def test_report_walk_alone_simulates(family, tmp_path):
    script = (
        SOURCE
        + f"""
set.seed(3)
d <- draw_design()
d$strata <- "single"; d$n_places <- 1L; d$n_ages <- 1L; d$max_delay <- 56L
d$report_week <- "rw"; d$report_dow <- FALSE; d$report_holiday <- FALSE
d$family <- "{family}"
v <- draw_values(d); v$seed_size <- 2; v$weeks <- 16L
cat("cases", sum(simulate_outbreak(d, v)$counts$count), "\\n")
"""
    )
    path = tmp_path / "walk.R"
    path.write_text(script)
    out = subprocess.run(
        ["Rscript", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    cases = int(out.stdout.split("cases")[-1].split()[0])
    assert cases > 0
