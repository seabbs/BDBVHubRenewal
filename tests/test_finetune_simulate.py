"""The fine-tune simulator's factorial design and its delay forms."""

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
draws <- 2L
"""

needs_r = pytest.mark.skipif(
    not LIB.exists() or shutil.which("Rscript") is None,
    reason="needs R and epinowcast's simulate branch in output/finetune/rlib",
)


def run_r(script: str, tmp_path: Path) -> str:
    """Run R code after loading the simulator and return its output."""
    path = tmp_path / "run.R"
    path.write_text(SOURCE + script)
    out = subprocess.run(
        ["Rscript", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return out.stdout


@needs_r
def test_every_factor_level_appears_equally_in_a_block(tmp_path):
    out = run_r(
        """
cells <- lapply(0:(block - 1L), cell_for)
for (f in names(factors)) {
  counts <- table(vapply(cells, `[[`, "", f))
  cat(f, length(unique(counts)) == 1L, "\\n")
}
""",
        tmp_path,
    )
    assert all(line.split()[-1] == "TRUE" for line in out.strip().splitlines())


@needs_r
@pytest.mark.parametrize("form", ["parametric", "non_parametric", "both"])
def test_each_delay_form_simulates(form, tmp_path):
    out = run_r(
        f"""
set.seed(3)
d <- draw_design(1L)
d$strata <- "single"; d$n_places <- 1L; d$n_ages <- 1L
d$delay_form <- "{form}"; d$delay_change <- "walk"
v <- draw_values(d); v$weeks <- 16L
cat("cases", sum(simulate_outbreak(d, v)$counts$count), "\\n")
""",
        tmp_path,
    )
    assert int(out.split("cases")[-1].split()[0]) > 0
