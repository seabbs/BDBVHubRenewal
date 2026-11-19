"""The simulator runs with the epinowcast and tools it was pinned to."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
RULES = (ROOT / "workflow" / "rules" / "simulate.smk").read_text()
SCRIPT = (ROOT / "scripts" / "finetune_simulate.R").read_text()


def batch_shell():
    """The shell of rule simulate_batch."""
    rule = RULES.split("rule simulate_batch:", 1)[1].split("\nrule ", 1)[0]
    return rule.split("shell:", 1)[1]


def test_the_simulator_loads_epinowcast_only_from_its_library():
    # From another library it ran, and every draw came out empty.
    assert re.search(r"library\(epinowcast, lib\.loc = lib\)", SCRIPT)


def test_a_batch_uses_the_machine_s_pinned_library():
    shell = batch_shell()
    assert "ENW_LIB={input.lib}" in shell or "ENW_LIB={params.lib}" in shell


def test_r_s_tools_come_from_the_system_paths():
    # R puts the system libraries on LD_LIBRARY_PATH, so a bash or make
    # from another prefix (such as linuxbrew) fails under it. Rscript is
    # found first on the usual PATH, so the R is the same.
    shell = batch_shell()
    assert "rscript=$(command -v Rscript)" in shell
    assert 'PATH="$(getconf PATH):$PATH"' in shell
    assert '"$rscript" scripts/finetune_simulate.R' in shell
