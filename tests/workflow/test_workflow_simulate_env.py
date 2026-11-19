"""The simulator runs with the epinowcast and tools it was pinned to."""

import re
from pathlib import Path

import pytest
from flowlib import simulator
from test_workflow_dryrun import dry_run

ROOT = Path(__file__).parents[2]
SCRIPT = (ROOT / "scripts" / "finetune_simulate.R").read_text()


def test_the_simulator_loads_epinowcast_only_from_its_library():
    # From another library it ran, and every draw came out empty.
    assert re.search(r"library\(epinowcast, lib\.loc = lib\)", SCRIPT)


def test_the_pin_is_read_from_the_simulator(tmp_path):
    assert re.fullmatch(r"[0-9a-f]{7,40}", simulator.epinowcast_sha())
    other = tmp_path / "x.R"
    other.write_text("x <- 1\n")
    with pytest.raises(ValueError, match="epinowcast_sha"):
        simulator.epinowcast_sha(other)


def test_a_batch_runs_with_the_pinned_library_and_system_tools(tmp_path):
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    lib = tmp_path / "rlib" / f"epinowcast-{simulator.epinowcast_sha()}"
    command = out.stdout.split("rule simulate_batch", 1)[1]
    # The R found first on the usual PATH, then the system's tools first
    # for the whole job command (R's own children among them).
    assert command.index("rscript=$(command -v Rscript)") < command.index(
        'PATH="$(getconf PATH):$PATH"'
    )
    assert f"ENW_LIB={lib} " in " ".join(command.split())
    assert '"$rscript" scripts/finetune_simulate.R' in command


def test_heavy_jobs_wait_below_the_load_ceiling(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nMAX_LOAD=12\n"
    )
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    command = " ".join(out.stdout.split("rule simulate_batch", 1)[1].split())
    sims = tmp_path / "rounds" / "r4" / "sims"
    assert f"wait_for_room.sh 20 16 {sims} 12" in command
