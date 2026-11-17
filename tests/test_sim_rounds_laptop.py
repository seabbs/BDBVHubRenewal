"""Laptop batches of an archie round, recorded as archie records its own."""

import subprocess
from pathlib import Path

LAPTOP = Path(__file__).parents[1] / "scripts" / "sim_rounds_laptop.sh"
HEADER = (
    "round,seed,kind,grid,grid_commit,draws,batch,counts_sha256,"
    "settings_sha256,failures_sha256,peak_rss_kb,wall_s,host,finished"
)


def run(tmp_path, code):
    return subprocess.run(
        ["bash", "-c", f'source "{LAPTOP}"; {code}'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_manifest_row_matches_archie_and_reads_macos_time(tmp_path):
    sims = tmp_path / "sims"
    (sims / "logs").mkdir(parents=True)
    (sims / "settings-30324.csv").write_text("outbreak\n1\n")
    (sims / "counts-30324.csv.gz").write_bytes(b"x")
    log = sims / "logs" / "30324.log"
    log.write_text(
        "done\n"
        "     1034.12 real       900.00 user        10.00 sys\n"
        "         2147483648  maximum resident set size\n"
    )
    run(tmp_path, f'time_line "{log}" >> "{log}"')
    assert "peak_rss_kb 2097152 wall_s 1034.12" in log.read_text()
    row = run(
        tmp_path,
        "ROUND=3; GRID=r4.1; COMMIT=abc; HOST_NAME=laptop-x; "
        f'manifest_row 30324 train "{sims}"',
    ).strip()
    fields = row.split(",")
    assert len(fields) == len(HEADER.split(","))
    assert fields[:7] == ["3", "30324", "train", "r4.1", "abc", "10", "25"]
    # No failures file: its hash is empty, as archie writes it.
    assert fields[9] == ""
    assert fields[10:13] == ["2097152", "1034.12", "laptop-x"]


def test_header_is_archies():
    archie = (LAPTOP.parent / "sim_rounds.sh").read_text()
    assert HEADER in archie
    assert HEADER in LAPTOP.read_text()
