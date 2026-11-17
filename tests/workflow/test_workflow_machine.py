"""One default profile: this machine's cores, memory and GPU, less headroom."""

import os
import subprocess
from pathlib import Path

from flowlib import machine

ROOT = Path(__file__).parents[2]


def _nvidia_smi(tmp_path, gpus):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    lines = "".join(f"echo 'GPU {i}: Fake (UUID: x)'\n" for i in range(gpus))
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\n{lines}")
    (bin_dir / "nvidia-smi").chmod(0o755)
    return str(bin_dir)


def test_a_set_device_wins():
    assert machine.device({"DEVICE": "cpu"}, path="") == "cpu"


def test_cuda_is_found_with_nvidia_smi(tmp_path):
    path = _nvidia_smi(tmp_path, 2)
    assert machine.device({}, path=path) == "cuda"
    assert machine.gpus({}, path=path) == 2


def test_apple_silicon_has_mps():
    got = machine.device({}, path="", system="Darwin", arch="arm64")
    assert got == "mps"


def test_otherwise_the_cpu():
    assert machine.device({}, path="", system="Linux", arch="x86_64") == "cpu"


def test_cores_and_memory_less_the_headroom():
    cfg = {"HEADROOM_CORES": "4", "HEADROOM_GB": "16"}
    got = machine.budget(cfg, cores=16, total_mb=65536)
    assert got["cores"] == 12
    assert got["mem_mb"] == 65536 - 16 * 1024


def test_set_cores_and_memory_win_over_the_machine():
    cfg = {"CORES": "3", "MEM_MB": "8000", "HEADROOM_CORES": "4"}
    got = machine.budget(cfg, cores=16, total_mb=65536)
    assert got["cores"] == 3
    assert got["mem_mb"] == 8000


def test_headroom_never_leaves_less_than_one_core():
    got = machine.budget({"HEADROOM_CORES": "40"}, cores=8, total_mb=4096)
    assert got["cores"] == 1


def test_the_arguments_hold_cores_and_resources():
    args = machine.args(
        {"SIM_SLOTS": "3"}, cores=8, total_mb=32768, path="", system="Linux"
    )
    assert args[:2] == ["--cores", "8"]
    res = args[args.index("--resources") + 1 :]
    assert "mem_mb=32768" in res
    assert "sim_slots=3" in res
    assert "gpu=1" in res
    assert "awaiting=1" in res


def test_the_command_line_prints_the_arguments_and_device(tmp_path):
    local = tmp_path / "local.env"
    local.write_text("CORES=2\nMEM_MB=4096\nDEVICE=cpu\n")
    env = {**os.environ, "LOCAL_CONFIG": str(local)}
    script = ROOT / "workflow" / "flowlib" / "machine.py"
    out = subprocess.run(
        ["python3", str(script), "args"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[:2] == ["--cores", "2"]
    dev = subprocess.run(
        ["python3", str(script), "device"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert dev == "cpu\n"


def test_this_machine_reads():
    got = machine.budget({})
    assert got["cores"] >= 1
    assert got["mem_mb"] > 0
