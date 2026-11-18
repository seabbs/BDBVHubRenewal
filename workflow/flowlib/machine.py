"""Cores, memory and accelerator of this machine, for the default profile.

Every workflow run uses `workflow/profiles/default`, with the cores and
resources from here: the machine's cores and memory less the headroom
its local config keeps free (`HEADROOM_CORES`, `HEADROOM_GB`; `CORES`
and `MEM_MB` set them outright), and its device, `DEVICE` or else CUDA
when `nvidia-smi` lists a GPU, MPS on Apple silicon, or the CPU.
Nothing about a machine is committed.

    python3 workflow/flowlib/machine.py args     # Snakemake options
    python3 workflow/flowlib/machine.py device   # cuda, mps or cpu
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig


def gpus(path: str | None = None) -> int:
    """Return how many CUDA GPUs `nvidia-smi` lists (0 without it)."""
    smi = shutil.which("nvidia-smi", path=path)
    if not smi:
        return 0
    out = subprocess.run([smi, "-L"], capture_output=True, text=True)
    if out.returncode:
        return 0
    return sum(line.startswith("GPU ") for line in out.stdout.splitlines())


def device(
    cfg: Mapping[str, str],
    path: str | None = None,
    system: str | None = None,
    arch: str | None = None,
) -> str:
    """Return the device to run on: `DEVICE` unless auto, else detected."""
    set_ = cfg.get("DEVICE", "")
    if set_ and set_ != "auto":
        return set_
    if gpus(path):
        return "cuda"
    system = system or platform.system()
    arch = arch or platform.machine()
    if system == "Darwin" and arch == "arm64":
        return "mps"
    return "cpu"


_device = device


THREAD_VARS = (
    "TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
"""The thread counts a torch or JAX job reads, all set to its threads."""


def torch_env(
    cfg: Mapping[str, str],
    threads: int,
    device: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    """Return the environment of a job running torch here.

    The job runs on `device`, by default this machine's (`device`), with
    `threads` CPU threads, its rule's. `CHRONOS_BATCH` in the local
    config, if set, is the series in one Chronos-2 forward pass.
    """
    env = {"DEVICE": device or _device(cfg, path)}
    env |= dict.fromkeys(THREAD_VARS, str(threads))
    if cfg.get("CHRONOS_BATCH"):
        env["CHRONOS_BATCH"] = str(cfg["CHRONOS_BATCH"])
    return env


def shell_env(env: Mapping[str, str]) -> str:
    """Return `env` as shell assignments to put before a command."""
    return " ".join(f"{k}={v}" for k, v in env.items())


def machine_mb() -> int:
    """Return this machine's memory in MB."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout) // 2**20


def budget(
    cfg: Mapping[str, str],
    cores: int | None = None,
    total_mb: int | None = None,
) -> dict[str, int]:
    """Return the cores and memory (MB) the workflow may use here."""
    if cfg.get("CORES"):
        n = int(cfg["CORES"])
    else:
        n = (cores or os.cpu_count() or 1) - int(cfg.get("HEADROOM_CORES") or 0)
    if cfg.get("MEM_MB"):
        mem = int(cfg["MEM_MB"])
    else:
        total = machine_mb() if total_mb is None else total_mb
        mem = total - int(cfg.get("HEADROOM_GB") or 0) * 1024
    return {"cores": max(1, n), "mem_mb": max(1024, mem)}


def args(
    cfg: Mapping[str, str],
    cores: int | None = None,
    total_mb: int | None = None,
    path: str | None = None,
) -> list[str]:
    """Return the Snakemake options of this machine's cores and resources.

    `gpu` is one trainer at a time, on whichever device; `sim_slots`
    (`SIM_SLOTS`, default 2) caps simulation batches; `awaiting` lets one
    batch from another host be waited for at a time.
    """
    got = budget(cfg, cores, total_mb)
    return [
        "--cores",
        str(got["cores"]),
        "--resources",
        f"mem_mb={got['mem_mb']}",
        f"gpu={max(1, gpus(path))}",
        f"sim_slots={cfg.get('SIM_SLOTS') or 2}",
        "awaiting=1",
    ]


if __name__ == "__main__":
    cfg = localconfig.read()
    what = sys.argv[1] if len(sys.argv) > 1 else "args"
    if what == "device":
        print(device(cfg))
    else:
        print(" ".join(args(cfg)))
