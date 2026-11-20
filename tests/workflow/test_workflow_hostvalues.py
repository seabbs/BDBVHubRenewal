"""No machine's values in tracked files: they live in config/local.env.

One guard for the whole repository: home folders, host names, IP
addresses and host:port literals. Tests are not scanned, as they name
hosts to check their absence; the local config example is where machine
values are shown.
"""

import subprocess
from pathlib import Path

import pytest
from flowlib.hostvalues import problems

ROOT = Path(__file__).parents[2]


def _local_hosts() -> tuple[str, ...]:
    """Host and worker names from this machine's untracked config."""
    env = ROOT / "config" / "local.env"
    if not env.is_file():
        return ()
    names = set()
    for line in env.read_text().splitlines():
        key, _, value = line.partition("=")
        value = value.strip().strip('"')
        if key.strip() == "HOST_NAME" and value:
            names.add(value)
        if key.strip() == "WORKERS":
            names |= {w.split("=", 1)[0] for w in value.split() if "=" in w}
    return tuple(sorted(names - {"local"}))


HOSTS = _local_hosts()
"""This machine's host and worker names (config/local.env), if any."""

EXAMPLE_HOSTS = ("examplehost",)
"""A host name for the checks of what the guard finds."""

SKIP = ("tests/", "config/local.env.example", "uv.lock", "ci/uv.lock")


def _tracked() -> list[str]:
    files = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [f for f in files if not f.startswith(SKIP)]


def test_no_tracked_file_holds_a_machine_s_values():
    found = []
    for rel in _tracked():
        path = ROOT / rel
        try:
            text = path.read_text()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue
        found += [f"{rel}: {p}" for p in problems(text, HOSTS)]
    assert found == []


@pytest.mark.parametrize(
    "text",
    [
        "ROOT=/home/someone/rounds",
        "cache: /Users/someone/hf",
        "ssh examplehost ls",
        "serve on 10.0.4.2",
        "open http://127.0.0.1:8765/",
        "rsync rounds-host.example.org:22/x",
    ],
)
def test_each_kind_of_machine_value_is_found(text):
    assert problems(text, EXAMPLE_HOSTS)


@pytest.mark.parametrize(
    "text",
    [
        'ROOT = "$HOME/bdbv-rounds"',
        "path: ~/.cache/jax",
        'MONITOR_PORT = "8766"',
        "snakemake>=9.27,<10",
        "https://github.com/owner/repo",
        "bind 127.0.0.1 by default",
    ],
)
def test_documented_defaults_and_versions_are_allowed(text):
    assert problems(text, EXAMPLE_HOSTS) == []
