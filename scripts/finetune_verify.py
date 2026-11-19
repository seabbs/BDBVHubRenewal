# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check that a fine-tuned checkpoint's hash recomputes from its commit.

A checkpoint's `settings.json` holds the sha256 of each file it depends
on and of the source of the `foundation` functions its blocks are laid
out with (`code_sha256`), and the commit the trainer ran from
(`code_commit`). This recomputes each from that commit (`git show`), the
settings hash from the hashed settings, and the sha256 of the outbreak
lists, and prints any mismatch. It exits 1 if there is one.

    python scripts/finetune_verify.py output/finetune/ckpt-r3/n4000
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

FOUNDATION = "src/bdbvhubrenewal/forecasters/foundation.py"
"""The module whose functions `code_sha256` names `foundation.<name>`."""

NOT_HASHED = (
    "hash",
    "code_commit",
    "device",
    "held_out_loss_start",
    "held_out_loss",
    "held_out_loss_nowcast",
    "held_out_loss_forecast",
    "held_out_loss_by_level",
    "held_out_loss_by_source",
    "seconds",
    "real_validation_sha256",
    "real_validation_loss",
    "real_validation_loss_nowcast",
    "real_validation_loss_forecast",
)
"""Settings written after the hash is taken."""


def function_source(text: str, name: str) -> str:
    """Return a top-level function's source as `inspect.getsource` does."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = text.splitlines(keepends=True)
            return "".join(lines[first - 1 : node.end_lineno])
    raise KeyError(name)


def git_show(commit: str, path: str) -> bytes:
    """Return a file's bytes at a commit."""
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, check=True
    ).stdout


def problems(
    where: Path, read: Callable[[str, str], bytes] = git_show
) -> list[str]:
    """Return the ways a checkpoint does not match its recorded commit."""
    settings = json.loads((where / "settings.json").read_text())
    commit = settings.get("code_commit", "")
    found = []
    for name, want in settings["code_sha256"].items():
        if name.startswith("foundation."):
            text = read(commit, FOUNDATION).decode()
            got = hashlib.sha256(
                function_source(text, name.split(".", 1)[1]).encode()
            ).hexdigest()
        else:
            got = hashlib.sha256(read(commit, name)).hexdigest()
        if got != want:
            found.append(f"{name} at {commit[:9]} is not the code hashed")
    hashed = {k: v for k, v in settings.items() if k not in NOT_HASHED}
    text = json.dumps(hashed, sort_keys=True).encode()
    if hashlib.sha256(text).hexdigest()[:12] != settings["hash"]:
        found.append("the settings do not give the hash")
    for name, key in (
        ("outbreaks.txt", "outbreaks_sha256"),
        ("consumed.txt", "consumed_sha256"),
    ):
        if key in settings:
            got = hashlib.sha256((where / name).read_bytes()).hexdigest()
            if got != settings[key]:
                found.append(f"{name} is not the list hashed")
    return found


def main() -> None:
    """Check the checkpoint in `sys.argv[1]`."""
    found = problems(Path(sys.argv[1]))
    for line in found:
        print(line)
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
