"""Every workspace package module imports on its own."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _modules() -> list[str]:
    names = []
    for src in sorted(ROOT.glob("packages/*/src")):
        for path in sorted(src.rglob("*.py")):
            parts = path.relative_to(src).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            names.append(".".join(parts))
    return names


@pytest.mark.parametrize("module", _modules())
def test_package_module_imports_first(module: str) -> None:
    # A new interpreter, so that no earlier import (of the hub package,
    # say) hides an import cycle that a user importing this module first
    # meets.
    run = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
