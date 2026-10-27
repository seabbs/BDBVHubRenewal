# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Sync the PyRenew model code and its tests from the prototype.

Usage: `uv run python scripts/sync_pyrenew.py <prototype dir>`.

Copies the prototype's package (`src/province_forecast/`) into
`src/bdbv_hub_renewal/pyrenew/` and its tests (`tests/`) into
`tests/pyrenew_model/`. The prototype adapter's own fit path is dropped
(`ADAPTER_DROP`), with its tests, since `models/renewal.py` does that
work. References to the prototype's package and to the
old name of this one (`bdbv_forecasts`) are rewritten. The province
centres stay in `src/bdbv_hub_renewal/province_centres.csv`, where
`bdbv_hub_renewal.data.load_province_centres` reads them. The repository's
adapter (`src/bdbv_hub_renewal/models/renewal.py`) is not synced. Run
`task format` afterwards.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src" / "bdbv_hub_renewal" / "pyrenew"
TESTS = ROOT / "tests" / "pyrenew_model"
HEADER = (
    "# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.\n"
    "# Synced from the prototype by scripts/sync_pyrenew.py. Edit there.\n"
)
REPLACE = (
    ("province_forecast", "bdbv_hub_renewal.pyrenew"),
    ("bdbv_forecasts", "bdbv_hub_renewal"),
    (
        'Path(__file__).parent / "data" / "province_centres.csv"',
        'Path(__file__).parent.parent / "province_centres.csv"',
    ),
)
ADAPTER_DROP = (
    "ppc_rows",
    "diagnostics",
    "check_fit",
    "_horizon",
    "samples_table",
    "PyRenewProvinceModel",
    "fit_renewal",
    "forecast_renewal",
)
"""The prototype adapter's own fit path. This repository's adapter,
`models/renewal.py`, does that work, so only the input builders are kept."""
TEST_SKIP = ("test_adapter.py",)
"""Prototype tests of the dropped fit path."""
TEST_REPLACE = (
    (
        "from bdbv_hub_renewal.pyrenew.adapter import ppc_rows",
        "from bdbv_hub_renewal.models.renewal import ppc_rows",
    ),
)
DATA_PATH = re.compile(r'"/[^"\n]*/data"')
"""An absolute path to the main repository's data in the prototype's tests."""


def _rewrite(text: str) -> str:
    for old, new in REPLACE:
        text = text.replace(old, new)
    return DATA_PATH.sub(
        'str(Path(__file__).resolve().parents[2] / "data")', text
    )


def drop_definitions(text: str, names: tuple[str, ...]) -> str:
    """Remove top-level functions and classes by name from module source."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    spans = [
        (min([n.lineno, *(d.lineno for d in n.decorator_list)]), n.end_lineno)
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in names
    ]
    for start, end in sorted(spans, reverse=True):
        del lines[start - 1 : end]
    return "".join(lines)


def _copy_tree(src: Path, dest: Path, header: bool) -> list[str]:
    if dest.exists():
        shutil.rmtree(dest)
    copied = []
    for path in sorted(src.rglob("*")):
        if "__pycache__" in path.parts or path.is_dir():
            continue
        rel = path.relative_to(src)
        if rel.parts[0] == "data" and path.name == "province_centres.csv":
            continue
        if not header and path.name in TEST_SKIP:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py":
            text = _rewrite(path.read_text())
            if header and path.name == "adapter.py":
                text = drop_definitions(text, ADAPTER_DROP)
            if not header:
                for old, new in TEST_REPLACE:
                    text = text.replace(old, new)
            out.write_text((HEADER if header else "") + text)
        else:
            shutil.copy(path, out)
        copied.append(str(rel))
    return copied


def main(prototype: str) -> None:
    """Copy the prototype's package and tests into this repository."""
    proto = Path(prototype)
    shutil.copy(
        proto / "src" / "province_forecast" / "data" / "province_centres.csv",
        ROOT / "src" / "bdbv_hub_renewal" / "province_centres.csv",
    )
    code = _copy_tree(proto / "src" / "province_forecast", DEST, header=True)
    tests = _copy_tree(proto / "tests", TESTS, header=False)
    print(
        f"Synced {len(code)} files into {DEST} and {len(tests)} into {TESTS}."
    )


if __name__ == "__main__":
    main(sys.argv[1])
