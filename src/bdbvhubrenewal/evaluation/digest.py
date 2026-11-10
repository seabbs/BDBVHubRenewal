# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Hash what a backtest run reads, to key its cached outputs.

The Pages workflow runs the backtests that fit no model as separate
jobs and caches each one's outputs under `backtest_digest` of what it
reads. A change to report-only code or to docstrings and comments then
reruns none of them.

This lives apart from `evaluation.store`, whose own code is part of
every stored fit's key, so a change here refits nothing.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from bdbvhubrenewal.evaluation.store import _strip

__all__ = [
    "BACKTEST_INPUTS",
    "FORECAST_ENTRIES",
    "REPORT_ONLY",
    "backtest_digest",
    "model_code",
    "model_digest",
]

REPORT_ONLY: Final = tuple(
    f"src/bdbvhubrenewal/reporting/{name}.py"
    for name in (
        "hubtables",
        "nowcast_plots",
        "report",
        "scorechart",
        "store_note",
        "tables",
        "zone_nowcast",
    )
)
"""Modules only the `report` command runs, so no backtest reads them."""

BACKTEST_INPUTS: Final = ("src/bdbvhubrenewal", "config", "data")
"""What every backtest run reads, relative to the repository root."""

_SKIP: Final = ("data/raw", "data/sources.json", *REPORT_ONLY)
"""Not read by a backtest. `data/sources.json` records when the raw
downloads were made."""


def backtest_digest(
    root: Path | str = ".", extra: Sequence[str] = (), text: str = ""
) -> str:
    """Hash what a backtest run reads.

    The files in `BACKTEST_INPUTS` and `extra`, less `REPORT_ONLY`,
    `data/raw` and `data/sources.json`, are hashed by path and content,
    Python files as syntax trees without docstrings (as
    `evaluation.store.code_digest`). Then `text` (such as the commands
    run), the Python minor version and the locked runtime dependencies.

    Parameters
    ----------
    root : pathlib.Path or str
        The repository root.
    extra : sequence of str
        Further files or directories the run reads, relative to `root`.
    text : str
        Added to the hash as it is.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    return _digest(Path(root), (*BACKTEST_INPUTS, *extra), text)


def _digest(base: Path, names: Sequence[str], text: str) -> str:
    """Hash files by path and content, then `text` and the runtime."""
    skipped = [base / s for s in _SKIP]
    files: list[Path] = []
    for name in names:
        path = base / name
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if any(path == s or s in path.parents for s in skipped):
            continue
        digest.update(path.relative_to(base).as_posix().encode())
        if path.suffix == ".py":
            digest.update(_strip(path.read_text()).encode())
        else:
            digest.update(path.read_bytes())
    digest.update(text.encode())
    digest.update(repr(sys.version_info[:2]).encode())
    deps = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--no-hashes",
            "--no-header",
            "--no-emit-project",
        ],
        cwd=base,
        capture_output=True,
        text=True,
        check=True,
    )
    digest.update(deps.stdout.encode())
    return digest.hexdigest()


FORECAST_ENTRIES: Final = (
    "src/bdbvhubrenewal/evaluation/backtest.py",
    "src/bdbvhubrenewal/evaluation/nowcast.py",
    "src/bdbvhubrenewal/evaluation/zones.py",
)
"""The modules whose forecasting halves a forecast-only run calls."""

_PACKAGE: Final = "bdbvhubrenewal"


def _module_file(src: Path, module: str) -> Path | None:
    """Return the file of a module of the package, or None."""
    parts = module.split(".")
    if parts[0] != _PACKAGE:
        return None
    base = src.joinpath(*parts)
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _absolute(path: Path, src: Path, node: ast.ImportFrom) -> str:
    """Return the absolute module of a `from ... import` statement."""
    if not node.level:
        return node.module or ""
    # The package holding the module: for `pkg/__init__.py`, `pkg`.
    pkg = path.relative_to(src).with_suffix("").parts[:-1]
    base = ".".join(pkg[: len(pkg) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def _parents(src: Path, module: str) -> set[Path]:
    """Return the `__init__.py` of every package above a module."""
    parts = module.split(".")
    found = set()
    for k in range(1, len(parts)):
        hit = _module_file(src, ".".join(parts[:k]))
        if hit is not None and hit.name == "__init__.py":
            found.add(hit)
    return found


def _source_of(src: Path, init: Path, name: str) -> set[Path]:
    """Return the modules a package's `__init__.py` takes `name` from."""
    found = set()
    for node in ast.walk(ast.parse(init.read_text())):
        if isinstance(node, ast.ImportFrom) and any(
            a.name == name or a.asname == name for a in node.names
        ):
            module = _absolute(init, src, node)
            hit = _module_file(src, f"{module}.{name}") or _module_file(
                src, module
            )
            if hit is not None:
                found.add(hit)
    return found


def _imports(path: Path, src: Path) -> set[Path]:
    """Return the package modules a module imports, and their packages.

    Every import is followed, at module level or in a function, since a
    function's import runs when it is called. A name taken from a
    package is followed to the module its `__init__.py` takes it from,
    so a package's re-exports of other modules are not all followed. The
    `__init__.py` of each package above an imported module is listed.
    """
    found: set[Path] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _module_file(src, alias.name)
                if hit is not None:
                    found |= {hit, *_parents(src, alias.name)}
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(path, src, node)
            hit = _module_file(src, module)
            if hit is None:
                continue
            found |= {hit, *_parents(src, module)}
            for alias in node.names:
                sub = _module_file(src, f"{module}.{alias.name}")
                if sub is not None:
                    found.add(sub)
                elif hit.name == "__init__.py":
                    found |= _source_of(src, hit, alias.name)
    return found


def _follows(path: Path) -> bool:
    """Whether a module's own imports are followed.

    A package's `__init__.py` is only read: what is taken from it is
    followed to its source (`_imports`).
    """
    return path.name != "__init__.py"


def _model_modules() -> dict[str, str]:
    """Map each registered model to the module that defines it."""
    from bdbvhubrenewal.forecasters import list_models

    out = {}
    for spec in list_models():
        factory = getattr(spec.factory, "func", spec.factory)
        out[spec.name] = str(factory.__module__)
    return out


def _closure(
    src: Path,
    todo: list[Path],
    stop: set[Path],
    start: Path | None,
    seen: set[Path] | None = None,
) -> set[Path]:
    """Return the modules `todo` import, not going into `stop`."""
    seen = set() if seen is None else set(seen)
    todo = list(todo)
    while todo:
        path = todo.pop()
        if path in seen or path in stop:
            continue
        seen.add(path)
        if _follows(path) or path == start:
            todo += sorted(_imports(path, src) - seen)
    return seen


def model_code(
    model: str,
    root: Path | str = ".",
    entries: Sequence[str] = FORECAST_ENTRIES,
    cli: bool = True,
) -> list[str]:
    """List the source files a forecast-only run of one model reads.

    The import closure, within the package, of the model's own module, with
    any other model's module it builds on, and of `entries`,
    which stops at the modules that define the other models: the registry
    imports every model, but a model's forecasts do not run the others'
    code. With `cli`, the CLI module is added as a file, without
    following its imports, which reach every command.

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.
    entries : sequence of str
        The shared modules the run calls.
    cli : bool
        Add the CLI module.

    Returns
    -------
    list of str
        Paths relative to `root`, sorted.
    """
    base = Path(root)
    src = base / "src"
    modules = _model_modules()
    own = modules[model]
    others = {
        f
        for name, m in modules.items()
        if m != own and (f := _module_file(src, m)) is not None
    }
    start = _module_file(src, own)
    # Everything the model's own module needs, another model's module
    # included when it builds on it (pprandomwalknowcast on randomwalk.py).
    # A model defined outside the package (a test's stand-in) has only the
    # shared code.
    seen = set() if start is None else _closure(src, [start], set(), start)
    # Then the shared forecasting code, which reaches every model through
    # the registry: there the other models' modules are left out.
    seen = _closure(
        src, [base / e for e in entries], others - seen, start, seen
    )
    if cli:
        seen.add(src / _PACKAGE / "cli.py")
    return sorted(p.relative_to(base).as_posix() for p in seen)


def model_digest(
    model: str,
    root: Path | str = ".",
    extra: Sequence[str] = (),
    text: str = "",
) -> str:
    """Hash what a forecast-only run of one model reads.

    `model_code` (as syntax trees without docstrings), `config/`, `data/`
    less `data/raw` and `data/sources.json`, `extra`, `text`, the Python
    minor version and the locked runtime dependencies. A change to
    another model's code leaves it as it is.

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.
    extra : sequence of str
        Further files or directories the run reads.
    text : str
        Added to the hash as it is, such as the commands run.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    return _digest(
        Path(root), (*model_code(model, root), "config", "data", *extra), text
    )
