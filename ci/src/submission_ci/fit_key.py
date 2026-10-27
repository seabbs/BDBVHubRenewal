"""Content-hash cache keys for fits.

A fit is reused only when everything it depends on is unchanged: the
model source (the paths listed once, in `policy.toml` under
`[cache] inputs`), the hub data (pinned by the hub commit) and the
candidate's own settings (model config and sampler). This mirrors
`fit_content_hash` in BVDOutbreakSize, where the Actions cache key and
the on-disk key come from one list so they cannot drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

IGNORED_PARTS = frozenset({"__pycache__", ".git", ".venv", ".mypy_cache"})


def _files(root: Path, inputs: Iterable[str]) -> list[Path]:
    out: set[Path] = set()
    for item in inputs:
        p = root / item
        if p.is_file():
            out.add(p)
        elif p.is_dir():
            out.update(
                f
                for f in p.rglob("*")
                if f.is_file() and not IGNORED_PARTS & set(f.parts)
            )
        else:
            raise FileNotFoundError(f"fit input {item!r} not found in {root}")
    return sorted(out)


def source_hash(root: Path, inputs: Iterable[str]) -> str:
    """Hash the contents and relative paths of the fit's source inputs."""
    h = hashlib.sha256()
    for f in _files(root, inputs):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def fit_key(
    source: str, hub_ref: str, spec: Mapping[str, Any], length: int = 16
) -> str:
    """Key for one candidate fit on one data version."""
    payload = json.dumps(
        {"source": source, "hub_ref": hub_ref, "spec": spec}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def main(argv: Sequence[str] | None = None) -> int:
    """Print the source hash of a set of inputs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+")
    p.add_argument("--root", type=Path, default=Path())
    args = p.parse_args(argv)
    print(source_hash(args.root, args.inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
