# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""MkDocs hook: version the site's own scripts and styles by content.

GitHub Pages lets browsers keep a file for ten minutes, so after a deploy
a page could run a new script beside an old one it calls. Each reference
to a file in `extra_javascript` or `extra_css` gets `?v=` and the first
eight characters of the file's SHA-256, so a changed file is a new URL.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_versions: dict[str, str] = {}


def on_config(config: Any) -> Any:
    """Hash each local script and stylesheet the site loads."""
    docs = Path(config["docs_dir"])
    for entry in [*config["extra_javascript"], *config["extra_css"]]:
        path = str(entry)
        if (docs / path).is_file():
            digest = hashlib.sha256((docs / path).read_bytes()).hexdigest()
            _versions[path] = digest[:8]
    return config


def on_post_page(output: str, **_: Any) -> str:
    """Add each file's version to its references in the page."""
    for path, version in _versions.items():
        output = re.sub(rf'({re.escape(path)})"', rf'\1?v={version}"', output)
    return output
