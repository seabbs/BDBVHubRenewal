# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The docs site's settings, from `config/site.toml`.

Only the reporting reads them, so they never reach a stored fit's code or
the paper. `site_baseline` is the model the site's scores are relative to
by default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

__all__ = ["SITE_CONFIG", "SiteConfig", "load_site_config", "site_baseline"]

SITE_CONFIG: Final = Path("config/site.toml")
"""The site settings, from the repository root."""


@dataclass(frozen=True)
class SiteConfig:
    """The docs site's settings.

    Attributes
    ----------
    baseline : str
        The model scores are relative to by default (`[scores] baseline`).
    """

    baseline: str


def load_site_config(path: Path | str = SITE_CONFIG) -> SiteConfig:
    """Read the site settings.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with a `[scores]` table.

    Returns
    -------
    SiteConfig
    """
    data = tomllib.loads(Path(path).read_text())
    return SiteConfig(baseline=str(data["scores"]["baseline"]))


@cache
def site_baseline() -> str:
    """Return the model the site's scores are relative to by default."""
    return load_site_config().baseline
