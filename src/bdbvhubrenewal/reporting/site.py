# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The docs site's settings, from `config/site.toml`.

Only the reporting reads them, so they never reach a stored fit's code or
the paper. `site_baseline` is the model the site's scores are relative to
by default, and `site_models` keeps the models the site shows.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import Final

__all__ = [
    "SITE_CONFIG",
    "SiteConfig",
    "load_site_config",
    "site_baseline",
    "site_config",
    "site_models",
]

SITE_CONFIG: Final = Path("config/site.toml")
"""The site settings, from the repository root."""


@dataclass(frozen=True)
class SiteConfig:
    """The docs site's settings.

    Attributes
    ----------
    baseline : str
        The model scores are relative to by default (`[scores] baseline`).
    hide : tuple of str
        Patterns (`fnmatch`) of models left off the site.
    show : tuple of str
        Models shown even when a `hide` pattern matches them.
    """

    baseline: str
    hide: tuple[str, ...] = ()
    show: tuple[str, ...] = ()

    def shows(self, model: str) -> bool:
        """Return whether the site shows a model."""
        if model in self.show:
            return True
        return not any(fnmatchcase(model, p) for p in self.hide)


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
    models = data.get("models", {})
    return SiteConfig(
        baseline=str(data["scores"]["baseline"]),
        hide=tuple(str(p) for p in models.get("hide", ())),
        show=tuple(str(m) for m in models.get("show", ())),
    )


@cache
def site_config() -> SiteConfig:
    """Return the site settings from `SITE_CONFIG`, read once."""
    return load_site_config()


def site_baseline() -> str:
    """Return the model the site's scores are relative to by default."""
    return site_config().baseline


def site_models(models: Iterable[str]) -> list[str]:
    """Keep the models the site shows, in their order.

    Parameters
    ----------
    models : iterable of str
        Model names; hub teams' names pass, as no pattern matches them.

    Returns
    -------
    list of str
    """
    cfg = site_config()
    return [m for m in models if cfg.shows(m)]
