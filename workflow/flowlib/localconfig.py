"""Machine settings from `config/local.env`.

The file is the source of truth. The shell scripts read it with
`scripts/local_config.sh` (with the fine-tuning scripts), and this must
read it the same way. The file is untracked and holds `KEY=VALUE` lines (see
`config/local.env.example`). Blank lines, comments and lines whose key
is not upper case are skipped. A value may use `$NAME` or `${NAME}` of
the environment or of an earlier line, and one pair of surrounding
quotes is dropped. A variable set in the environment wins over the
file. With no file everything runs on this machine.

Shell scripts without `scripts/local_config.sh` read one setting with
`python3 workflow/flowlib/localconfig.py KEY`.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from string import Template

ROOT = Path(__file__).parents[2]
DEFAULT = ROOT / "config" / "local.env"
KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")

DEFAULTS = (
    ("HOST_NAME", None),
    ("ROOT", "$HOME/bdbv-rounds"),
    ("MAILBOX", "$HOME/bdbv-mailbox.md"),
    ("INBOX", ""),
    # The workflow's own settings.
    ("RUN", "$ROOT/checkpoints/slots"),
    ("SCORE_CACHE", "$ROOT/scores"),
    # The slots check's no-slots run and German blocks.
    ("NOSLOTS_RUN", "$ROOT/checkpoints"),
    ("GERMANY_BLOCKS", "$ROOT/germany/germany.pkl"),
    ("JOB_PREFIX", ""),
    # Mailbox lines (any value turns them on), publishing as it trains,
    # the targets `all` runs here, and where batches come from (empty:
    # workflow/config.yaml's `finetune.sims_from`).
    ("NOTIFY", ""),
    ("PUBLISH", ""),
    ("TARGETS", "simulate finetune score"),
    ("SIMS_FROM", ""),
    # Dashboard runs with a local folder (flowlib.dashboard.runs specs,
    # separated by `;`).
    ("DASHBOARD_RUNS", ""),
    # Memory and disk a job leaves free when it starts (GB).
    ("MIN_FREE_GB", "20"),
    ("MIN_DISK_GB", "16"),
    # Logs of the paper's detached jobs (paper/scripts).
    ("PAPER_LOGS", "$HOME/paper-jobs"),
)
"""Each setting and its default, in order, as later ones use earlier."""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _expand(value: str, names: Mapping[str, str], quoted: bool) -> str:
    # Single quotes keep a value as written, as the shell does.
    if quoted:
        return value
    return Template(value).safe_substitute(names)


def read(
    path: Path | str | None = None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the machine settings.

    Parameters
    ----------
    path
        The local config; by default `$LOCAL_CONFIG`, else
        `config/local.env`.
    environ
        The environment; by default `os.environ`.

    Returns
    -------
    dict
        Every key of the file and of `DEFAULTS`.
    """
    env = dict(os.environ if environ is None else environ)
    if path is None:
        path = env.get("LOCAL_CONFIG") or DEFAULT
    names = dict(env)
    out: dict[str, str] = {}
    path = Path(path)
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, sep, raw = line.partition("=")
            if not sep or not KEY.match(key):
                continue
            if key in env:
                out[key] = env[key]
                continue
            single = raw.startswith("'") and raw.endswith("'")
            value = _expand(_unquote(raw), names, single)
            names[key] = out[key] = value
    for key, default in DEFAULTS:
        if key in out:
            continue
        if key in env:
            value = env[key]
        elif default is None:
            value = socket.gethostname().split(".")[0]
        else:
            value = _expand(default, names, quoted=False)
        names[key] = out[key] = value
    return out


def expand(value: str, settings: Mapping[str, str] | None = None) -> str:
    """Return `value` with `$NAME` or `${NAME}` filled from the settings.

    Parameters
    ----------
    value
        A path or other value, such as `$ROOT/rounds/r1/sims`.
    settings
        The machine settings; by default `read()`.
    """
    names = read() if settings is None else settings
    return Template(value).safe_substitute(names)


if __name__ == "__main__":
    import sys

    cfg = read()
    for key in sys.argv[1:]:
        print(cfg.get(key, ""))
