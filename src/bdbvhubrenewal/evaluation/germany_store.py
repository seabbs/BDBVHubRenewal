# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Stored German forecasts of the fine-tuned Chronos-2 model.

The Pages build never runs `STORED_MODELS` on the German benchmark. The
checkpoint scorer (`scripts/checkpoint_scores.py germany` on the
`checkpoint-scoring` branch) already nowcasts every development date
with each published checkpoint, by `germany_forecasts` and `with_truth`
as `bdbvhubrenewal germany-backtest` does. `scripts/germany_ft_store.py`
writes a checkpoint's forecasts from its cache to the `STORE_BRANCH`
branch, one directory per release tag (`write_stored`), and the Pages
build loads the one `choose_checkpoint` picks (`bdbvhubrenewal
germany-ft-load`), so it is scored and shown like every other model.

Only development dates, nowcast dates before the split date, are ever
stored or loaded (`development_rows`).
"""

from __future__ import annotations

import gzip
import io
import json
import re
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import pandas as pd

__all__ = [
    "FT_PREFIX",
    "SLOTS_PREFIX",
    "STORED_MODELS",
    "STORE_BRANCH",
    "choose_checkpoint",
    "development_rows",
    "load_stored",
    "release_tag",
    "stored_tags",
    "write_stored",
]

STORED_MODELS: Final = (
    "fm_chronos_daily_ragged_ft",
    "fm_chronos_weekly_ragged_ft",
)
"""The models whose German forecasts come from the store, both at the
same checkpoint."""

STORE_BRANCH: Final = "germany-ft-forecasts"
"""The branch that holds the stored forecasts."""

FT_PREFIX: Final = "chronos-ft"
"""Tag prefix of the fine-tuned checkpoints (`foundation.FINETUNED_PREFIX`)."""

SLOTS_PREFIX: Final = "chronos-slots"
"""Tag prefix of the final run's checkpoints, labelled `slots-<hash>`."""

FORECASTS: Final = "forecasts.csv.gz"
MANIFEST: Final = "manifest.json"

_HASH = re.compile(r"[0-9a-f]{6,64}")


def release_tag(label: str) -> str:
    """Return the release tag of a checkpoint scorer label.

    Parameters
    ----------
    label : str
        A settings hash (a `chronos-ft-<hash>` release) or `slots-<hash>`
        (a `chronos-slots-<hash>` release).

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the label is neither.
    """
    slots = label.startswith("slots-")
    checkpoint = label.removeprefix("slots-")
    if not _HASH.fullmatch(checkpoint):
        raise ValueError(
            f"Not a checkpoint label: {label!r}. Give a settings hash or "
            "slots-<hash>."
        )
    return f"{SLOTS_PREFIX if slots else FT_PREFIX}-{checkpoint}"


def _published(releases: Iterable[Mapping[str, Any]], prefix: str) -> list[str]:
    """Return the published tags with a prefix, oldest first."""
    rows = sorted(
        (str(r["published_at"]), str(r["tag_name"]))
        for r in releases
        if str(r["tag_name"]).startswith(f"{prefix}-")
        and not r.get("draft")
        and r.get("published_at")
    )
    return [tag for _, tag in rows]


def choose_checkpoint(
    stored: Collection[str],
    releases: Iterable[Mapping[str, Any]],
    spec: str,
    live: Collection[str] | None = None,
) -> tuple[str | None, str | None]:
    """Return the checkpoint whose stored forecasts the page shows.

    The newest published `SLOTS_PREFIX` release with stored forecasts
    wins: the final run's checkpoint, once the scorer has stored it.
    Otherwise the checkpoint `spec` pins, as the `_ft` models load it
    (`foundation.resolve_checkpoint`): the newest `FT_PREFIX` release for
    `"latest"`, else that hash's release. If its forecasts are not stored
    yet, the newest stored `FT_PREFIX` release is shown instead.

    Parameters
    ----------
    stored : collection of str
        Release tags with stored forecasts (`stored_tags`).
    releases : iterable of mapping
        The repository's releases, each with `tag_name`, `published_at`
        and `draft` (`foundation.finetuned_releases`).
    spec : str
        `hash` of config/finetune.toml: `"latest"` or a settings hash.
    live : collection of str, optional
        The live fine-tuning run's release tags
        (`live_checkpoint.live_tags`). When given, only its newest stored
        checkpoint is shown, and none while it has none stored.

    Returns
    -------
    tuple
        The tag shown, None if no candidate is stored, and the tag `spec`
        pins, None if there is no such release.
    """
    releases = list(releases)
    slots = [t for t in _published(releases, SLOTS_PREFIX) if t in stored]
    if live is not None:
        slots = [t for t in slots if t in live]
    ft = _published(releases, FT_PREFIX)
    if spec == "latest":
        pinned = ft[-1] if ft else None
    else:
        pinned = f"{FT_PREFIX}-{spec}"
    if slots:
        return slots[-1], pinned
    if live is not None:
        return None, pinned
    if pinned in stored:
        return pinned, pinned
    fallback = [t for t in ft if t in stored]
    return (fallback[-1] if fallback else None), pinned


def development_rows(
    forecasts: pd.DataFrame,
    split_date: str,
    models: tuple[str, ...] = STORED_MODELS,
) -> pd.DataFrame:
    """Return the models' forecasts, refusing any held-out date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `germany_forecasts` output with truth (`with_truth`), of any
        models.
    split_date : str
        The first held-out nowcast date (`GermanyConfig.split_date`).
    models : tuple of str
        The models kept.

    Returns
    -------
    pandas.DataFrame
        The models' rows, sorted.

    Raises
    ------
    ValueError
        If a row of the models has a nowcast date or a target end date on
        or after the split date, or there are none.
    """
    rows = forecasts.loc[forecasts["model"].isin(models)].copy()
    for col in ("origin", "target_end_date", "truth_vintage"):
        rows[col] = rows[col].astype(str)
    late = (rows["origin"] >= split_date) | (
        rows["target_end_date"] >= split_date
    )
    if late.any():
        first = rows.loc[late, "origin"].min()
        raise ValueError(
            f"{int(late.sum())} rows of {', '.join(models)} are held-out "
            f"(on or after {split_date}, from {first}); only development "
            "dates may be "
            "stored or shown."
        )
    if rows.empty:
        raise ValueError(f"No forecasts of {', '.join(models)}.")
    keys = [
        "model",
        "origin",
        "location",
        "target_end_date",
        "horizon",
        "quantile",
    ]
    return rows.sort_values(keys, kind="stable").reset_index(drop=True)


def _csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV with no timestamp.

    The same rows give the same bytes, so an unchanged checkpoint commits
    nothing.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False).encode())
    path.write_bytes(buf.getvalue())


def write_stored(
    store: Path | str,
    tag: str,
    forecasts: pd.DataFrame,
    split_date: str,
    label: str,
    source: str = "",
) -> Path:
    """Write a checkpoint's development forecasts into a store directory.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root, as the `STORE_BRANCH` branch's tree.
    tag : str
        The checkpoint's release tag, the directory written.
    forecasts : pandas.DataFrame
        Its forecasts (`development_rows` keeps `STORED_MODELS`).
    split_date : str
        The first held-out nowcast date.
    label : str
        The checkpoint scorer's label, for the manifest.
    source : str
        Where the forecasts came from, for the manifest.

    Returns
    -------
    pathlib.Path
        The directory written: `forecasts.csv.gz` and `manifest.json`.
    """
    rows = development_rows(forecasts, split_date)
    out = Path(store) / tag
    out.mkdir(parents=True, exist_ok=True)
    _csv_gz(rows, out / FORECASTS)
    manifest = {
        "tag": tag,
        "label": label,
        "models": sorted(set(rows["model"])),
        "split_date": split_date,
        "first_origin": str(rows["origin"].min()),
        "last_origin": str(rows["origin"].max()),
        "origins": int(rows["origin"].nunique()),
        "rows": len(rows),
        "source": source,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def stored_tags(store: Path | str) -> set[str]:
    """Return the release tags with stored forecasts.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root; missing means none.

    Returns
    -------
    set of str
    """
    root = Path(store)
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if (p / FORECASTS).exists()}


def load_stored(store: Path | str, tag: str, split_date: str) -> pd.DataFrame:
    """Read a checkpoint's stored forecasts, refusing any held-out date.

    Parameters
    ----------
    store : pathlib.Path or str
        The store's root.
    tag : str
        The release tag.
    split_date : str
        The first held-out nowcast date.

    Returns
    -------
    pandas.DataFrame
        `germany-backtest`'s forecast columns, `STORED_MODELS` only.
    """
    dtype = {"origin": str, "truth_vintage": str, "target_end_date": str}
    frame = pd.read_csv(Path(store) / tag / FORECASTS, dtype=dtype)
    return development_rows(frame, split_date)
