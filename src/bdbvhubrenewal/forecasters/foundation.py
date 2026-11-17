# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The Chronos-2 foundation model on the hub target's releases.

See the Foundation models methods page for the model.
Each pair of a source (`SOURCES`) and a layout (`LAYOUTS`) is one
registered model, `fm_chronos_<source>_<layout>`, and each source's
ragged model with a fine-tuned checkpoint (`resolve_checkpoint`, the
latest release unless `config/finetune.toml` pins one) is
`fm_chronos_<source>_ragged_ft`. For the SI, `fm_chronos_<source>_ragged_real`
is the ragged model with the checkpoint fine-tuned on real data alone
(the German development half and the public BDBV data to date,
`scripts/finetune_realdata.py`), pinned in `config/finetune.toml` and
released as `realdata-ft-<hash>`, which the latest lookup never takes.
`hub_triangle` lays out the hub target by place, time and release lag,
`FoundationModel` gives it to Chronos-2 as one group of series, and
`chronos_sampler` draws trajectories from Chronos-2's quantiles.
Chronos-2 is an optional dependency group (`uv sync --group chronos`).
Without it, or without the fine-tuned checkpoint's release, a fit raises
and the fit store stores it as failed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import time
import tomllib
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache, partial
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.zones import province_of, weekly_zone_counts
from bdbvhubrenewal.forecasters import series_bias
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from bdbvhubrenewal.forecasters.chain import start_weeks
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk
from bdbvhubrenewal.forecasters.zones import reconcile_zones

__all__ = [
    "ADAPTER_FILES",
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "FINETUNED_CACHE",
    "FINETUNED_CONFIG",
    "FINETUNED_ENV",
    "FINETUNED_PREFIX",
    "FINETUNED_RELEASES",
    "FINETUNED_URL",
    "LAYOUTS",
    "MAX_LAG",
    "REALDATA_PREFIX",
    "SLOTS_PREFIX",
    "SOURCES",
    "SUMS_FILE",
    "Checkpoint",
    "FoundationModel",
    "HubTriangle",
    "Sampler",
    "chronos_sampler",
    "finetuned_adapter",
    "finetuned_releases",
    "finetuned_spec",
    "hub_triangle",
    "layout_block",
    "native_sampler",
    "quantile_draws",
    "realdata_pinned",
    "realdata_spec",
    "register_realdata_models",
    "release_tag",
    "resolve_checkpoint",
]

SOURCES: Final = ("weekly", "daily")
"""The hub target's releases by onset week (`target.csv`) or onset day
(`daily.csv`)."""

LAYOUTS: Final = ("complete", "ragged")
"""The ways the releases are given to Chronos-2."""

MAX_LAG: Final = {"weekly": 4, "daily": 28}
"""Release lag $D$, in the source's steps, from which a count is taken as
final."""

CHRONOS_MODEL: Final = "amazon/chronos-2"
"""The Hugging Face checkpoint."""

CHRONOS_REVISION: Final = "29ec3766d36d6f73f0696f85560a422f50e8498c"
"""The checkpoint's commit, pinned so runs are reproducible."""

FINETUNED_CONFIG: Final = Path("config/finetune.toml")
"""Which fine-tuned checkpoint the `_ft` models load: `hash`, `"latest"`
or a settings hash."""

FINETUNED_URL: Final = (
    "https://github.com/seabbs/BDBVHubRenewal/releases/download"
)
"""Where the fine-tuned checkpoints' releases are downloaded from."""

FINETUNED_RELEASES: Final = (
    "https://api.github.com/repos/seabbs/BDBVHubRenewal/releases"
)
"""The repository's releases, listed to find the latest checkpoint."""

FINETUNED_PREFIX: Final = "chronos-ft"
"""Tag prefix of the simulation-trained checkpoints' releases, the only
ones `"latest"` takes."""

REALDATA_PREFIX: Final = "realdata-ft"
"""Tag prefix of the real-data checkpoints' releases."""

FINETUNED_ENV: Final = "BDBV_CHRONOS_FT"
"""Environment variable with the settings hash a run resolved once, which
then overrides `FINETUNED_CONFIG`, so its jobs never list the releases."""

SUMS_FILE: Final = "sha256sums.txt"
"""A checkpoint release's asset with the SHA-256 of its other assets."""

RETRY_SECONDS: Final = (2, 10, 30)
"""Waits before each retry of a GitHub request that failed in passing."""

ADAPTER_FILES: Final = ("adapter_config.json", "adapter_model.safetensors")
"""The LoRA adapter's files. A checkpoint's release holds them and its
`settings.json`."""

FINETUNED_CACHE: Final = "~/.cache/bdbvhubrenewal"
"""Directory each downloaded checkpoint is kept in, as
`chronos-ft-<hash>`."""

SLOTS_PREFIX: Final = "slots-"
"""Prefix of the spec of a checkpoint with the series bias
(`forecasters.series_bias`). Such a checkpoint is released as
`chronos-slots-<hash>`, the paper's name for the run, so the `latest`
lookup, which takes only `chronos-ft-*` releases, never gives it to code
without the bias."""


def release_tag(spec: str) -> str:
    """Return the release tag of a checkpoint.

    `chronos-ft-<hash>`, or `chronos-slots-<hash>` for `slots-<hash>`.
    """
    if spec.startswith(SLOTS_PREFIX):
        return f"chronos-{spec}"
    return f"chronos-ft-{spec}"


Series = Mapping[str, Any]
"""One group for Chronos-2: `target` `[V, S]`, and optionally
`past_covariates` (name to `[S]`) and `future_covariates` (name to
`[H]`, NaN where unknown)."""


@dataclass(frozen=True)
class Checkpoint:
    """A fine-tuned checkpoint of Chronos-2, from its release.

    Attributes
    ----------
    hash : str
        Settings hash from `scripts/finetune_stream.py`. The release
        is `tag`: `release_tag(hash)` for a simulation checkpoint, else
        `<prefix>-<hash>`.
    sha256 : tuple of (str, str)
        SHA-256 of each of `ADAPTER_FILES`, by file name.
    prefix : str
        The release's tag prefix, `FINETUNED_PREFIX` or `REALDATA_PREFIX`.
        Left out of the repr, which keys stored fits; the hash tells the
        checkpoints apart.
    """

    hash: str
    sha256: tuple[tuple[str, str], ...]
    prefix: str = field(default=FINETUNED_PREFIX, repr=False)

    @property
    def tag(self) -> str:
        """The checkpoint's release tag.

        A simulation checkpoint's is `release_tag(hash)`, so `slots-<hash>`
        is `chronos-slots-<hash>`.
        """
        if self.prefix == FINETUNED_PREFIX:
            return release_tag(self.hash)
        return f"{self.prefix}-{self.hash}"


def finetuned_spec(path: Path | str = FINETUNED_CONFIG) -> str:
    """Return the checkpoint the `_ft` models load: `"latest"` or a hash.

    `FINETUNED_ENV`, when set, overrides the file.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `hash`.

    Returns
    -------
    str
    """
    if os.environ.get(FINETUNED_ENV):
        return os.environ[FINETUNED_ENV]
    return str(tomllib.loads(Path(path).read_text())["hash"])


def _passing(err: BaseException) -> bool:
    """Whether a failed request may succeed if tried again."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code >= 500 or err.code == 429
    if isinstance(err, urllib.error.URLError):
        return not isinstance(err.reason, FileNotFoundError)
    return isinstance(err, (OSError, http.client.HTTPException))


def _missing(err: BaseException) -> bool:
    """Whether a failed request means the file is not there (a 404)."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code == 404
    reason = getattr(err, "reason", None)
    return isinstance(err, FileNotFoundError) or isinstance(
        reason, FileNotFoundError
    )


def _unreachable(url: str, err: BaseException) -> ConnectionError:
    """Return the error for a request refused or dropped, not missing."""
    return ConnectionError(
        f"Cannot read {url}: {err!r}. If GitHub is rate limiting, set "
        "GH_TOKEN or GITHUB_TOKEN to a GitHub token."
    )


def _get(url: str) -> bytes:
    """Read a URL, with a GitHub token from the environment if there is one.

    A server error or a dropped connection is tried again after each of
    `RETRY_SECONDS`.
    """
    request = urllib.request.Request(url)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    for wait in (*RETRY_SECONDS, None):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return bytes(response.read())
        except (OSError, http.client.HTTPException) as err:
            if wait is None or not _passing(err):
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def finetuned_releases() -> list[dict[str, Any]]:
    """List the repository's releases from the GitHub API.

    Returns
    -------
    list of dict
        Each release's `tag_name`, `published_at` and `draft`, among
        others.

    Raises
    ------
    ConnectionError
        If GitHub cannot be reached.
    """
    out: list[dict[str, Any]] = []
    for page in range(1, 11):
        url = f"{FINETUNED_RELEASES}?per_page=100&page={page}"
        try:
            batch = json.loads(_get(url))
        except (OSError, http.client.HTTPException) as err:
            raise ConnectionError(
                f"Cannot list the releases at {url} to find the fine-tuned "
                f"checkpoint: {err!r}. Set GH_TOKEN or GITHUB_TOKEN to a "
                "GitHub token (anonymous requests are rate limited), or "
                f"set {FINETUNED_ENV} to a settings hash."
            ) from err
        out += batch
        if len(batch) < 100:
            break
    return out


def realdata_spec(path: Path | str = FINETUNED_CONFIG) -> str:
    """Return the settings hash the `_real` models load.

    No environment variable overrides it, so a run that pins the
    simulation-trained checkpoint leaves it as it is.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `realdata`.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the pin is pending (empty) until the release is published.
    """
    spec = str(tomllib.loads(Path(path).read_text())["realdata"])
    if not spec:
        raise ValueError(
            f"No {REALDATA_PREFIX} release is pinned yet: set `realdata` in "
            f"{path} to its settings hash."
        )
    return spec


@cache
def resolve_checkpoint(spec: str, prefix: str = FINETUNED_PREFIX) -> Checkpoint:
    """Return a fine-tuned checkpoint from its release.

    `"latest"` is the newest published `chronos-ft-*` release by publish
    date; releases of other prefixes, such as `realdata-ft-*`, are never
    taken. The release's `SUMS_FILE` gives the SHA-256 of its adapter
    files. Each process resolves once.

    Parameters
    ----------
    spec : str
        `"latest"` or a settings hash.
    prefix : str
        The release's tag prefix. `"latest"` needs `FINETUNED_PREFIX`.

    Returns
    -------
    Checkpoint

    Raises
    ------
    ConnectionError
        If GitHub cannot be reached. No cached checkpoint is used instead.
    LookupError
        If there is no checkpoint release.
    FileNotFoundError
        If the release has no `SUMS_FILE`.
    ValueError
        If `SUMS_FILE` lacks an adapter file.
    """
    checkpoint = spec
    if spec == "latest" and prefix != FINETUNED_PREFIX:
        raise ValueError(f"Only {FINETUNED_PREFIX} releases resolve latest.")
    if spec == "latest":
        tags = sorted(
            (r["published_at"], r["tag_name"])
            for r in finetuned_releases()
            if str(r["tag_name"]).startswith(f"{FINETUNED_PREFIX}-")
            and not r.get("draft")
            and r.get("published_at")
        )
        if not tags:
            raise LookupError("No chronos-ft-* release holds a checkpoint.")
        checkpoint = tags[-1][1].removeprefix(f"{FINETUNED_PREFIX}-")
    tag = (
        release_tag(checkpoint)
        if prefix == FINETUNED_PREFIX
        else f"{prefix}-{checkpoint}"
    )
    url = f"{FINETUNED_URL}/{tag}/{SUMS_FILE}"
    try:
        text = _get(url).decode()
    except (OSError, http.client.HTTPException) as err:
        if _missing(err):
            raise FileNotFoundError(
                f"The release {tag} has no {SUMS_FILE} ({url})."
            ) from err
        raise _unreachable(url, err) from err
    sums = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        # A line that is not `<sha> <name>` is left to the check below.
        if len(parts) == 2:
            sums[parts[1].lstrip("*").strip()] = parts[0]
    missing = [f for f in ADAPTER_FILES if f not in sums]
    if missing:
        raise ValueError(
            f"{SUMS_FILE} of the release {tag} has no SHA-256 of "
            f"{', '.join(missing)}."
        )
    return Checkpoint(
        checkpoint, tuple((f, sums[f]) for f in ADAPTER_FILES), prefix
    )


def _finetuned(name: str, source: str) -> FoundationModel:
    """Build a fine-tuned model with the checkpoint `finetuned_spec` names.

    The weekly and daily models share the checkpoint; the daily one reads
    its weekly forecast from per-place weekly total rows, as trained.
    """
    return FoundationModel(
        name,
        source,
        "ragged",
        resolve_checkpoint(finetuned_spec()),
        weekly_totals=source == "daily",
    )


def _real_data(name: str, source: str) -> FoundationModel:
    """Build a model with the real-data checkpoint `realdata_spec`."""
    return FoundationModel(
        name,
        source,
        "ragged",
        resolve_checkpoint(realdata_spec(), REALDATA_PREFIX),
    )


Sampler = Callable[[Sequence[Series], int, int, int], list[np.ndarray]]
"""Draws trajectories: `(groups, horizon, n, seed)` to one
`[n, V, horizon]` array per group."""


@dataclass(frozen=True)
class HubTriangle:
    """The hub target by place, time step and release lag.

    Attributes
    ----------
    times : list of str
        Consecutive weeks (ISO Saturdays) or days.
    final : numpy.ndarray
        The count of each place and step at the latest release, `[L, T]`.
    partial : numpy.ndarray
        The count of each place and step at release lag $d < D$, from the
        release whose cut-off is $d$ steps after it, NaN where no release
        is, `[L, T, D]`.
    last : int
        Index of the step holding the latest release's cut-off.
    """

    times: list[str]
    final: np.ndarray
    partial: np.ndarray
    last: int

    @property
    def n_context(self) -> int:
        """Steps whose count at the latest release is at lag $D$ or more."""
        return self.last - int(self.partial.shape[2]) + 1


def _step(source: str, day: str) -> dt.date:
    """Return the step holding a day: its week's Saturday, or the day."""
    return epiweek_end(day) if source == "weekly" else as_date(day)


@dataclass(frozen=True)
class PeriodRows:
    """Each place's period totals on a daily grid, for its total row.

    Attributes
    ----------
    ends : numpy.ndarray
        Whether each day of the grid ends a whole period, `[T]`.
    totals : numpy.ndarray
        Each place's total over the period ending on each such day, NaN on
        every other day, `[L, T]`.
    """

    ends: np.ndarray
    totals: np.ndarray


def week_rows(tri: HubTriangle) -> PeriodRows:
    """Return each place's weekly totals of a daily triangle's counts.

    Each week (Sunday to Saturday) whose seven days are all on the grid is
    summed from the daily counts at the latest release and put on its
    Saturday; a week the grid cuts has no total.
    """
    days = [as_date(t) for t in tri.times]
    ends = np.array(
        [epiweek_end(d.isoformat()) == d and i >= 6 for i, d in enumerate(days)]
    )
    totals = np.full(tri.final.shape, np.nan)
    for t in np.flatnonzero(ends):
        totals[:, t] = tri.final[:, t - 6 : t + 1].sum(axis=1)
    return PeriodRows(ends, totals)


def hub_triangle(
    rows: pd.DataFrame,
    source: str,
    places: Sequence[str],
    last_time: str,
    max_lag: int | None = None,
) -> HubTriangle:
    """Lay out the hub target of every release by place, step and lag.

    Parameters
    ----------
    rows : pandas.DataFrame
        The releases. The latest gives the final counts. Weekly rows have
        `target_end_date` and `total`; daily rows have `onset_date`,
        `observed` and `imputed`. Both have `vintage`, `location` and
        `cutoff`.
    source : str
        One of `SOURCES`.
    places : sequence of str
        Places, in the order of the output.
    last_time : str
        Last step of the grid.
    max_lag : int, optional
        Release lag $D$, `MAX_LAG` of the source by default.

    Returns
    -------
    HubTriangle
        Steps from the first in `rows` to `last_time`.
    """
    width = 7 if source == "weekly" else 1
    time_col = "target_end_date" if source == "weekly" else "onset_date"
    rows = rows.loc[rows["location"].isin(places)]
    if rows.empty:
        raise ValueError(f"no release rows for {list(places)}")
    value = (
        rows["total"]
        if source == "weekly"
        else rows["observed"] + rows["imputed"]
    )
    first = _step(source, str(rows[time_col].min()))
    n_times = (as_date(last_time) - first).days // width + 1
    times = [
        (first + dt.timedelta(days=width * i)).isoformat()
        for i in range(n_times)
    ]
    place = rows["location"].map({c: i for i, c in enumerate(places)})
    days = (pd.to_datetime(rows[time_col]) - pd.Timestamp(first)).dt.days
    step = days.to_numpy() // width
    lags = MAX_LAG[source] if max_lag is None else max_lag
    partial = np.full((len(places), n_times, lags), np.nan)
    final = np.zeros((len(places), n_times))
    last = 0
    for release in sorted(set(rows["vintage"])):
        mine = (rows["vintage"] == release).to_numpy()
        cutoff = str(rows.loc[mine, "cutoff"].iloc[0])
        at = (_step(source, cutoff) - first).days // width
        counts = np.zeros((len(places), n_times))
        keep = mine & (step >= 0) & (step < n_times)
        np.add.at(
            counts,
            (place.to_numpy()[keep].astype(int), step[keep]),
            value.to_numpy()[keep],
        )
        for d in range(lags):
            if 0 <= at - d < n_times:
                partial[:, at - d, d] = counts[:, at - d]
        # Sorted, so the last release's counts and cut-off are the latest.
        final, last = counts, at
    return HubTriangle(times, final, partial, last)


def quantile_draws(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read quantile functions at uniform draws.

    Linear between the levels and flat beyond them.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    values : numpy.ndarray
        Quantiles at those levels, `[..., Q]`. They are sorted first, so
        crossed quantiles give a valid function.
    u : numpy.ndarray
        Uniform draws, the shape of `values` less its last axis.

    Returns
    -------
    numpy.ndarray
        The draws, the shape of `u`.
    """
    values = np.sort(values, axis=-1)
    hi = np.clip(np.searchsorted(levels, u), 1, len(levels) - 1)
    lo = hi - 1
    weight = np.clip((u - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
    at_lo = np.take_along_axis(values, lo[..., None], axis=-1)[..., 0]
    at_hi = np.take_along_axis(values, hi[..., None], axis=-1)[..., 0]
    return np.asarray(at_lo + weight * (at_hi - at_lo))


def finetuned_adapter(checkpoint: Checkpoint) -> Path:
    """Return a fine-tuned checkpoint's LoRA adapter, downloading it once.

    The release `<prefix>-<hash>` at `FINETUNED_URL` holds
    `ADAPTER_FILES` and `settings.json`. They are kept in
    `FINETUNED_CACHE` only once all are downloaded, the settings hash
    matches and each adapter file has its SHA-256. A cached copy is
    checked against the SHA-256 again on each use.

    Parameters
    ----------
    checkpoint : Checkpoint
        The checkpoint.

    Returns
    -------
    pathlib.Path
        The adapter directory.

    Raises
    ------
    FileNotFoundError
        If the release has no such asset.
    ConnectionError
        If GitHub refuses the request or the connection fails.
    ValueError
        If the release's settings hash or an adapter file's SHA-256 is not
        the checkpoint's.
    """
    tag = checkpoint.tag
    cache = Path(FINETUNED_CACHE).expanduser()
    out = cache / tag
    if out.is_dir():
        if all(
            (out / name).is_file()
            and hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
            for name, sha in checkpoint.sha256
        ):
            return out
        # A cached copy that has changed is downloaded again.
        shutil.rmtree(out)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=cache, prefix=f".{tag}-"))
    try:
        for name in (*ADAPTER_FILES, "settings.json"):
            url = f"{FINETUNED_URL}/{tag}/{name}"
            try:
                (tmp / name).write_bytes(_get(url))
            except (OSError, http.client.HTTPException) as err:
                if _missing(err):
                    raise FileNotFoundError(
                        f"The release {tag} has no {name} ({url})."
                    ) from err
                raise _unreachable(url, err) from err
        found = json.loads((tmp / "settings.json").read_text()).get("hash")
        if found != checkpoint.hash.removeprefix(SLOTS_PREFIX):
            raise ValueError(
                f"The release {tag} holds the checkpoint with settings "
                f"hash {found}."
            )
        for name, expected in checkpoint.sha256:
            sha = hashlib.sha256((tmp / name).read_bytes()).hexdigest()
            if sha != expected:
                raise ValueError(
                    f"{name} of the release {tag} has SHA-256 {sha}, not "
                    f"{expected}."
                )
        try:
            tmp.rename(out)
        except OSError:
            # Another process cached it first.
            if not out.is_dir():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


@cache
def _chronos(checkpoint: Checkpoint | None = None) -> Any:
    adapter = None if checkpoint is None else finetuned_adapter(checkpoint)
    import chronos

    from bdbvhubrenewal import torchdevice as devices

    devices.apply_threads()
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL,
        revision=CHRONOS_REVISION,
        device_map=devices.pick_device(),
    )
    if adapter is None:
        return pipeline
    # Slots, if the adapter holds them, load with it.
    return chronos.Chronos2Pipeline(
        model=series_bias.load_adapter(pipeline.model, adapter)
    )


CHRONOS_KEYS: Final = ("target", "past_covariates", "future_covariates")
"""The parts of a group Chronos-2 reads; the rest name its rows' places."""


def _chronos_input(group: Series) -> dict[str, Any]:
    """Return the parts of a group Chronos-2 reads (`CHRONOS_KEYS`)."""
    return {k: group[k] for k in CHRONOS_KEYS if k in group}


def _group_rows(group: Series) -> np.ndarray:
    """Return each of a group's rows' place and parent, `[2, rows]`.

    In Chronos-2's row order: the targets, then the covariates by sorted
    name, past-only ones before known-future ones. Places are numbered in
    the group's `places` order; a target row's place is its
    `target_places` entry, one row per place by default.

    Raises
    ------
    ValueError
        If a covariate names no place, or the target rows do not match
        their places.
    """
    past = list(group.get("past_covariates", {}))
    known = set(group.get("future_covariates", {}))
    n_targets = np.atleast_2d(group["target"]).shape[0]
    named = dict.fromkeys(n.rsplit("_lag_", 1)[0] for n in past)
    places = list(group.get("places") or named)
    target_places = list(group.get("target_places", range(len(places))))
    if len(target_places) != n_targets:
        raise ValueError(
            f"{len(target_places)} target places for {n_targets} targets"
        )
    rows = sorted(n for n in past if n not in known)
    rows += sorted(n for n in past if n in known)
    at = {p: i for i, p in enumerate(places)}
    try:
        owner = [at[n.rsplit("_lag_", 1)[0]] for n in rows]
    except KeyError as err:
        raise ValueError(f"covariate of no target place: {err}") from err
    place = np.concatenate([target_places, owner]).astype(int)
    parent = np.asarray(series_bias.parents_of(places), dtype=int)
    return np.stack([place, parent[place]])


def _predict(
    pipeline: Any, groups: Sequence[Series], horizon: int, levels: np.ndarray
) -> list[Any]:
    """Return Chronos-2's quantiles at `levels` for each group.

    With the series bias each group is predicted on its own, so its rows
    are one batch, with its rows' places and parents (`_group_rows`), as
    in held-out losses.
    """
    model = getattr(pipeline, "model", None)
    if model is not None and series_bias.has_bias(model):
        # One group a batch, its rows' places and parents by name.
        biased = []
        for g in groups:
            with series_bias.assigned(model, _group_rows(g)):
                q, _ = pipeline.predict_quantiles(
                    [_chronos_input(g)],
                    prediction_length=horizon,
                    quantile_levels=levels.tolist(),
                )
            biased.extend(q)
        return biased
    quantiles, _ = pipeline.predict_quantiles(
        [_chronos_input(g) for g in groups],
        prediction_length=horizon,
        quantile_levels=levels.tolist(),
    )
    return list(quantiles)


def chronos_sampler(
    groups: Sequence[Series],
    horizon: int,
    n_samples: int,
    seed: int,
    checkpoint: Checkpoint | None = None,
) -> list[np.ndarray]:
    """Draw trajectories from Chronos-2's quantiles.

    Every series and step draws its own uniform level, so draws are
    independent across series and steps. Runs on a CUDA or Apple GPU when
    torch sees one.

    Parameters
    ----------
    groups : sequence of Series
        The groups of series.
    horizon : int
        Steps to draw.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the uniform draws.
    checkpoint : Checkpoint, optional
        The fine-tuned checkpoint (`finetuned_adapter`) to load into
        Chronos-2. The pretrained model by default.

    Returns
    -------
    list of numpy.ndarray
        Draws, `[n_samples, V, horizon]` per group.

    Raises
    ------
    ImportError
        If Chronos-2 is not installed (`uv sync --group chronos`).
    FileNotFoundError
        If the checkpoint's release has no such asset.
    ConnectionError
        If GitHub cannot be reached.
    ValueError
        If the release holds another checkpoint or adapter.
    """
    pipeline = _chronos(checkpoint)
    levels = np.asarray(pipeline.quantiles, dtype=float)
    rng = np.random.default_rng(seed)
    quantiles = _predict(pipeline, groups, horizon, levels)
    out = []
    for q in quantiles:
        q = q.float().cpu().numpy()  # [V, horizon, Q]
        u = rng.uniform(size=(n_samples, *q.shape[:2]))
        out.append(
            quantile_draws(levels, np.broadcast_to(q, (n_samples, *q.shape)), u)
        )
    return out


def layout_block(
    tri: HubTriangle,
    layout: str,
    places: Sequence[str],
    start: int,
    period: PeriodRows | None = None,
) -> Series:
    """Return the one block of every place's series, split at `start`.

    Parameters
    ----------
    tri : HubTriangle
        The hub target by place, step and lag.
    layout : str
        One of `LAYOUTS`.
    places : sequence of str
        Places, in the order of `tri`.
    start : int
        Steps of context, the first `start` steps of `tri`.
    period : PeriodRows, optional
        Each place's period totals on `tri`'s grid. With them each place
        has a second target row, of its totals, after the count rows; it
        has no lags.

    Returns
    -------
    Series
        The targets' context, the place of each target row
        (`target_places`, indices into `places`), and for `ragged` each
        place's lag series as past and future covariates.
    """
    target = tri.final[:, :start]
    target_places = list(range(len(places)))
    if period is not None:
        target = np.concatenate([target, period.totals[:, :start]])
        target_places += target_places
    if layout == "complete":
        return {"target": target, "target_places": target_places}
    lags = tri.partial.shape[2]
    names = [f"{p}_lag_{d}" for p in places for d in range(lags)]
    cells = tri.partial.transpose(0, 2, 1).reshape(len(names), -1)
    return {
        "target": target,
        "past_covariates": dict(zip(names, cells[:, :start], strict=True)),
        "future_covariates": dict(zip(names, cells[:, start:], strict=True)),
        "places": list(places),
        "target_places": target_places,
    }


def _walk_start(task: ForecastTask) -> str:
    """Return the earliest week `randomwalk` starts its walk from.

    It is read from the delay alone, so it exists where the walk has no
    spread.
    """
    walk = RandomWalk()
    snap = walk.snapshots(task)
    return str(snap.week_ends[int(start_weeks(snap, walk.settings).min())])


def _weekly_from_totals(
    drawn: np.ndarray,
    period: PeriodRows,
    tri: HubTriangle,
    start: int,
    weeks: Sequence[str],
) -> np.ndarray:
    """Return weekly draws `[S, L, W]` read from the total rows.

    A week whose Saturday is past the context takes its total row's draw
    there; an earlier week takes its total summed from the daily counts.
    `drawn` is `[S, 2L, horizon]`, the count rows then the total rows.
    """
    n_places = period.totals.shape[0]
    out = []
    for week in weeks:
        t = tri.times.index(week)
        if t >= start:
            out.append(drawn[:, n_places:, t - start])
        else:
            total = np.nan_to_num(period.totals[:, t])
            out.append(np.broadcast_to(total, drawn[:, :n_places, 0].shape))
    return np.stack(out, axis=2)


DAILY_COLUMNS: Final = ("location", "date", "sample", "value")
"""Columns of the daily trajectories (`FoundationModel.forecast_both`)."""


def _daily_frame(
    drawn: np.ndarray,
    tri: HubTriangle,
    start: int,
    places: Sequence[str],
    weeks: Sequence[str],
) -> pd.DataFrame:
    """Return each day of the target weeks from the daily count rows.

    A day past the context takes its count row's draw, no lower than the
    count reported; an earlier day the count reported. `drawn` is
    `[S, V, horizon]`, the count rows first.
    """
    first = as_date(weeks[0]) - dt.timedelta(days=6)
    days = [t for t in tri.times if first <= as_date(t) <= as_date(weeks[-1])]
    n, n_loc = drawn.shape[0], len(places)
    cols = []
    for day in days:
        t = tri.times.index(day)
        final = np.broadcast_to(tri.final[:, t], (n, n_loc))
        cols.append(
            np.maximum(drawn[:, :n_loc, t - start], final)
            if t >= start
            else final
        )
    values = np.stack(cols, axis=2)  # [S, L, D]
    return pd.DataFrame(
        {
            "location": np.repeat(list(places), n * len(days)),
            "date": np.tile(days, n * n_loc),
            "sample": np.tile(np.repeat(np.arange(1, n + 1), len(days)), n_loc),
            "value": values.transpose(1, 0, 2).reshape(-1),
        }
    )[list(DAILY_COLUMNS)]


def _lags_of(group: Series, places: set[str]) -> Series:
    """Keep the lag series of `places` only.

    A place with no cases has lag series of zeros, which tell Chronos-2
    nothing and in the daily block would take most of its memory.
    """
    out = dict(group)
    for key in ("past_covariates", "future_covariates"):
        if key in out:
            out[key] = {
                name: values
                for name, values in out[key].items()
                if name.rsplit("_lag_", 1)[0] in places
            }
    return out


def _at(rows: pd.DataFrame | None, vintage: str) -> pd.DataFrame | None:
    """Return the rows of one release, or None."""
    return None if rows is None else rows.loc[rows["vintage"] == vintage]


def _rows(*tables: pd.DataFrame | None) -> pd.DataFrame | None:
    """Stack the tables that are there, None if none is."""
    there = [t for t in tables if t is not None and not t.empty]
    return pd.concat(there, ignore_index=True) if there else None


@dataclass
class FoundationModel:
    """Chronos-2 on one source and layout of the hub target's releases.

    Attributes
    ----------
    name : str
        Model name.
    source : str
        One of `SOURCES`.
    layout : str
        One of `LAYOUTS`.
    checkpoint : Checkpoint or None
        The fine-tuned checkpoint Chronos-2 loads, or None for the
        pretrained model.
    weekly_totals : bool
        For the daily source: give each place a row of its weekly totals,
        summed from its daily counts (`week_rows`), and read the weekly
        forecast from that row. As the fine-tuned model was trained.
    sample : Sampler or None
        Draws the trajectories, Chronos-2 with `checkpoint` by default.
    stream : str or None
        The task's random stream the draws are seeded from, the model's
        name by default. Models given one stream draw the same random
        numbers (`bdbvhubrenewal.evaluation.marginals.forecast_quantiles`).
    """

    name: str
    source: str
    layout: str
    checkpoint: Checkpoint | None = None
    weekly_totals: bool = False
    sample: Sampler | None = field(default=None, repr=False)
    stream: str | None = field(default=None, repr=False)
    reads_zones: ClassVar[bool] = True

    @property
    def needs_snapshots(self) -> bool:
        """Whether the model reads release snapshots: `complete` does."""
        return self.layout == "complete"

    def forecast_both(
        self, task: ForecastTask
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the weekly and the daily trajectories of one draw.

        The weekly ones are `forecast`'s. The daily ones are the daily
        source's count rows, each day of the target weeks, floored at the
        counts reported; empty for the weekly source.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `target_history` and
            `zone_history` hold the weekly releases, its `daily_history`
            and `zone_daily_history` the daily ones, and its `zone_codes`
            the zones.

        Returns
        -------
        tuple of pandas.DataFrame
            The weekly trajectories, columns `location`, `target_end_date`,
            `sample`, `value`, for the country, the provinces and their
            zones, each its own series, with the zones made to sum to the
            provinces (`forecasters.zones.reconcile_zones`); and the daily
            ones, columns `location`, `date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no releases of the source or fewer than two
            steps of context, or, for `complete`, no week `randomwalk`
            can start from.
        ImportError
            If Chronos-2 is not installed.
        FileNotFoundError
            If the fine-tuned checkpoint's release has no such asset.
        ConnectionError
            If GitHub cannot be reached.
        ValueError
            If the release holds another checkpoint or adapter.
        """
        weekly = self.source == "weekly"
        base = task.target_history if weekly else task.daily_history
        zone = task.zone_history if weekly else task.zone_daily_history
        if base is None or base.empty:
            raise ModelUnavailableError(
                f"{self.name} needs the {self.source} releases, and the task "
                f"for {task.vintage} has none."
            )
        # Only the releases of the country and provinces, so a release
        # gives every place's lag cells or none.
        if zone is not None:
            zone = zone.loc[zone["vintage"].isin(set(base["vintage"]))]
        has_zones = zone is not None and task.zone_daily_history is not None
        zones = tuple(
            z
            for z in task.zone_codes
            if has_zones and province_of(z) in task.locations
        )
        rows = _rows(base, zone if zones else None)
        assert rows is not None
        places = (NATIONAL, *task.locations, *zones)
        weeks = list(task.target_end_dates)
        tri = hub_triangle(rows, self.source, places, weeks[-1])
        start = tri.n_context
        if self.layout == "complete":
            walk = _walk_start(task)
            if walk not in tri.times:
                raise ModelUnavailableError(
                    f"{self.name} starts from the randomwalk week {walk}, "
                    f"which the task for {task.vintage} has no "
                    f"{self.source} release of."
                )
            start = tri.times.index(walk) + 1
        if start < 2:
            raise ModelUnavailableError(
                f"{self.name} needs two steps of context, and the task for "
                f"{task.vintage} has {max(start, 0)}."
            )
        horizon = len(tri.times) - start
        seed = int(task.rng(self.stream or self.name).integers(0, 2**31 - 1))
        period = week_rows(tri) if self.weekly_totals else None
        group = layout_block(tri, self.layout, places, start, period)
        if self.checkpoint is None:
            # Zero-shot drops the lags of places with no cases, which tell
            # Chronos-2 nothing. A fine-tuned model was trained with them
            # (all-zero rows), so it keeps them.
            cased = {places[i] for i in np.flatnonzero(tri.final.sum(axis=1))}
            group = _lags_of(group, cased)
        sample = self.sample or partial(
            chronos_sampler, checkpoint=self.checkpoint
        )
        draws = sample([group], horizon, task.n_samples, seed)[0]
        drawn = np.maximum(np.round(draws), 0.0)
        if period is not None:
            values = _weekly_from_totals(drawn, period, tri, start, weeks)
        else:
            final = tri.final
            steps = np.broadcast_to(
                final, (task.n_samples, *final.shape)
            ).copy()
            steps[:, :, start:] = np.maximum(
                drawn[:, : len(places)], final[:, start:]
            )
            # The hub's target is weekly: the daily source is summed to it.
            week_of = np.array(
                [_step("weekly", t).isoformat() for t in tri.times]
            )
            values = np.stack(
                [steps[:, :, week_of == w].sum(axis=2) for w in weeks], axis=2
            )
        latest = _rows(task.target, _at(task.zone_history, task.vintage))
        assert latest is not None
        reported = (
            latest.pivot_table(
                index="location",
                columns="target_end_date",
                values="total",
                aggfunc="sum",
            )
            .reindex(index=list(places), columns=weeks)
            .fillna(0.0)
            .to_numpy()
        )
        values = np.maximum(values, reported)  # [S, L, W]
        n, n_loc = task.n_samples, len(places)
        weekly_out = pd.DataFrame(
            {
                "location": np.repeat(list(places), n * len(weeks)),
                "target_end_date": np.tile(weeks, n * n_loc),
                "sample": np.tile(
                    np.repeat(np.arange(1, n + 1), len(weeks)), n_loc
                ),
                "value": values.transpose(1, 0, 2).reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]
        if zones:
            # The zones are made to sum to the provinces, as submitted.
            assert task.zone_daily_history is not None
            counts = weekly_zone_counts(task.zone_daily_history, task.vintage)
            own = weekly_out["location"].isin(set(zones))
            split = reconcile_zones(
                weekly_out.loc[own], weekly_out.loc[~own], task, counts, zones
            )
            weekly_out = pd.concat(
                [weekly_out.loc[~own], split], ignore_index=True
            )
        if self.source != "daily":
            return weekly_out, pd.DataFrame(columns=list(DAILY_COLUMNS))
        return weekly_out, _daily_frame(drawn, tri, start, places, weeks)

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return trajectories of every hub target in the task.

        The weekly part of `forecast_both`, which says what is raised.
        """
        return self.forecast_both(task)[0]

    def quantiles(
        self, task: ForecastTask, levels: Sequence[float]
    ) -> pd.DataFrame | None:
        """Return Chronos-2's own quantiles where each target is one step.

        Each week of the weekly source is one of Chronos-2's steps, so its
        quantiles are Chronos-2's native quantiles read at `levels` by
        linear interpolation between the native levels (`native_sampler`).
        They pass through `forecast` as one trajectory per level; every
        step after the draws (rounding, the floors at the reported counts)
        is monotone, so trajectory `k` is the quantile at level `k`.
        A week of the daily source is a sum of steps, which has no native
        quantile.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.
        levels : sequence of float
            Increasing quantile levels.

        Returns
        -------
        pandas.DataFrame or None
            Columns `location`, `target_end_date`, `quantile` and `value`,
            or None for the daily source.
        """
        if self.source != "weekly":
            return None
        at = np.asarray(levels, dtype=float)
        run = replace(self, sample=native_sampler(at, self.checkpoint))
        out = run.forecast(replace(task, n_samples=len(at)))
        out["quantile"] = at[out["sample"].to_numpy() - 1]
        return out[["location", "target_end_date", "quantile", "value"]]


def native_sampler(
    levels: np.ndarray, checkpoint: Checkpoint | None = None
) -> Sampler:
    """Return a sampler giving Chronos-2's quantiles at `levels` as paths.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[K]`.
    checkpoint : Checkpoint, optional
        The fine-tuned checkpoint, the pretrained model by default.

    Returns
    -------
    Sampler
        Gives `[K, V, horizon]` per group: path `k` is each series and
        step's quantile at `levels[k]`, linear between Chronos-2's native
        levels and flat beyond them. It draws no random numbers.
    """

    def sample(
        groups: Sequence[Series],
        horizon: int,
        n: int,  # noqa: ARG001
        seed: int,  # noqa: ARG001
    ) -> list[np.ndarray]:
        pipeline = _chronos(checkpoint)
        native = np.asarray(pipeline.quantiles, dtype=float)
        quantiles, _ = pipeline.predict_quantiles(
            [_chronos_input(g) for g in groups],
            prediction_length=horizon,
            quantile_levels=native.tolist(),
        )
        out = []
        for q in quantiles:
            q = np.sort(q.float().cpu().numpy(), axis=-1)  # [V, H, Q]
            at = np.apply_along_axis(
                lambda v: np.interp(levels, native, v), -1, q
            )  # [V, H, K]
            out.append(np.moveaxis(at, -1, 0))  # [K, V, H]
        return out

    return sample


for _source in SOURCES:
    for _layout in LAYOUTS:
        _name = f"fm_chronos_{_source}_{_layout}"
        register_model(
            _name,
            partial(FoundationModel, _name, _source, _layout),
            f"Chronos-2 on the hub target's {_source} releases, {_layout} "
            "layout, every target in one block.",
            fitted=True,
        )
    register_model(
        f"fm_chronos_{_source}_ragged_ft",
        partial(_finetuned, f"fm_chronos_{_source}_ragged_ft", _source),
        "Chronos-2 fine-tuned on simulated outbreaks, on the hub target's "
        f"{_source} releases, ragged layout.",
        fitted=True,
    )


def realdata_pinned(path: Path | str = FINETUNED_CONFIG) -> bool:
    """Whether `path` pins the real-data checkpoint.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `realdata`.

    Returns
    -------
    bool
    """
    return bool(tomllib.loads(Path(path).read_text()).get("realdata"))


def register_realdata_models() -> None:
    """Register the `_real` models, fine-tuned on real data alone.

    The module registers them only once `realdata_pinned`, so while the
    pin is pending no store plan or backtest resolves their checkpoint.
    """
    for source in SOURCES:
        name = f"fm_chronos_{source}_ragged_real"
        register_model(
            name,
            partial(_real_data, name, source),
            f"Chronos-2 fine-tuned on real data (German and BDBV), on the "
            f"hub target's {source} releases, ragged layout (SI only).",
            fitted=True,
            replace=True,
        )


if Path(FINETUNED_CONFIG).exists() and realdata_pinned():
    register_realdata_models()
