# The workflow

One Snakemake workflow simulates outbreaks, fine-tunes Chronos-2 on them, publishes the checkpoints and scores them.
The paper's workflow (`paper/workflow/Snakefile`) runs beside it for now and joins as a module next (#257).
This page walks through it in order, with the command for each piece, from a whole target down to one file.

## Running it

High level, one Taskfile task per target:

| Task | Target | Does | Writes outside this machine |
|---|---|---|---|
| `task all` | `all` | simulate, fine-tune and score | no |
| `task simulate ROUNDS="4 5"` | `simulate` | simulation rounds, with QC and manifests | no |
| `task release ROUNDS=4` | `release` | each batch to its `sims-r<N>` release | GitHub releases |
| `task finetune` | `finetune` | every checkpoint prefix of the run | no |
| `task publish` | `publish` | verify and release each checkpoint | GitHub releases |
| `task score` | `score` | score each released checkpoint, then its follow-ups | no |
| `task land` | `land` | commit the score tables to `evidence/` and push | git push |
| `task paper` | (paper workflow) | the paper, as now | no |

Pass Snakemake options after `--`: `task score -- -n` is a dry run, and a file path builds just that output.
Low level, any output file, for example:

```sh
uv run --group paper snakemake <ROOT>/rounds/r4/sims/counts-30421.csv.gz  # one batch
uv run --group paper snakemake <RUN>/train/n1000/settings.json            # one checkpoint
uv run --group paper snakemake <SCORE_CACHE>/germany/slots-<hash>.done    # one checkpoint's German scores
```

Helpers: `task dag` draws the rule graph, `task progress` lists what exists and what would run, `task report` writes Snakemake's HTML report, and `task workflow:check` runs the tests and dry runs CI runs.

## Machine settings

Paths and host details are in `config/local.env`, which git ignores; `config/local.env.example` lists every key.
The same file is read by the Taskfile (`dotenv`), by the shell scripts (`scripts/local_config.sh`) and by the workflow (`workflow/flowlib/localconfig.py`).
A variable in the environment wins, and with no file everything runs on this machine.
The workflow uses:

- paths: `ROOT` (rounds), `RUN` (the fine-tuning run), `SCORE_CACHE`, `NOSLOTS_RUN` and `GERMANY_BLOCKS` (the slots check's);
- what this machine does: `TARGETS` (what `all` runs; by default `simulate finetune score`, so a CPU machine sets `simulate score` and a GPU machine `finetune`), `SIMS_FROM` (`local` or `release`), `PUBLISH` (release each checkpoint as it is trained);
- resources: `JOB_PREFIX` (a per-job memory cap such as a `systemd-run` scope; `{mem}` becomes the job's budget), `MIN_FREE_GB` and `MIN_DISK_GB` (room a heavy job waits for), and the Taskfile's `CORES`, `MEM_MB` and `SIM_SLOTS`;
- `HOST_NAME`.

Nothing in the workflow needs a notes file. As an optional local extra, setting both `NOTIFY` and `MAILBOX` appends a one-line note to the file `MAILBOX` when a round is ready, a checkpoint is saved or published, the checked checkpoint misses zero-shot, scores land or a run fails. Both are empty by default, a note that cannot be written is only a warning, and no rule's output or success depends on it. The optional detached-job helpers in `tools/` use the same switch.

Task reads `config/local.env` itself, not `$LOCAL_CONFIG`.

Profiles in `workflow/profiles/` hold resources only and are named by role: `cpu` (simulate, score), `gpu` (fine-tune, publish) and `ci`.
All of them rerun a step only when an input is newer (`rerun-triggers: mtime`), so editing a script does not reschedule finished work.
Settings that are the same on every machine are in `workflow/config.yaml`; override them with `--config` or `--configfile`.

Two machines share only GitHub releases: simulation batches (`sims-r<N>`, one asset per batch) and checkpoints (`chronos-slots-<hash>`, `chronos-ft-<hash>`).
Each machine reruns its task on a timer as releases land.

## 1. Simulate (`rules/simulate.smk`)

Round r is 40 training and 4 held-out batches of 25 outbreaks, seeds from 30000 + 100 r, each with exactly one observation draw.

- `simulate_batch`: one batch of `scripts/finetune_simulate.R`, under `JOB_PREFIX` at nice 19, once there is memory and disk to spare (`workflow/scripts/wait_for_room.sh`). A batch that fails is retried once at the higher cap. The simulator must be the committed one.
- `batch_record`: the batch's manifest row (sha256 of each file, memory, time, grid, commit, host), keeping a row the round's manifest already holds.
- `round_manifest`: the round's `manifest.csv`, rows in seed order.
- `sim_qc`: the recorded sim-QC review (`scripts/sim_qc.py`), whose exclusions (`qc/excluded.csv`) the trainer skips. Nothing is deleted.
- `round_ready`: the seed lists and `READY`, last.
- `simulate`: the rounds in `rounds`. At most `sim_slots` batches run at once (2 by default).
- `release_batch`, `release_qc`, `release`: opt-in, each batch and the round's QC exclusions to its round's release.

Outputs are in `<ROOT>/rounds/r<N>/`.

## 2. Fine-tune (`rules/finetune.smk`)

One continuous run of `scripts/finetune_stream.py` from base Chronos-2, cut into one segment per checkpoint prefix (250, 500, 750, 1,000, 2,000, 4,000, 8,000).
Segment N resumes from the state kept with the prefix before and stops once `n<N>` is saved (`--until N`).
The state carries the learning-rate schedule, the step, the batch position and torch's random state, so the segments in turn are the run that never stopped.

- `seed_list`: the training seeds in order, the rounds' in seed order (rounds 9 to 17, grid r5) or a fixed `seed_file`.
- `fetch_batch`: a batch into `<RUN>/sims/`, hard linked from this machine's rounds or downloaded from its release (`sims_from`).
- `real_validation`: the BDBV pseudo-origin blocks each checkpoint is scored on (`scripts/finetune_real_validation.py`).
- `train_segment`: one segment on the GPU. Its inputs are the batches that hold its first N outbreaks after QC exclusions, one spare, and the held-out batches. A finished checkpoint is final and is never retrained.
- `checkpoint_check`: a report of whether checkpoint `check_at` (1,000) beats zero-shot on the held-out simulations and the pseudo-origins. It never stops training.
- `finetune`: every prefix.

Outputs are in `<RUN>/train/n<N>/`.

## 3. Publish (`rules/publish.smk`)

- `verify_checkpoint`: the checkpoint's hash recomputes from its commit (`scripts/finetune_verify.py`).
- `publish_checkpoint`: opt-in, the release `chronos-slots-<hash>` or `chronos-ft-<hash>` (`workflow/scripts/publish_checkpoint.py`).
- `publish`: every checkpoint.

## 4. Score (`rules/score.smk`, `rules/followups.smk`)

- `release_list`: the repository's releases, the only rule that reads the network; `task score` reruns it. The checkpoints scored are the slots ones first, then the others, oldest first, from `published_from`, or `score.labels`.
- `bdbv_data`: the hub's data (`bdbvhubrenewal data`), or `task data`. Newer data reruns the BDBV units, which add only the new origins.
- `score_bdbv`, `score_germany`, `score_heldout`: the units of `scripts/checkpoint_scores.py` for each checkpoint and the references. BDBV units run before German ones. Each caches one file per origin or date and resumes from it.
- `score_heldout_scores`, `score_zones`: proper scores on the `score.held_set` held-out simulations (`scripts/heldout_scores.py`), by layout and design factor level, and the backtest's health zones (`scripts/bdbv_zones.py`). The reference's held-out scores wait for a checkpoint's settings, whose held-out hash they are checked against. Labels in `score.kept` join the tables from the cache and are never scored.
- `score_table`: the rows of every checkpoint in `<SCORE_CACHE>/tables/`, keeping other rules' rows of the committed tables.
- `land`: opt-in, commit the tables to `evidence/` and push, only from the evidence branch (`score.branch`), never main.
- `score_germany_forecast`, `germany_forecast`: opt-in (`score.germany_forecast.enabled`, off by default), the SI's German 1 to 4 week forecasts of the development half for the reference and one checkpoint (`score.germany_forecast.label`), each after its German nowcasts. With it on, `score` takes them too. The rows are study `germany_forecast`.

After each slots checkpoint's scores:

- `fetch_adapter`: its adapter from the release, checked against `sha256sums.txt`.
- `slots_check_shard`, `slots_check_summary`: the slots check (`scripts/slots_check.py`) against zero-shot and the no-slots checkpoint of its size.
- `agg_data`, `agg_sanity`, `agg_fit`, `agg_rows`: the aggregation arms (`scripts/agg/`), with the data pinned to the zero-shot rows' hub commit and a sanity check that a zero-shot arm reproduces its rows, once per package version.
- `followups`: all of these; `score` includes them.

## 5. Dashboard (`workflow/flowlib/dashboard/`)

`task dashboard`, or `uv run --group paper python workflow/scripts/dashboard.py --port 8765`, serves a page for a running fine-tune on 127.0.0.1; it is not a Snakemake job, so it never holds the workflow's lock.
`task dashboard:static` builds the rule `dashboard`, a static render of the same page in `<RUN>/reports/dashboard.html`.

The page has a sticky bar (section links, the live run's step, one run and reference selector every chart follows) and sections: Status (tiles), Loss (client-side plots), Attention bias, Performance on real data (scores by study, score, target and source), Nowcasts (every dataset), Recent activity (with a notes file) and Tables. Every chart has an info marker (`page/explain.json`).

Settings:

- `workflow/config.yaml`'s `dashboard`: references (matched on rows by model and checkpoint, with a role: `baseline`, `start` or `other`), score studies and their labels, level order and defaults, the datasets (Germany: quantiles and WIS; BDBV: samples, summarised to quantiles with the scorer's truth rule and sample CRPS), the notes filter.
- `config/dashboard.local.yaml` (git ignores it; `config/dashboard.example.yaml` is committed; `DASHBOARD_CONFIG` in `config/local.env` names another file): this machine's runs (training output folders with a hue, the live run, a base run, or release prefixes), exclusions, the mirror folder, and where the score table, each dataset, the BDBV truth table, extra tables, notes and the notes file come from. A `host:path` source is copied read-only with rsync over ssh; nothing is written back.
- `DASHBOARD_RUNS` and `--run`, `--ref`: more runs and references on top.

On a laptop that trains while another machine scores:

```sh
uv sync --group paper                                          # once
cp config/dashboard.example.yaml config/dashboard.local.yaml   # then edit paths
uv run --group paper python workflow/scripts/dashboard.py --port 8765
# open http://127.0.0.1:8765/; with the venv active, plain `python` works too
```

## Scheduled runs

The workflow updates itself as results come in. Each machine runs its own targets on a timer; Snakemake reruns only what changed.

| Where | `SCHEDULE_TARGETS` | Does, each tick |
|---|---|---|
| the machine that scores (CPU) | `score land` | lists new checkpoint releases, scores them, rebuilds the table, and commits and pushes it to the evidence branch only if it changed |
| the machine that trains (GPU) | `finetune publish` | trains the next segment once its batches are there, and releases each new checkpoint |
| a simulating machine | `simulate release` | simulates the configured rounds and releases each batch |

A tick (`task tick`, `workflow/scripts/tick.py`) runs the targets in order at nice 19 with the machine's `CORES` and `MEM_MB`. It is skipped when another tick is still running (a lock in `output/`) or when less than `MIN_FREE_GB` of memory or `MIN_DISK_GB` of disk is free; a failed target stops the tick, so `land` never follows a failed `score`. Its log is `output/workflow-tick.log`.

`task schedule ACTION=install` installs a timer that runs the tick every `SCHEDULE_EVERY` minutes (default 60): a systemd user timer on Linux with systemd, else a cron line; a launchd agent on macOS; or the one `SCHEDULE_KIND` names. The unit is written from where the checkout is and the `PATH` it is installed from, so nothing machine-specific is committed. `ACTION=show` prints it, `ACTION=uninstall` removes it.

On CI, the site and the paper (`.github/workflows/pages.yml`) rebuild every 6 hours (`43 */6 * * *`, after the fit store's run) and whenever `evidence/` changes on main, so new checkpoint scores reach the docs and the paper without a manual step.

Nothing here needs an agent or a notes file; with `NOTIFY` set, rules still add their optional notes.

## CI

| Target | On CI | Why |
|---|---|---|
| tests and dry runs of every target (`task workflow:check`) | yes | fast, no network |
| the paper and its figures and tables | yes, as now (Pages) | from stored forecasts and pinned evidence |
| `score_table` from stored caches | yes, once the paper module lands | cheap |
| checkpoint forecasts, slots check, aggregation fits | no | hours, needs the checkpoints |
| `simulate`, `finetune` | never | heavy R, GPU |
| `release`, `publish`, `land` | never | outward-facing |

`all` is the default target.
