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
| `task push ROUNDS="19 18"` | `push` | a helping host's batches, pushed to the rounds host | the rounds host |
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
- resources: `JOB_PREFIX` (a per-job memory cap such as a `systemd-run` scope, or `workflow/scripts/mem_cap.sh {mem}` on macOS; `{mem}` becomes the job's budget), `MIN_FREE_GB`, `MIN_DISK_GB` and `MAX_LOAD` (memory, disk and a load average below which a heavy job starts; `MAX_LOAD` empty is the core count), `HEADROOM_CORES` and `HEADROOM_GB` (cores and memory kept free), `CORES` and `MEM_MB` (set them outright instead), `SIM_SLOTS` (batches at once, 2 by default) and `DEVICE` (`auto`: CUDA, then MPS, then the CPU);
- sharing the rounds: `SIM_SEEDS` (the seeds this machine takes: seeds, `first-last` ranges or rounds as `r<N>`; empty: all), and on a helping machine `REMOTE` and `REMOTE_ROOT` (the rounds host and its rounds folder; empty: `ROOT`);
- `HOST_NAME`.

Nothing in the workflow needs a notes file. As an optional local extra, setting both `NOTIFY` and `MAILBOX` appends a one-line note to the file `MAILBOX` when a round is ready, a checkpoint is saved or published, the checked checkpoint misses zero-shot, scores land or a run fails. Both are empty by default, a note that cannot be written is only a warning, and no rule's output or success depends on it. The optional detached-job helpers in `tools/` use the same switch.

Task reads `config/local.env` itself, not `$LOCAL_CONFIG`.

Every target runs with one profile, `workflow/profiles/default`, and this machine's cores, memory and GPU less its headroom (`workflow/flowlib/machine.py`, through `workflow/flowlib/compute.py`, which the Taskfile and the tick call; `python3 workflow/flowlib/machine.py device` prints the device). `workflow/profiles/ci` is for CI.

### Where jobs run (`COMPUTE`)

`COMPUTE=single` (the default) runs every job on this machine, as above.
`COMPUTE=ssh` makes this machine the coordinator: it holds the one Snakemake run and places each job on a worker (`workflow/profiles/ssh`, the cluster-generic executor with `workflow/scripts/remote/`).

- `WORKERS` lists the workers as `name=host:ROOT:CHECKOUT`: the host as ssh names it (or `local`, which may leave out ROOT and CHECKOUT for this machine's), its rounds folder and its checkout of this repository.
- `WORKER_GROUPS` places rule groups on a worker, for example `WORKER_GROUPS="train=laptop score=archie"`. The groups are `simulate`, `train`, `publish` and `score` (`flowlib.compute.RULE_GROUPS`); a group left out, or `any`, runs on the first worker with room. Local rules (listings, records, publishing) always run on the coordinator.
- The pool is the sum of the workers' budgets, each read from the worker's own local config at launch (`machine.py budget`); a worker that does not answer is left out, and its jobs fail at submit until it does. A GPU rule placed by its group also takes that worker's own `gpu_<worker>`, so one worker's jobs never hold another's GPU. The submit script holds a job until it fits beside the jobs already on its worker.
- A job on another root or checkout: the worker must answer over ssh at submit, or the job fails at once (no other worker takes it); its checkout must be at the coordinator's commit; the job's inputs are pushed by rsync, its worker's Snakemake (`WORKER_SNAKEMAKE`) runs it with the coordinator's run settings (paths moved to the worker's ROOT) and the worker's own machine settings (device, headroom, caches); then its log, and on success its outputs and `params.copy_back`, are pulled back before the job counts as done. A worker silent for 10 minutes fails the job, and its retry resumes from the saved state.
- The trainer's own state stays on its worker, so `train` stays on one worker.

Only `single` and `ssh` ship. Rules declare only their needs (`threads`, `mem_mb`, `gpu`), placement lives only in the local config, and any other `COMPUTE=<name>` runs with `workflow/profiles/<name>` and nothing else from `flowlib.compute`, so a cluster or cloud profile needs no code:

- a cluster with a shared filesystem: add the executor plugin to the `paper` group (for SLURM, `snakemake-executor-plugin-slurm`) and a `workflow/profiles/<name>/config.yaml` with `executor: slurm`, `jobs`, the settings of `workflow/profiles/default` (`rerun-triggers: [mtime]`, `default-resources`, `set-threads`), the account and partition in `default-resources` (`slurm_account`, `slurm_partition`, `runtime`), and in `set-resources` where groups go (for example the GPU partition for `train_segment`, `score_bdbv` and `score_germany`; the rules' `gpu: 1` is the plugin's GPU request). No files move: every job sees ROOT;
- a cloud batch service: add an executor plugin (`snakemake-executor-plugin-googlebatch` or `-kubernetes`) and a storage plugin (`snakemake-storage-plugin-gcs` or `-s3`), and in the profile `executor`, `default-storage-provider`, `default-storage-prefix` (the bucket), `shared-fs-usage: none` and the executor's project, region and machine settings. Files then move through the bucket, not rsync, and ROOT must be a path the storage prefix maps (see the plugin's docs).

Neither is tested here.

A machine that needs more can name an untracked profile of its own with `SNAKEMAKE_PROFILE` (see `config/profile.example`), never one in the repository.
All of them rerun a step only when an input is newer (`rerun-triggers: mtime`), so editing a script does not reschedule finished work.
Settings that are the same on every machine are in `workflow/config.yaml`; override them with `--config` or `--configfile`.

Two machines share only GitHub releases: simulation batches (`sims-r<N>`, one asset per batch) and checkpoints (`chronos-slots-<hash>`, `chronos-ft-<hash>`).
Each machine reruns its task as releases land, in its coordinator run (`task run`).

## 1. Simulate (`rules/simulate.smk`)

Round r is 40 batches of 21 outbreaks, seeds from 30000 + 100 r, each with exactly one observation draw.
A round is grid block r, 840 simulations, and is used whole for training or as the held-out block.

- `epinowcast_lib`: installs the epinowcast the simulator is pinned to (`epinowcast_sha` in `scripts/finetune_simulate.R`) from GitHub into `ROOT/rlib/epinowcast-<sha>`, once a machine, at nice 19 under `JOB_PREFIX`. Its dependencies must already be in R's libraries.
- `simulate_batch`: one batch of `scripts/finetune_simulate.R`, under `JOB_PREFIX` at nice 19, once there is memory, disk and load to spare (`workflow/scripts/wait_for_room.sh`). A batch that fails is retried once at the higher cap. The simulator must be the committed one, and its whole command runs with that library and the system's tools ahead of PATH (R puts the system libraries on `LD_LIBRARY_PATH`, under which a bash from another prefix fails).
- `batch_record`: the batch's manifest row (sha256 of each file, memory, time, grid, commit, host), keeping a row the round's manifest already holds.
- `round_manifest`: the round's `manifest.csv`, rows in seed order.
- `sim_qc`: the recorded sim-QC review (`scripts/sim_qc.py`), whose exclusions (`qc/excluded.csv`) the trainer skips. Nothing is deleted.
- `round_ready`: the seed list and `READY`, last.
- `simulate`: the rounds in `rounds`. At most `sim_slots` batches run at once (2 by default).
- `release_batch`, `release_qc`, `release`: opt-in, each batch and the round's QC exclusions to its round's release.
- `push_batch`, `push`: on a helping host, its seeds (`SIM_SEEDS`), each pushed to the rounds host's round (`REMOTE`, `REMOTE_ROOT`): its files, then its manifest row (`flowlib.hosts`).
- `await_batch`: on the rounds host, a seed not in its `SIM_SEEDS`, done once the host that simulates it has pushed its manifest row; one waits at a time. Its row is kept as pushed.

Outputs are in `<ROOT>/rounds/r<N>/`.

## 2. Fine-tune (`rules/finetune.smk`)

One continuous run of `scripts/finetune_stream.py` from base Chronos-2, cut into one segment per checkpoint prefix (250, 500, 750, 1,000, 2,000, 4,000, 8,400).
Segment N resumes from the state kept with the prefix before and stops once `n<N>` is saved (`--until N`).
The state carries the learning-rate schedule, the step, the batch position and torch's random state, so the segments in turn are the run that never stopped.

- `seed_list`: the training seeds in order, the rounds' in seed order or a fixed `seed_file`. A run takes its grid's rounds (`grid_rounds`): r5 holds out round 9 and trains on 10 to 19, r6 holds out round 20 and trains on 21 to 30. No two grids share a round, as a round sets its seeds and its release `sims-r<N>`. The `release` target refuses another grid's round, and each upload refuses a release whose notes name another grid (unmarked releases are r5's).
- `fetch_batch`: a batch into `<RUN>/sims/`, hard linked from this machine's rounds, copied from the rounds host's (`remote`, `REMOTE` and `REMOTE_ROOT`) or downloaded from its release (`release`; `sims_from`, or `SIMS_FROM` in `config/local.env`).
- The batches a machine without the rounds can have are listed afresh each time a run plans (`listed_batches` in rules/common.smk, recorded in `<RUN>/available-batches.txt`): those the rounds host's manifests list (mirrored with its QC exclusions in `<RUN>/rounds/`), or those released. A segment is planned only once its batches are listed, counting the training seeds up to the first batch not yet listed, and nothing trains before the whole held-out round is listed.
- `real_validation`: the BDBV pseudo-origin blocks each checkpoint is scored on (`scripts/finetune_real_validation.py`).
- `train_segment`: one segment on the GPU. Its inputs are the batches that hold its first N outbreaks after QC exclusions, one spare, and the held-out round's batches. A finished checkpoint is final and is never retrained.
- `checkpoint_check`: a report of whether checkpoint `check_at` (1,000) beats zero-shot on the held-out simulations and the pseudo-origins. It never stops training.
- `finetune`: every prefix.

Outputs are in `<RUN>/train/n<N>/`.

## 3. Publish (`rules/publish.smk`)

- `verify_checkpoint`: the checkpoint's hash recomputes from its commit (`scripts/finetune_verify.py`).
- `publish_checkpoint`: opt-in, the release `chronos-slots-<hash>` or `chronos-ft-<hash>` (`workflow/scripts/publish_checkpoint.py`).
- `publish`: every checkpoint.

## 4. Score (`rules/score.smk`, `rules/followups.smk`)

- `release_list`: the repository's releases, the only rule that reads the network; `task score` and each tick rerun it in a run of its own before scoring, as Snakemake 9 ends a run after a forced checkpoint without the jobs it unlocks. The checkpoints scored are the slots ones first, then the others, oldest first, from `published_from`, or `score.labels`.
- `bdbv_data`: the hub's data (`bdbvhubrenewal data`), or `task data`. Newer data reruns the BDBV units, which add only the new origins.
- `score_bdbv`, `score_zones`, `score_germany`, `score_release`: the units of `bdbvhubrenewal.evaluation.checkpoint` for each checkpoint and the references, newest checkpoint first. BDBV units run before German ones. Each caches one file per origin or date and resumes from it.
  Units run torch on the machine's device with `SCORE_THREADS` threads and the `gpu` resource. `score_zones` forecasts with the health zones in Chronos-2's block, as submitted, on the CPU with `score.zones_mem_mb`.
- `score_table`: the rows of every checkpoint in `<SCORE_CACHE>/tables/`, keeping other rules' rows of the committed tables.
- `zones`: target, not part of `score`. It runs `score_zones` for every label, then writes the tables again with their zone rows. `score_table` takes zone rows only from finished zone units, so BDBV and German rows never wait for zones.
- `land`: opt-in, commit the tables to `evidence/` and push, only from the evidence branch (`score.branch`), never main.

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
# open the address it prints; with the venv active, plain `python` works too
```

## Scheduled runs

The workflow updates itself as results come in. Each machine runs its own targets in one persistent coordinator run, one tick after another; Snakemake reruns only what changed. With `COMPUTE=ssh` one coordinator runs every machine's jobs.

| Where | `SCHEDULE_TARGETS` | Does, each tick |
|---|---|---|
| the machine that scores (CPU) | `score land` | lists new checkpoint releases, scores them, rebuilds the table, and commits and pushes it to the evidence branch only if it changed |
| the machine that trains (GPU) | `finetune publish` | trains the next segment once its batches are there, and releases each new checkpoint |
| a simulating machine | `simulate release` | simulates the configured rounds and releases each batch |

A tick (`task tick`, `workflow/scripts/tick.py`) runs the pool targets (`simulate`, `push`, `finetune`, `score`, `dashboard`) as one Snakemake run at nice 19 with the default profile and the machine's budget, so a free core or the GPU takes whichever queued job fits.
When jobs compete, the trainer goes first (priority 20), then simulation (15), then checkpoint scoring (10 and below).
Before it, the lists the pool plans from (the checkpoint releases, and on a machine without the rounds the batches it can have) are refreshed in a run of their own.
A failed refresh leaves the last lists, and the pool still runs.
The outward targets (`land`, `publish`, `release`) follow in their configured order.
A failed run skips `land` and `release`, so `land` never follows a failed `score`, but `publish` still releases the checkpoints saved.
A tick's run trains only the next segment (`next_segment_only`), so it ends after one segment and the next run plans afresh.
Work that appears during a run, such as a new checkpoint or a landed round, waits for the next tick's run, so it waits at most as long as the longest running job, one segment.
Snakemake locks the worktree it runs in, so each machine has one driver worktree.
A tick is skipped when another tick is still running (a lock in `output/`) or when less than `MIN_FREE_GB` of memory or `MIN_DISK_GB` of disk is free. Its log is `output/workflow-tick.log`.

`task run` (`workflow/scripts/coordinate.py`) is the coordinator run: it runs a tick, and as soon as one ends starts the next, after `COORDINATE_PAUSE` seconds (60), or `COORDINATE_BACKOFF` (600) after a failed tick. No timer runs it: start it once, under nohup or tmux, and stop it with `touch output/coordinator.stop` (it ends after the running tick) or by its pid.
The timer that ran ticks before is not installed any more; `task schedule` shows an old one (`ACTION=show`) and removes it (`ACTION=uninstall`).

On CI, the site and the paper (`.github/workflows/pages.yml`) rebuild every 6 hours (`43 */6 * * *`, after the fit store's run) and whenever `evidence/` changes on main, so new checkpoint scores reach the docs and the paper without a manual step.

Nothing here needs an agent or a notes file; with `NOTIFY` set, rules still add their optional notes.

## Monitor

`task monitor` serves one page of what runs where: each machine's step, its running jobs (rule, wildcards, resources, start) and its recent ones.
Each tick writes its machine's status to `<ROOT>/status/<host>.json` every `STATUS_EVERY` seconds while a step runs, and when it is skipped or ends (`workflow/flowlib/runstatus.py`, read from the run's own Snakemake log).
With a rounds host (`REMOTE`) the file is copied to its `status/` folder, which the monitor reads, so one page shows every machine.
A run that stops writing for two intervals, because it was killed or its machine went down, is shown as stale.
The page is served on `MONITOR_HOST` and `MONITOR_PORT` and refreshes every `MONITOR_REFRESH` seconds, all from `config/local.env`.
It is separate from the fine-tuning dashboard, of which several may run at once.
Work run outside the workflow does not appear until it runs under the tick.

We chose this over Panoptes.
Snakemake 9 has no `--wms-monitor`; monitoring goes through logger plugins.
The Panoptes logger plugin is not maintained by the Snakemake organisation, and Panoptes needs a server that every machine can reach.
The status files need neither: the machines already reach the rounds host.

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
