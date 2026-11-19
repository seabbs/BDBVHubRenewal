# Shared settings and helper functions of the workflow's rules, kept
# apart from the rules themselves (rules/*.smk). Machine settings come
# from config/local.env (flowlib.localconfig).

import functools
import json

from flowlib import (
    grids,
    hosts,
    localconfig,
    manifest,
    notify,
    releases,
    segments,
    simulator,
    topup,
)

LOCAL = localconfig.read()
HOST = LOCAL["HOST_NAME"]
ROUNDS_DIR = f"{LOCAL['ROOT']}/rounds"
TARGETS = LOCAL["TARGETS"].split()
# The tools the shell rules call (envs/tools.yaml).
TOOLS_ENV = "../envs/tools.yaml"


def say(message):
    """A mailbox line, when NOTIFY is set in config/local.env."""
    notify.post(message, LOCAL)


def job_prefix(mem_mb):
    """The machine's command prefix for a job capped at `mem_mb` MB."""
    return LOCAL["JOB_PREFIX"].replace("{mem}", f"{int(mem_mb)}M")


def git_commit():
    """The commit the workflow runs at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# From simulate.smk.

SIM = config["simulate"]
# The sim QC's thresholds as scripts/sim_qc.py's options (simulate.qc).
SIM_QC_ARGS = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in SIM["qc"].items())
REMOTE = LOCAL["REMOTE"]
REMOTE_ROUNDS = f"{LOCAL['REMOTE_ROOT'] or LOCAL['ROOT']}/rounds"


def seeds_of(round_):
    """A round's batch seeds, its grid block."""
    return segments.round_seeds(int(round_), SIM["seed_base"], SIM["batches"])


MINE = hosts.seeds(LOCAL["SIM_SEEDS"], seeds_of)


def here(seed):
    """Whether this host simulates batch `seed`."""
    return hosts.simulates(int(seed), MINE)


# The epinowcast the simulator is pinned to (finetune_simulate.R's
# `epinowcast_sha`), installed once a machine under ROOT.
ENW_SHA = simulator.epinowcast_sha()
ENW_LIB = f"{LOCAL['ROOT']}/rlib/epinowcast-{ENW_SHA}"


def batch_lib(wc):
    """Already simulated batches take no library input.

    So installing it never reruns them.
    """
    sims = Path(f"{ROUNDS_DIR}/r{wc.round}/sims")
    done = all(
        (sims / f"{kind}-{wc.seed}.{ext}").exists()
        for kind, ext in (("counts", "csv.gz"), ("settings", "csv"))
    )
    return [] if done else ancient(f"{ENW_LIB}/.installed")


def batch_priority(wc):
    """A batch's priority: 15, earlier seeds a fraction higher."""
    return segments.batch_priority(int(wc.seed), 15, SIM["seed_base"])


def batch_files(wc):
    """A batch's files: simulated here, or pushed by its host."""
    if here(wc.seed):
        return {
            "counts": rules.simulate_batch.output.counts.format(**wc),
            "settings": rules.simulate_batch.output.settings.format(**wc),
        }
    return {"arrived": rules.await_batch.output[0].format(**wc)}


def grid_name():
    """The grid the simulator names (`grid_name <- "..."`), if any."""
    text = Path("scripts/finetune_simulate.R").read_text()
    found = re.search(r'^grid_name <- "([^"]*)"', text, re.M)
    return found.group(1) if found else ""


# From finetune.smk.

FT = grids.resolve(config["finetune"], config["grid_rounds"])
# Where batches come from is this machine's choice when set.
if LOCAL["SIMS_FROM"]:
    FT["sims_from"] = LOCAL["SIMS_FROM"]
RUN = LOCAL["RUN"]
TRAIN = f"{RUN}/train"

segments.check_chunks(FT["prefixes"], FT["chunk"], FT["outbreaks"])
segments.check_eval(FT["eval"])
PREV = dict((n, p) for p, n in segments.segments(FT["prefixes"]))


def round_of(seed):
    return (int(seed) - SIM["seed_base"]) // 100


def training_seeds():
    """The run's training seeds, in the order it trains on them."""
    if FT["seed_file"]:
        return [int(s) for s in Path(FT["seed_file"]).read_text().split()]
    return [s for r in FT["rounds"] for s in seeds_of(r)]


def release_rounds(n):
    """The rounds segment n trains on, space separated."""
    return " ".join(str(r) for r in sorted({round_of(s) for s in segment_seeds(n)}))


def held_seeds():
    return seeds_of(FT["validation_round"])


ELSEWHERE = FT["sims_from"] in ("release", "remote")
# The rounds host's rounds mirrored here: manifests and QC exclusions.
MIRROR = f"{RUN}/rounds"


def available_seeds():
    """The training seeds whose batches can be had now, in order.

    All of them from this machine's rounds, which the workflow simulates.
    From the rounds host or releases, those before the first batch not
    yet listed, and none until the validation round is all listed.
    """
    order = training_seeds()
    if not ELSEWHERE:
        return order
    listed = listed_batches()
    if not set(held_seeds()) <= listed:
        return []
    for i, seed in enumerate(order):
        if seed not in listed:
            return order[:i]
    return order


def exclusion_files():
    """The sim-QC exclusion lists here now."""
    paths = [Path(f"{RUN}/excluded.csv")]
    paths += Path(ROUNDS_DIR).glob("r*/qc/excluded.csv")
    paths += Path(MIRROR).glob("r*/qc/excluded.csv")
    paths += Path(f"{RUN}/excluded-releases").glob("*.csv")
    return sorted(str(p) for p in paths if p.is_file())


def known_exclusions():
    """The sim-QC exclusions known now."""
    return set().union(*(segments.excluded(p) for p in exclusion_files()))


def kept_or_full(seed, skip):
    """A batch's kept outbreaks if simulated, else a full batch."""
    for settings in (
        Path(f"{RUN}/sims/settings-{seed}.csv"),
        Path(f"{ROUNDS_DIR}/r{round_of(seed)}/sims/settings-{seed}.csv"),
    ):
        if settings.exists():
            return segments.kept_outbreaks(settings, skip)
    return SIM["batch"]


def segment_seeds(n):
    """The seeds segment n needs: n outbreaks, one spare, the held-out."""
    skip = known_exclusions()
    order = available_seeds()
    need = segments.seeds_for(n, order, lambda s: kept_or_full(s, skip))
    spare = order[len(need) : len(need) + 1]
    return need + spare + held_seeds()


def segment_inputs(wc):
    n = int(wc.n)
    # A finished checkpoint is final: its batches are not gathered again,
    # so a machine without them never retrains it.
    if Path(f"{TRAIN}/n{n}/settings.json").exists():
        return {}
    out = {
        "sims": expand(f"{RUN}/sims/settings-{{seed}}.csv", seed=segment_seeds(n)),
        "seeds": f"{RUN}/seeds.txt",
        # Read for the seeds above and by the trainer: inputs, so a
        # worker elsewhere has them (flowlib.compute).
        "qc": exclusion_files(),
    }
    if PREV[n] is not None:
        out["prev"] = f"{TRAIN}/n{PREV[n]}/settings.json"
    if FT["real_validation"]:
        out["real"] = FT["real_validation"]
    return out


def fetch_source(wc):
    if FT["sims_from"] != "local":
        return []
    r = round_of(wc.seed)
    return {
        "counts": f"{ROUNDS_DIR}/r{r}/sims/counts-{wc.seed}.csv.gz",
        "settings": f"{ROUNDS_DIR}/r{r}/sims/settings-{wc.seed}.csv",
    }


CHECK = FT.get("check_at", 1000)
AFTER = [n for n in sorted(FT["prefixes"]) if n > CHECK]


def trainable():
    """The prefixes done or within reach of the seed list."""
    skip = known_exclusions()
    order = available_seeds()
    # A spare batch past each prefix but the run's last, which takes
    # every training simulation.
    return [
        n
        for n in FT["prefixes"]
        if Path(f"{TRAIN}/n{n}/settings.json").exists()
        or segments.reachable(
            n + SIM["batch"] * (n < FT["outbreaks"]),
            order,
            lambda s: kept_or_full(s, skip),
        )
    ]


def to_train():
    """Prefixes `finetune` asks for."""
    # The no-bias control is another run: it must not train into the slots
    # run's folder (RUN's default).
    if not FT["series_bias"] and Path(RUN) == Path(LOCAL["ROOT"], "checkpoints/slots"):
        raise ValueError(
            f"a no-bias run (series_bias false) needs its own RUN, not {RUN}"
        )
    prefixes = trainable()
    if not config.get("next_segment_only"):
        return prefixes
    saved = [n for n in prefixes if Path(f"{TRAIN}/n{n}/settings.json").exists()]
    rest = [n for n in prefixes if n not in saved]
    return saved + rest[:1]


# From score.smk.

SC = config["score"]
CACHE = LOCAL["SCORE_CACHE"]


def checkpoint_labels(wc=None):
    """The checkpoints to score, less `score.skip`."""
    if SC["labels"] is not None:
        return list(SC["labels"])
    path = checkpoints.release_list.get().output[0]
    found = releases.labels(json.loads(Path(path).read_text()), SC["published_from"])
    skip = set(SC.get("skip") or ())
    return [lb for lb in found if lb not in skip]


ROOM = (
    f"workflow/scripts/wait_for_room.sh {LOCAL['MIN_FREE_GB']} "
    f"{LOCAL['MIN_DISK_GB']} {CACHE} {LOCAL['MAX_LOAD']} && "
)


# A unit's CPU threads (SCORE_THREADS, default 2): a few, so simulations
# keep the rest of the machine's cores.
UNIT_THREADS = int(LOCAL.get("SCORE_THREADS") or 2)


def score_order():
    """The checkpoint labels, oldest published first.

    Read from the release list if it is there, none otherwise: a job's
    priority cannot wait on the release_list checkpoint.
    """
    if SC["labels"] is not None:
        return list(SC["labels"])
    path = Path(f"{CACHE}/releases.json")
    if not path.exists():
        return []
    return releases.by_time(json.loads(path.read_text()), SC["published_from"])


def unit_priority(base):
    """A unit's priority: `base`, newer checkpoints a fraction higher."""
    return lambda wc: releases.priority(score_order(), wc.label, base)


def unit_env(wc, threads):
    """The unit's torch device, threads and Chronos-2 batch."""
    return machine.shell_env(machine.torch_env(LOCAL, threads))


EVALUATE = "uv run --group chronos python -m bdbvhubrenewal.evaluation.checkpoint"


def unit_rows(study):
    """The unit's rows folder, which a worker brings back with its output."""
    return lambda wc: f"{CACHE}/{study}/{wc.label}/"


def unit_shell(study):
    return (
        f"mkdir -p {{params.copy_back}} && {ROOM}{{params.prefix}} nice -n 19 "
        f"env {{params.env}} {EVALUATE} {study} {CACHE} {{wildcards.label}} "
        "> {log} 2>&1 && touch {output}"
    )


def table_inputs(wc):
    labels = checkpoint_labels()
    every = [*SC["reference"], *labels]
    return {
        "germany": expand(f"{CACHE}/germany/{{label}}.done", label=every),
        "release": expand(f"{CACHE}/heldout/{{label}}.done", label=labels),
    }


def unit_stamps(wc=None):
    """Every unit the tables read, as one list (`rule score` asks for them
    itself, so a new label's units run under tables already built)."""
    return [f for files in table_inputs(wc).values() for f in files]


# From followups.smk.

SLC = config["slots_check"]
AGG = config["agg"]
AGG_ROOT = f"{CACHE}/agg"
HEAVY = (
    f"{ROOM}{{params.prefix}} nice -n 19 env OMP_NUM_THREADS=1 "
    "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
)


def slots_labels():
    return [lb for lb in checkpoint_labels() if lb.startswith("slots-")]


def main_scores(wc):
    return [
        f"{CACHE}/germany/{wc.label}.done",
    ]


def src_tree():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD:src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def followups(wc=None):
    labels = slots_labels()
    out = []
    if SLC["run"]:
        out += expand(f"{CACHE}/slots-check/{{label}}/summary.csv", label=labels)
    if AGG["run"]:
        out += expand(f"{AGG_ROOT}/{{label}}/aggregation-scores.csv.gz", label=labels)
    return out


# The longer run directives' steps.


def write_record(input, output, params, wildcards):
    """Write a batch's manifest row, kept if the manifest has it."""
    import datetime as dt

    r, s = int(wildcards.round), int(wildcards.seed)
    line = manifest.existing(params.old, r, s)
    if line is None and not here(s):
        raise ValueError(f"batch {s} arrived without its manifest row")
    line = line or manifest.row(
        params.sims,
        round_=r,
        seed=s,
        kind="block",
        grid=grid_name(),
        commit=git_commit(),
        draws=SIM["draws"],
        batch=SIM["batch"],
        host=HOST,
        finished=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    Path(output[0]).write_text(f"{line}\n")


def mark_ready(input, output, params, wildcards):
    """Write a round's seed list and READY, and say so."""
    import datetime as dt

    seeds = seeds_of(wildcards.round)
    Path(output.seeds).write_text("".join(f"{s}\n" for s in seeds))
    Path(output.ready).write_text(f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}\n")
    qc = Path(input[0]).read_text().strip()
    say(
        f"round {wildcards.round} ready: {len(seeds)} batches, one grid " f"block; {qc}"
    )


def push_record(input, output, params, wildcards):
    """Send a helping host's batch and its manifest row to the rounds host."""
    import datetime as dt

    r, s = int(wildcards.round), int(wildcards.seed)
    row = manifest.row(
        params.sims,
        round_=r,
        seed=s,
        kind="block",
        grid=grid_name(),
        commit=git_commit(),
        draws=SIM["draws"],
        batch=SIM["batch"],
        host=HOST,
        finished=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    hosts.push(params.sims, s, params.there, row, REMOTE)


@functools.cache
def listed_batches():
    """The batches that can be had now, listed once per Snakemake run.

    Listed when the run plans, not kept in a file, so a later run sees
    rounds finished since. From the rounds host the manifests and QC
    exclusions of its rounds are mirrored in MIRROR, and a round counts
    once its QC has run; else the `sims-r<N>` releases' assets. Written to <RUN>/available-batches.txt for the record.
    """
    rounds = sorted({FT["validation_round"], *(round_of(s) for s in training_seeds())})
    seeds = set()
    for r in rounds:
        if FT["sims_from"] == "remote":
            # The manifest and QC exclusions of a round there so far.
            here = Path(f"{MIRROR}/r{r}")
            (here / "qc").mkdir(parents=True, exist_ok=True)
            for name in ("manifest.csv", "qc/excluded.csv"):
                there = f"{REMOTE_ROUNDS}/r{r}/{name}"
                got = subprocess.run(
                    ["rsync", "-a", hosts._at(REMOTE, there), str(here / name)],
                    capture_output=True,
                    text=True,
                )  # fmt: skip
                # A round with no QC yet has no exclusions; a round not
                # started has no manifest, and nor has an unreachable
                # host, so that is said (the mirror kept so far stays).
                if got.returncode and name == "manifest.csv":
                    print(
                        f"could not copy {there}: {got.stderr.strip()[:200]}",
                        file=sys.stderr,
                    )
            # QC runs once the round is whole: no batch it later excludes
            # is trained on.
            if (here / "qc" / "excluded.csv").exists():
                seeds |= manifest.seeds(here / "manifest.csv")
            continue
        # A round not released yet has no release.
        got = subprocess.run(
            ["gh", "release", "view", f"sims-r{r}", "-R", SIM["repo"],
             "--json", "assets"],
            capture_output=True,
            text=True,
        )  # fmt: skip
        if got.returncode == 0:
            names = [a["name"] for a in json.loads(got.stdout)["assets"]]
            seeds |= releases.released_seeds(names)
    record = Path(f"{RUN}/available-batches.txt")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("".join(f"{s}\n" for s in sorted(seeds)))
    return seeds


def check_report(input, output, params, wildcards):
    """Report whether checkpoint `check_at` beats zero-shot, and say so.

    On the held-out simulations only: BDBV is held out for the paper.
    """
    s = json.loads(Path(input[0]).read_text())
    ok = s["held_out_loss"] < s["held_out_loss_start"]
    text = (
        f"n{CHECK} {'beats' if ok else 'does NOT beat'} zero-shot: "
        f"held-out {s['held_out_loss']:.4f} vs "
        f"{s['held_out_loss_start']:.4f}"
    )
    Path(output[0]).write_text(f"{text}\n")
    nxt = f", check before n{AFTER[0]}" if AFTER else ""
    say(text if ok else f"ALERT {text}; training goes on{nxt}")


def land_tables(input, output, params, wildcards):
    """Commit the score tables on the evidence branch and push them."""
    import shutil

    def git(*args):
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    notify.check_branch(git("rev-parse", "--abbrev-ref", "HEAD"), params.branch)
    shutil.copy(input[0], params.evidence)
    shutil.copy(input[1], params.checkpoints)
    git("add", params.evidence, params.checkpoints)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-q", "-m", "evidence: checkpoint scores")
    git("push", "-q", "origin", params.branch)
    git("fetch", "-q", "origin", params.branch)
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", f"origin/{params.branch}"):
        raise SystemExit(f"{head} is not on origin/{params.branch}")
    say(f"checkpoint scores landed at {head[:9]} on origin/{params.branch}")


# From topup.smk.

GRID = config["grid_rounds"].get(grid_name()) or {}
TOPUP = GRID.get("topup_round")
TOPUP_DIR = f"{ROUNDS_DIR}/r{TOPUP}"
TOPUP_TRAINING = list(GRID.get("rounds") or [])
TOPUP_FIRST = SIM["seed_base"] + 100 * (TOPUP or 0)


def topup_plans():
    """The top-up round's plans so far, by try."""
    return sorted(
        Path(TOPUP_DIR, "topup").glob("plan-*.txt"),
        key=lambda p: int(p.stem.split("-")[1]),
    )


def batch_cells(wc):
    """A batch's ENW_CELLS: a top-up batch's planned cells, else none."""
    if TOPUP is None or int(wc.round) != TOPUP:
        return ""
    return ",".join(map(str, topup.cells_of(topup_plans(), int(wc.seed))))


def batch_size(wc):
    """A batch's simulations: a top-up batch's cells, else `batch`."""
    cells = batch_cells(wc)
    return len(cells.split(",")) if cells else SIM["batch"]


def topup_plan_inputs(attempt):
    """A try's inputs: the training rounds' QC, and the try before's."""
    if TOPUP is None:
        raise ValueError(f"grid {grid_name()} has no topup_round")
    out = [f"{ROUNDS_DIR}/r{r}/qc/excluded.csv" for r in TOPUP_TRAINING]
    if attempt:
        out.append(f"{TOPUP_DIR}/qc/try-{attempt - 1}.done")
    return out


def topup_write_plan(attempt, out, log):
    """Write try `attempt`'s batches of the cells still missing."""
    cells = topup.missing_cells(Path(ROUNDS_DIR), TOPUP_TRAINING, TOPUP)
    batches = topup.plan(cells, TOPUP_FIRST, attempt, SIM["batch"])
    topup.write_plan(batches, Path(out))
    Path(log).write_text(
        f"try {attempt}: {len(cells)} missing cells in {len(batches)} batches\n"
    )


def topup_records(plan):
    """The batch records of a plan's seeds."""
    return [
        f"{TOPUP_DIR}/records/record-{s}.csv" for s, _ in topup.read_plan(plan)
    ]


def topup_try_records(attempt):
    """Try `attempt`'s batch records, once its plan is written."""
    return topup_records(checkpoints.topup_plan.get(attempt=attempt).output[0])


def topup_done(wc):
    """Every try's plan, records and QC, up to the first with nothing left."""
    out = []
    for attempt in range(SIM.get("topup_tries", 5)):
        plan = checkpoints.topup_plan.get(attempt=attempt).output[0]
        out.append(plan)
        records = topup_records(plan)
        if not records:
            break
        out += [*records, f"{TOPUP_DIR}/qc/try-{attempt}.done"]
    return out


def topup_finish(input, output, log):
    """Write the top-up round's manifest, seeds and READY; say any shortfall."""
    import datetime as dt

    records = [p for p in input if "/records/" in p]
    manifest.join(records, output.manifest)
    seeds = [int(Path(p).stem.split("-")[1]) for p in records]
    Path(output.seeds).write_text("".join(f"{s}\n" for s in seeds))
    left = topup.missing_cells(Path(ROUNDS_DIR), TOPUP_TRAINING, TOPUP)
    note = (
        f"top-up round {TOPUP}: {len(seeds)} batches; "
        + (
            "every training cell filled"
            if not left
            else f"SHORT: {len(left)} cells still missing after "
            f"{SIM.get('topup_tries', 5)} tries ({left[:10]})"
        )
    )
    Path(log).write_text(note + "\n")
    Path(output.ready).write_text(f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}\n")
    say(note)
