# Checkpoint scoring.
#
# Each checkpoint label gets three units of scripts/checkpoint_scores.py:
# `germany` (WIS on the German development half), `bdbv` (sample CRPS on
# the BDBV release backtest) and `heldout` (the trainer's held-out loss).
# A unit caches one file per date or origin in <SCORE_CACHE>/<study>/
# <label>/ and resumes from it; its done stamp is written when it exits
# cleanly. `score_table` merges every label's rows into the evidence
# tables in <SCORE_CACHE>/tables/. BDBV units run before German ones,
# and within each the newest checkpoint first. A unit runs torch on
# this machine's device (flowlib.machine) with SCORE_THREADS threads,
# and takes the `gpu` resource, so units on one GPU queue.
#
# Targets: `score`, or one unit, <SCORE_CACHE>/germany/<label>.done.
# `land` (opt-in) commits the tables to evidence/ and pushes.
# `germany_forecast` (opt-in, `score.germany_forecast.enabled`) runs the
# German 1 to 4 week forecasts of the reference and one checkpoint, each
# after its German nowcasts; with it on, `score` takes them too.

import json

from flowlib import releases

SC = config["score"]
CACHE = LOCAL["SCORE_CACHE"]


checkpoint release_list:
    # Lists the releases to score; `task score` reruns it.
    output:
        f"{CACHE}/releases.json",
    localrule: True
    params:
        repo=SIM["repo"],
    shell:
        "gh release list -R {params.repo} --limit 500 "
        "--json tagName,publishedAt > {output}"


def checkpoint_labels(wc=None):
    """The checkpoints to score."""
    if SC["labels"] is not None:
        return list(SC["labels"])
    path = checkpoints.release_list.get().output[0]
    return releases.labels(json.loads(Path(path).read_text()), SC["published_from"])


rule bdbv_data:
    # The hub's data (`bdbvhubrenewal data`). A refresh (`task data`) makes
    # the BDBV units rerun, and they add only the new origins.
    output:
        SC["data"],
    localrule: True
    shell:
        "uv run bdbvhubrenewal data"


ROOM = (
    f"workflow/scripts/wait_for_room.sh {LOCAL['MIN_FREE_GB']} "
    f"{LOCAL['MIN_DISK_GB']} {CACHE} && "
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


def unit_shell(study):
    return (
        "mkdir -p " + CACHE + " && " + ROOM + "{params.prefix} nice -n 19 "
        "env {params.env} uv run --group chronos python "
        f"scripts/checkpoint_scores.py {study} {CACHE} {{wildcards.label}} "
        "> {log} 2>&1 && touch {output}"
    )


rule score_bdbv:
    input:
        SC["data"],
    output:
        f"{CACHE}/bdbv/{{label}}.done",
    log:
        f"{CACHE}/log/bdbv-{{label}}.log",
    priority: unit_priority(10)
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=1,
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
    shell:
        unit_shell("bdbv")


rule score_germany:
    output:
        f"{CACHE}/germany/{{label}}.done",
    log:
        f"{CACHE}/log/germany-{{label}}.log",
    priority: unit_priority(5)
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=1,
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
    shell:
        unit_shell("germany")


rule score_heldout:
    output:
        f"{CACHE}/heldout/{{label}}.done",
    log:
        f"{CACHE}/log/heldout-{{label}}.log",
    localrule: True
    shell:
        f"uv run python scripts/checkpoint_scores.py heldout {CACHE} "
        "{wildcards.label} > {log} 2>&1 && touch {output}"


rule score_germany_forecast:
    # Opt-in (`score.germany_forecast`): never part of `score` unless on.
    input:
        f"{CACHE}/germany/{{label}}.done",
    output:
        f"{CACHE}/germany_forecast/{{label}}.done",
    log:
        f"{CACHE}/log/germany_forecast-{{label}}.log",
    priority: unit_priority(1)
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=1,
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
    shell:
        unit_shell("germany_forecast")


def forecast_inputs(wc=None):
    """The German forecast units, none unless switched on."""
    fc = SC.get("germany_forecast") or {}
    if not fc.get("enabled"):
        return []
    labels = [*SC["reference"], *([fc["label"]] if fc.get("label") else [])]
    return expand(f"{CACHE}/germany_forecast/{{label}}.done", label=labels)


def table_inputs(wc):
    labels = checkpoint_labels()
    every = [*SC["reference"], *labels]
    return {
        "bdbv": expand(f"{CACHE}/bdbv/{{label}}.done", label=every),
        "germany": expand(f"{CACHE}/germany/{{label}}.done", label=every),
        "heldout": expand(f"{CACHE}/heldout/{{label}}.done", label=labels),
        "forecast": forecast_inputs(),
    }


rule score_table:
    input:
        unpack(table_inputs),
    output:
        evidence=f"{CACHE}/tables/checkpoint-scores.csv",
        checkpoints=f"{CACHE}/tables/checkpoints.csv",
    localrule: True
    params:
        labels=lambda wc: " ".join(checkpoint_labels()),
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
    shell:
        # Rows of other rules in the committed tables are kept.
        """
        [ -f {params.evidence} ] && cp {params.evidence} {output.evidence}
        [ -f {params.checkpoints} ] && cp {params.checkpoints} {output.checkpoints}
        uv run python scripts/checkpoint_scores.py table {CACHE} \
            {output.evidence} {output.checkpoints} {params.labels}
        """


rule land:
    # Opt-in: commit the tables to evidence/ on the evidence branch only
    # (never main) and push, checking the commit reached origin.
    input:
        rules.score_table.output,
    output:
        touch(f"{CACHE}/tables/landed"),
    localrule: True
    params:
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        branch=SC["branch"],
    run:
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


rule germany_forecast:
    # Opt-in target: the German forecast units alone.
    input:
        forecast_inputs,


rule score:
    input:
        rules.score_table.output,
        lambda wc: followups(wc),
