# Checkpoint scoring.
#
# Each checkpoint label gets three units of scripts/checkpoint_scores.py:
# `germany` (WIS on the German development half), `bdbv` (sample CRPS on
# the BDBV release backtest) and `heldout` (the trainer's held-out loss).
# A unit caches one file per date or origin in <SCORE_CACHE>/<study>/
# <label>/ and resumes from it; its done stamp is written when it exits
# cleanly. `score_table` merges every label's rows into the evidence
# tables in <SCORE_CACHE>/tables/. BDBV units run before German ones.
#
# Targets: `score`, or one unit, <SCORE_CACHE>/germany/<label>.done.
# `land` (opt-in) commits the tables to evidence/ and pushes.

import json

from flowlib import releases

SC = config["score"]
CACHE = LOCAL["SCORE_CACHE"]


checkpoint release_list:
    # The only rule that reads the network; `task score` reruns it.
    output:
        f"{CACHE}/releases.json",
    params:
        repo=SIM["repo"],
    localrule: True
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


def unit_shell(study):
    return (
        "mkdir -p " + CACHE + " && " + ROOM + "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false "
        "{params.prefix} nice -n 19 uv run --group chronos python "
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
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
    priority: 10
    threads: 1
    resources:
        mem_mb=SC["mem_mb"],
    shell:
        unit_shell("bdbv")


rule score_germany:
    output:
        f"{CACHE}/germany/{{label}}.done",
    log:
        f"{CACHE}/log/germany-{{label}}.log",
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
    priority: 5
    threads: 1
    resources:
        mem_mb=SC["mem_mb"],
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


def table_inputs(wc):
    labels = checkpoint_labels()
    every = [*SC["reference"], *labels]
    return {
        "bdbv": expand(f"{CACHE}/bdbv/{{label}}.done", label=every),
        "germany": expand(f"{CACHE}/germany/{{label}}.done", label=every),
        "heldout": expand(f"{CACHE}/heldout/{{label}}.done", label=labels),
    }


rule score_table:
    input:
        unpack(table_inputs),
    output:
        evidence=f"{CACHE}/tables/checkpoint-scores.csv",
        checkpoints=f"{CACHE}/tables/checkpoints.csv",
    params:
        labels=lambda wc: " ".join(checkpoint_labels()),
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
    localrule: True
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
    params:
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        branch=SC["branch"],
    localrule: True
    run:
        import shutil

        def git(*args):
            return subprocess.run(
                ["git", *args], check=True, capture_output=True, text=True
            ).stdout.strip()

        notify.check_branch(
            git("rev-parse", "--abbrev-ref", "HEAD"), params.branch
        )
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


rule score:
    input:
        rules.score_table.output,
        lambda wc: followups(wc),
