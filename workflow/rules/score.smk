# Checkpoint scoring (was scripts/checkpoint_poll.py).
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


def unit_shell(study):
    return (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false "
        "{params.prefix} nice -n 19 uv run --group chronos python "
        f"scripts/checkpoint_scores.py {study} {CACHE} {{wildcards.label}} "
        "> {log} 2>&1 && touch {output}"
    )


rule score_bdbv:
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
    # Opt-in: commit the tables to evidence/ on this branch and push,
    # checking the commit reached origin.
    input:
        rules.score_table.output,
    output:
        touch(f"{CACHE}/tables/landed"),
    params:
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
    localrule: True
    shell:
        """
        cp {input[0]} {params.evidence}
        cp {input[1]} {params.checkpoints}
        git add {params.evidence} {params.checkpoints}
        git diff --cached --quiet || git commit -q -m "evidence: checkpoint scores"
        git push -q
        git fetch -q origin
        [ "$(git rev-parse HEAD)" = "$(git rev-parse @{{u}})" ]
        """


rule score:
    input:
        rules.score_table.output,
