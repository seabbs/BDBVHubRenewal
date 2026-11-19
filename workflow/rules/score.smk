# Checkpoint scoring.
#
# Each checkpoint label gets the units of bdbvhubrenewal.evaluation
# .checkpoint: `germany` (WIS on the German development half) and
# `release` (the release's settings, with the trainer's held-out loss).
# BDBV is held out for the paper (Sam, 2026-11-19), so no BDBV unit runs
# here; BDBV rows already in the tables stay.
# A unit caches one file per date or origin in <SCORE_CACHE>/<study>/
# <label>/ and resumes from it; its done stamp is written when it exits
# cleanly. `score_table` merges every label's rows into the evidence
# tables in <SCORE_CACHE>/tables/, newest checkpoint first. A unit runs
# torch on this machine's device (flowlib.machine) with SCORE_THREADS
# threads, and takes the `gpu` resource, so units on one GPU queue.
#
# Targets: `score`, or one unit, <SCORE_CACHE>/germany/<label>.done.
# `land` (opt-in) commits the tables to evidence/ and pushes.


checkpoint release_list:
    # Lists the releases to score; `task score` reruns it.
    output:
        f"{CACHE}/releases.json",
    log:
        f"{CACHE}/log/release-list.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
    shell:
        "gh release list -R {params.repo} --limit 500 "
        "--json tagName,publishedAt > {output} 2>{log}"


rule score_germany:
    output:
        f"{CACHE}/germany/{{label}}.done",
    log:
        f"{CACHE}/log/germany-{{label}}.log",
    priority: unit_priority(5)
    conda:
        TOOLS_ENV
    threads: UNIT_THREADS
    resources:
        mem_mb=SC["mem_mb"],
        gpu=1,
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        env=unit_env,
    shell:
        unit_shell("germany")


rule score_release:
    output:
        f"{CACHE}/heldout/{{label}}.done",
    log:
        f"{CACHE}/log/heldout-{{label}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        evaluate=EVALUATE,
        cache=lambda wc: CACHE,
    shell:
        "{params.evaluate} release {params.cache} {wildcards.label} > {log} 2>&1 "
        "&& touch {output}"


rule score_table:
    input:
        unpack(table_inputs),
    output:
        evidence=f"{CACHE}/tables/checkpoint-scores.csv",
        checkpoints=f"{CACHE}/tables/checkpoints.csv",
    log:
        f"{CACHE}/log/score-table.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        labels=lambda wc: " ".join(checkpoint_labels()),
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        cache=lambda wc: CACHE,
        evaluate=EVALUATE,
    shell:
        # Rows of other rules in the committed tables are kept.
        """
        exec >{log} 2>&1
        [ -f {params.evidence} ] && cp {params.evidence} {output.evidence}
        [ -f {params.checkpoints} ] && cp {params.checkpoints} {output.checkpoints}
        {params.evaluate} table {params.cache} \
            {output.evidence} {output.checkpoints} {params.labels}
        """


rule land:
    # Opt-in: commit the tables to evidence/ on the evidence branch only
    # (never main) and push, checking the commit reached origin.
    input:
        rules.score_table.output,
    output:
        touch(f"{CACHE}/tables/landed"),
    log:
        f"{CACHE}/log/land.log",
    localrule: True
    params:
        evidence=SC["evidence"],
        checkpoints=SC["checkpoints"],
        branch=SC["branch"],
    run:
        land_tables(input, output, params, wildcards)


rule score:
    input:
        # The units are asked for themselves too: with mtime triggers alone
        # (the profiles'), a new label's units missing under tables already
        # built were "missing, no update", and ticks scored nothing new.
        unit_stamps,
        rules.score_table.output,
        lambda wc: followups(wc),
