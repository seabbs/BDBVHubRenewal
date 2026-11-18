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


rule bdbv_data:
    # The hub's data (`bdbvhubrenewal data`). A refresh (`task data`) makes
    # the BDBV units rerun, and they add only the new origins.
    output:
        SC["data"],
    localrule: True
    shell:
        "uv run bdbvhubrenewal data"


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
        cache=lambda wc: CACHE,
    shell:
        # Rows of other rules in the committed tables are kept.
        """
        [ -f {params.evidence} ] && cp {params.evidence} {output.evidence}
        [ -f {params.checkpoints} ] && cp {params.checkpoints} {output.checkpoints}
        uv run python scripts/checkpoint_scores.py table {params.cache} \
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
        land_tables(input, output, params, wildcards)


rule germany_forecast:
    # Opt-in target: the German forecast units alone.
    input:
        forecast_inputs,


rule score:
    input:
        rules.score_table.output,
        lambda wc: followups(wc),
