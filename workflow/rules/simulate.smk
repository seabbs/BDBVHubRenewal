# Simulation rounds.
#
# Round r is the batches of `flowlib.segments.round_seeds`, each
# scripts/finetune_simulate.R with exactly `draws` observation draws, and
# is grid block r: 40 batches of 21 are its 840 simulations. Into
# <ROOT>/rounds/r<r>/sims/. Each batch gets a manifest row, the round's
# manifest joins them, the sim QC (scripts/sim_qc.py) reviews the round,
# and `round_ready` writes the seed lists and READY last. Every
# simulation is kept: QC exclusions are a list (qc/excluded.csv).
#
# Targets: `simulate` (the rounds in `rounds`), and the opt-in `release`,
# which uploads each batch to the release sims-r<r>.
#
# Two hosts can share the rounds (flowlib.hosts): each lists the seeds
# it takes in its local config (SIM_SEEDS; empty: all). A helping
# host's target is `push`, which sends each of its batches to the rounds
# host (REMOTE) with its manifest row; the rounds host's `simulate`
# simulates its own seeds and waits for the rest.


wildcard_constraints:
    round=r"\d+",
    seed=r"\d+",
    n=r"\d+",


def batch_priority(wc):
    """A batch's priority: 15, earlier seeds a fraction higher."""
    return segments.batch_priority(int(wc.seed), 15, SIM["seed_base"])


rule simulate_batch:
    output:
        counts=f"{ROUNDS_DIR}/r{{round}}/sims/counts-{{seed}}.csv.gz",
        settings=f"{ROUNDS_DIR}/r{{round}}/sims/settings-{{seed}}.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/sims/logs/{{seed}}.log",
    # In one pool, after the trainer (20) and before scoring (10 and
    # below): the trainer waits on these batches. Among themselves in
    # seed order, the order training reads them, so a run over several
    # rounds finishes each round before the next.
    priority: batch_priority
    retries: 1
    conda:
        TOOLS_ENV
    threads: 1
    resources:
        mem_mb=lambda wc, attempt: (
            SIM["mem_mb"] if attempt == 1 else SIM["retry_mem_mb"]
        ),
        # At most `--resources sim_slots=N` batches at once (2 by default).
        sim_slots=1,
    params:
        sims=lambda wc, output: Path(output.counts).parent,
        batch=SIM["batch"],
        draws=SIM["draws"],
        seed_base=SIM["seed_base"],
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        min_free=LOCAL["MIN_FREE_GB"],
        min_disk=LOCAL["MIN_DISK_GB"],
    shell:
        # The simulator must be the committed one, as the manifest records
        # the commit. A retry keeps the first attempt's log. R and Stan
        # write temporary files to a folder of the batch's own, removed
        # however the batch ends.
        """
        if ! git diff --quiet HEAD -- scripts/finetune_simulate.R; then
            echo "scripts/finetune_simulate.R differs from HEAD" >{log}
            exit 1
        fi
        workflow/scripts/wait_for_room.sh {params.min_free} {params.min_disk} \
            {params.sims}
        [ -s {log} ] && mv {log} {params.sims}/logs/{wildcards.seed}.first.log
        tmp={params.sims}/tmp/{wildcards.seed}
        rm -rf "$tmp" && mkdir -p "$tmp"
        trap 'rm -rf "$tmp"' EXIT
        TMPDIR="$tmp" ENW_DRAWS={params.draws} \
            ENW_SEED_BASE={params.seed_base} {params.prefix} \
            workflow/scripts/timed.sh \
            nice -n 19 Rscript scripts/finetune_simulate.R \
            {params.batch} {wildcards.seed} {params.sims} >{log} 2>&1
        gzip -t {output.counts}
        """


rule await_batch:
    # A batch another host simulates (not in SIM_SEEDS): done once that
    # host has pushed its files and then its row to this round's
    # manifest. One waits at a time (`--resources awaiting=1`).
    output:
        f"{ROUNDS_DIR}/r{{round}}/arrived/{{seed}}.done",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/await-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    resources:
        awaiting=1,
    params:
        manifest=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
        poll=60,
    shell:
        """
        exec >{log} 2>&1
        until grep -q '^{wildcards.round},{wildcards.seed},' \
            {params.manifest} 2>/dev/null; do
            sleep {params.poll}
        done
        touch {output}
        """


rule batch_record:
    input:
        unpack(batch_files),
    output:
        f"{ROUNDS_DIR}/r{{round}}/records/record-{{seed}}.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/record-{{seed}}.log",
    localrule: True
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        old=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    run:
        write_record(input, output, params, wildcards)


rule round_manifest:
    input:
        lambda wc: expand(
            f"{ROUNDS_DIR}/r{wc.round}/records/record-{{seed}}.csv",
            seed=seeds_of(wc.round),
        ),
    output:
        f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/manifest.log",
    localrule: True
    run:
        manifest.join(input, output[0])


rule sim_qc:
    input:
        rules.round_manifest.output,
    output:
        excluded=f"{ROUNDS_DIR}/r{{round}}/qc/excluded.csv",
        summary=f"{ROUNDS_DIR}/r{{round}}/qc/summary.txt",
    log:
        f"{ROUNDS_DIR}/r{{round}}/qc/run.log",
    conda:
        TOOLS_ENV
    params:
        round_dir=lambda wc, input: Path(input[0]).parent,
    shell:
        "uv run python scripts/sim_qc.py {params.round_dir} > {log} 2>&1"


rule round_ready:
    input:
        rules.sim_qc.output.summary,
    output:
        seeds=f"{ROUNDS_DIR}/r{{round}}/seeds.txt",
        ready=f"{ROUNDS_DIR}/r{{round}}/READY",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/ready.log",
    localrule: True
    run:
        mark_ready(input, output, params, wildcards)


# Refuses to upload into a release of another grid's batches: its notes
# name the grid (`Grid r6.`), and unmarked releases are r5's
# (flowlib.grids.release_grid).
GRID_GUARD = """
        notes=$(gh release view {params.tag} -R {params.repo} --json body -q .body)
        found=$(printf '%s\\n' "$notes" | sed -n 's/.*Grid \\(r[0-9][0-9]*\\)\\..*/\\1/p' | head -n 1)
        if [ "${{found:-r5}}" != "{params.grid}" ]; then
            echo "{params.tag} holds grid ${{found:-r5}}'s batches, not {params.grid}'s: not uploading"
            exit 1
        fi
"""


rule release_batch:
    # Opt-in (target `release`): uploads a batch to the release sims-r<r>,
    # so another machine can train on it before the round ends.
    input:
        unpack(batch_files),
        record=rules.batch_record.output,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/release-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        grid=grid_name(),
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    shell:
        """
        exec >{log} 2>&1
        gh release view {params.tag} -R {params.repo} >/dev/null 2>&1 \
            || gh release create {params.tag} -R {params.repo} --prerelease \
                --title "Simulations, round {wildcards.round}" \
                --notes "Fine-tuning simulations of round {wildcards.round}, one batch per asset, with their manifest rows. Grid {params.grid}." \
                >/dev/null 2>&1 \
            || gh release view {params.tag} -R {params.repo} >/dev/null
        """
        + GRID_GUARD
        + """
        gh release upload {params.tag} -R {params.repo} --clobber \
            {params.counts} {params.settings} {input.record}
        """


rule release_qc:
    # Opt-in (target `release`): the round's QC exclusions, so a machine
    # training from the releases skips them too.
    input:
        rules.sim_qc.output.excluded,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/qc.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/release-qc.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        named=f"{ROUNDS_DIR}/r{{round}}/qc/excluded-r{{round}}.csv",
        grid=grid_name(),
    shell:
        """
        exec >{log} 2>&1
        """
        + GRID_GUARD
        + """
        cp {input} {params.named}
        gh release upload {params.tag} -R {params.repo} --clobber {params.named}
        """


rule push_batch:
    # A helping host's batch, sent to the rounds host's round: its files,
    # then its manifest row (flowlib.hosts.push).
    input:
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/sent/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/push-{{seed}}.log",
    retries: 3
    localrule: True
    params:
        sims=lambda wc, input: str(Path(input.counts).parent),
        there=lambda wc: f"{REMOTE_ROUNDS}/r{wc.round}",
    run:
        push_record(input, output, params, wildcards)


rule push:
    # A helping host's target: its seeds (SIM_SEEDS) of the rounds in
    # `rounds`.
    input:
        [
            f"{ROUNDS_DIR}/r{r}/sent/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
            if here(s)
        ],


rule simulate:
    input:
        expand(f"{ROUNDS_DIR}/r{{round}}/READY", round=config["rounds"]),


rule release:
    input:
        # Never another grid's round: its release holds that grid's batches.
        lambda wc: grids.check_release(
            config["rounds"], grid_name(), config["grid_rounds"]
        )
        or [],
        [
            f"{ROUNDS_DIR}/r{r}/released/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
        ],
        expand(f"{ROUNDS_DIR}/r{{r}}/released/qc.done", r=config["rounds"]),
