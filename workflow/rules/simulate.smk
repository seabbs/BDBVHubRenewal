# Simulation rounds (was scripts/sim_rounds.sh).
#
# Round r is the batches of `flowlib.segments.round_seeds`, each
# scripts/finetune_simulate.R with exactly `draws` observation draws, into
# <ROOT>/rounds/r<r>/sims/. Each batch gets a manifest row, the round's
# manifest joins them, the sim QC (scripts/sim_qc.py) reviews the round,
# and `round_ready` writes the seed lists and READY last. Every
# simulation is kept: QC exclusions are a list (qc/excluded.csv).
#
# Targets: `simulate` (the rounds in `rounds`), and the opt-in `release`,
# which uploads each batch to the release sims-r<r>.

from flowlib import manifest, segments

SIM = config["simulate"]


def seeds_of(round_):
    """A round's training and held-out seeds."""
    return segments.round_seeds(
        int(round_),
        SIM["seed_base"],
        SIM["train_batches"],
        SIM["held_batches"],
    )


def all_seeds(round_):
    train, held = seeds_of(round_)
    return train + held


def kind_of(round_, seed):
    train, _ = seeds_of(round_)
    return "train" if int(seed) in train else "held-out"


def grid_name():
    """The grid the simulator names (`grid_name <- "..."`), if any."""
    text = Path("scripts/finetune_simulate.R").read_text()
    found = re.search(r'^grid_name <- "([^"]*)"', text, re.M)
    return found.group(1) if found else ""


wildcard_constraints:
    round=r"\d+",
    seed=r"\d+",
    n=r"\d+",


rule simulate_batch:
    output:
        counts=f"{ROUNDS_DIR}/r{{round}}/sims/counts-{{seed}}.csv.gz",
        settings=f"{ROUNDS_DIR}/r{{round}}/sims/settings-{{seed}}.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/sims/logs/{{seed}}.log",
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        batch=SIM["batch"],
        draws=SIM["draws"],
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        min_free=LOCAL["MIN_FREE_GB"],
        min_disk=LOCAL["MIN_DISK_GB"],
    retries: 1
    resources:
        mem_mb=lambda wc, attempt: (
            SIM["mem_mb"] if attempt == 1 else SIM["retry_mem_mb"]
        ),
    threads: 1
    shell:
        # The simulator must be the committed one, as the manifest records
        # the commit. A retry keeps the first attempt's log. R and Stan
        # write temporary files to a folder of the batch's own, removed
        # however the batch ends.
        """
        git diff --quiet HEAD -- scripts/finetune_simulate.R || {{
          echo "scripts/finetune_simulate.R differs from HEAD" > {log}
          exit 1
        }}
        workflow/scripts/wait_for_room.sh {params.min_free} {params.min_disk} \
          {params.sims}
        [ -s {log} ] && mv {log} {params.sims}/logs/{wildcards.seed}.first.log
        tmp={params.sims}/tmp/{wildcards.seed}
        rm -rf "$tmp" && mkdir -p "$tmp"
        trap 'rm -rf "$tmp"' EXIT
        TMPDIR="$tmp" ENW_DRAWS={params.draws} {params.prefix} \
          /usr/bin/time -f "peak_rss_kb %M wall_s %e" \
          nice -n 19 Rscript scripts/finetune_simulate.R \
          {params.batch} {wildcards.seed} {params.sims} > {log} 2>&1
        gzip -t {output.counts}
        """


rule batch_record:
    input:
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    output:
        f"{ROUNDS_DIR}/r{{round}}/records/record-{{seed}}.csv",
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        old=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    localrule: True
    run:
        import datetime as dt

        r, s = int(wildcards.round), int(wildcards.seed)
        line = manifest.existing(params.old, r, s) or manifest.row(
            params.sims,
            round_=r,
            seed=s,
            kind=kind_of(r, s),
            grid=grid_name(),
            commit=git_commit(),
            draws=SIM["draws"],
            batch=SIM["batch"],
            host=HOST,
            finished=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        Path(output[0]).write_text(line + "\n")


rule round_manifest:
    input:
        lambda wc: expand(
            f"{ROUNDS_DIR}/r{wc.round}/records/record-{{seed}}.csv",
            seed=all_seeds(wc.round),
        ),
    output:
        f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
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
    params:
        round_dir=f"{ROUNDS_DIR}/r{{round}}",
    shell:
        "uv run python scripts/sim_qc.py {params.round_dir} > {log} 2>&1"


rule round_ready:
    input:
        rules.sim_qc.output.summary,
    output:
        seeds=f"{ROUNDS_DIR}/r{{round}}/seeds.txt",
        held=f"{ROUNDS_DIR}/r{{round}}/held-out.txt",
        ready=f"{ROUNDS_DIR}/r{{round}}/READY",
    localrule: True
    run:
        import datetime as dt

        train, held = seeds_of(wildcards.round)
        Path(output.seeds).write_text("".join(f"{s}\n" for s in train))
        Path(output.held).write_text(f"{held[0]} {held[-1]}\n")
        Path(output.ready).write_text(
            dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n"
        )
        qc = Path(input[0]).read_text().strip()
        say(
            f"round {wildcards.round} ready for training: {len(train)} "
            f"training and {len(held)} held-out batches; {qc}"
        )


rule release_batch:
    # Opt-in (target `release`): uploads a batch to the release sims-r<r>,
    # so another machine can train on it before the round ends.
    input:
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
        record=rules.batch_record.output,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/{{seed}}.done"),
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
    localrule: True
    shell:
        """
        gh release view {params.tag} -R {params.repo} > /dev/null 2>&1 || \
          gh release create {params.tag} -R {params.repo} --prerelease \
            --title "Simulations, round {wildcards.round}" \
            --notes "Fine-tuning simulations of round {wildcards.round}, one batch per asset, with their manifest rows." \
            > /dev/null 2>&1 || \
          gh release view {params.tag} -R {params.repo} > /dev/null
        gh release upload {params.tag} -R {params.repo} --clobber \
          {input.counts} {input.settings} {input.record}
        """


rule simulate:
    input:
        expand(f"{ROUNDS_DIR}/r{{round}}/READY", round=config["rounds"]),


rule release:
    input:
        [
            f"{ROUNDS_DIR}/r{r}/released/{s}.done"
            for r in config["rounds"]
            for s in all_seeds(r)
        ],
