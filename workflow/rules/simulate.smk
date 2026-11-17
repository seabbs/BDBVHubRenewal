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
# Two hosts can share the rounds (flowlib.hosts): `simulate.hosts` names
# the seeds a helping host simulates. That host's target is `push`,
# which sends each of its batches to the rounds host (REMOTE) with its
# manifest row; the rounds host's `simulate` waits for them.

from flowlib import hosts, manifest, segments

SIM = config["simulate"]
SPLIT = SIM.get("hosts") or {}
hosts.check(SPLIT)
REMOTE = LOCAL["REMOTE"]
REMOTE_ROUNDS = (
    LOCAL["REMOTE_ROOT"] or ("bdbv-rounds" if REMOTE else LOCAL["ROOT"])
) + "/rounds"


def seeds_of(round_):
    """A round's batch seeds, its grid block."""
    return segments.round_seeds(
        int(round_), SIM["seed_base"], SIM["batches"]
    )


def here(seed):
    """Whether this host simulates batch `seed`."""
    return hosts.simulates(int(seed), HOST, SPLIT)


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
        seed_base=SIM["seed_base"],
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        min_free=LOCAL["MIN_FREE_GB"],
        min_disk=LOCAL["MIN_DISK_GB"],
    retries: 1
    resources:
        mem_mb=lambda wc, attempt: (
            SIM["mem_mb"] if attempt == 1 else SIM["retry_mem_mb"]
        ),
        # At most `--resources sim_slots=N` batches at once (2 by default).
        sim_slots=1,
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
        TMPDIR="$tmp" ENW_DRAWS={params.draws} \
          ENW_SEED_BASE={params.seed_base} {params.prefix} \
          workflow/scripts/timed.sh \
          nice -n 19 Rscript scripts/finetune_simulate.R \
          {params.batch} {wildcards.seed} {params.sims} > {log} 2>&1
        gzip -t {output.counts}
        """


rule await_batch:
    # A batch another host simulates (`simulate.hosts`): done once that
    # host has pushed its files and then its row to this round's
    # manifest. One waits at a time (`--resources awaiting=1`).
    output:
        f"{ROUNDS_DIR}/r{{round}}/arrived/{{seed}}.done",
    params:
        manifest=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
        poll=60,
    resources:
        awaiting=1,
    localrule: True
    shell:
        """
        until grep -q '^{wildcards.round},{wildcards.seed},' \
            {params.manifest} 2> /dev/null; do
          sleep {params.poll}
        done
        touch {output}
        """


rule batch_record:
    input:
        unpack(batch_files),
    output:
        f"{ROUNDS_DIR}/r{{round}}/records/record-{{seed}}.csv",
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        old=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    localrule: True
    run:
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
        Path(output[0]).write_text(line + "\n")


rule round_manifest:
    input:
        lambda wc: expand(
            f"{ROUNDS_DIR}/r{wc.round}/records/record-{{seed}}.csv",
            seed=seeds_of(wc.round),
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
        ready=f"{ROUNDS_DIR}/r{{round}}/READY",
    localrule: True
    run:
        import datetime as dt

        seeds = seeds_of(wildcards.round)
        Path(output.seeds).write_text("".join(f"{s}\n" for s in seeds))
        Path(output.ready).write_text(
            dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n"
        )
        qc = Path(input[0]).read_text().strip()
        say(
            f"round {wildcards.round} ready: {len(seeds)} batches, one grid "
            f"block; {qc}"
        )


rule release_batch:
    # Opt-in (target `release`): uploads a batch to the release sims-r<r>,
    # so another machine can train on it before the round ends.
    input:
        unpack(batch_files),
        record=rules.batch_record.output,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/{{seed}}.done"),
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
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
          {params.counts} {params.settings} {input.record}
        """


rule release_qc:
    # Opt-in (target `release`): the round's QC exclusions, so a machine
    # training from the releases skips them too.
    input:
        rules.sim_qc.output.excluded,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/qc.done"),
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        named=f"{ROUNDS_DIR}/r{{round}}/qc/excluded-r{{round}}.csv",
    localrule: True
    shell:
        """
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
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        there=f"{REMOTE_ROUNDS}/r{{round}}",
    retries: 3
    localrule: True
    run:
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


rule push:
    # A helping host's target: its seeds of the rounds in `rounds`.
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
        [
            f"{ROUNDS_DIR}/r{r}/released/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
        ],
        expand(f"{ROUNDS_DIR}/r{{r}}/released/qc.done", r=config["rounds"]),
