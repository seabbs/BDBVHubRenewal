# Fine-tuning.
#
# One continuous run of scripts/finetune_stream.py from base Chronos-2,
# cut into one segment per checkpoint prefix: segment N resumes from the
# state kept with the prefix before and stops once n<N> is saved
# (`--until N`). Its inputs are the batches that hold the first N
# outbreaks of the seed list, one spare, and the held-out batches,
# gathered in <RUN>/sims/ from this machine's rounds, the rounds host's
# (REMOTE) or the sims-r<N> releases (`fetch_batch`).
#
# Targets: `finetune` (every prefix), or one checkpoint,
# <RUN>/train/n<N>/settings.json. Publishing is the publish module's.
#
# A machine without the rounds takes its batches from the rounds host
# (`sims_from: remote`, REMOTE and REMOTE_ROOT in config/local.env), as
# each round's manifest lists them, or from the sims-r<N> releases
# (`release`). A segment is planned only once its batches can be had: the
# training seeds count up to the first batch not yet listed, and nothing
# trains before the whole held-out round is (`available_batches`, listed
# afresh each tick). From the rounds host, its rounds' manifests and QC
# exclusions are mirrored in <RUN>/rounds/.


checkpoint available_batches:
    output:
        f"{RUN}/available-batches.txt",
    log:
        f"{RUN}/logs/available-batches.log",
    localrule: True
    params:
        repo=SIM["repo"],
        rounds=lambda wc: sorted(
            {FT["held_round"], *(round_of(s) for s in training_seeds())}
        ),
    run:
        list_available(input, output, params, wildcards)


rule fetch_batch:
    input:
        unpack(fetch_source),
    output:
        counts=f"{RUN}/sims/counts-{{seed}}.csv.gz",
        settings=f"{RUN}/sims/settings-{{seed}}.csv",
    log:
        f"{RUN}/logs/fetch-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        source=FT["sims_from"],
        repo=SIM["repo"],
        tag=lambda wc: f"sims-r{round_of(wc.seed)}",
        sims=lambda wc, output: Path(output.counts).parent,
        src=lambda wc, input: input.get("counts", ""),
        src_settings=lambda wc, input: input.get("settings", ""),
        there=lambda wc: hosts._at(REMOTE, f"{REMOTE_ROUNDS}/r{round_of(wc.seed)}/sims"),
    shell:
        # Hard links where the rounds are on this machine, a copy from the
        # rounds host, else the batch's assets of its round's release.
        """
        exec >{log} 2>&1
        if [ "{params.source}" = local ]; then
            ln -f {params.src} {output.counts} 2>/dev/null || cp {params.src} {output.counts}
            ln -f {params.src_settings} {output.settings} 2>/dev/null || cp {params.src_settings} {output.settings}
        elif [ "{params.source}" = remote ]; then
            rsync -a {params.there}/counts-{wildcards.seed}.csv.gz \
                {params.there}/settings-{wildcards.seed}.csv {params.sims}/
        else
            gh release download {params.tag} -R {params.repo} --clobber \
                -D {params.sims} -p counts-{wildcards.seed}.csv.gz \
                -p settings-{wildcards.seed}.csv
        fi
        gzip -t {output.counts}
        """


rule seed_list:
    output:
        f"{RUN}/seeds.txt",
    log:
        f"{RUN}/logs/seed-list.log",
    localrule: True
    run:
        Path(output[0]).write_text("".join(f"{s}\n" for s in training_seeds()))


rule real_validation:
    input:
        FT["delays"] or [],
    output:
        FT["real_validation"] or "no-real-validation",
    log:
        f"{RUN}/logs/real-validation.log",
    conda:
        TOOLS_ENV
    shell:
        "uv run python scripts/finetune_real_validation.py {input} {output} > {log} 2>&1"


rule train_segment:
    input:
        unpack(segment_inputs),
    output:
        settings=protected(f"{TRAIN}/n{{n}}/settings.json"),
        adapter=protected(f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors"),
    log:
        f"{TRAIN}/n{{n}}.log",
    # In one pool the trainer, the run's critical path, goes first: before
    # simulation (15) and checkpoint scoring (10 and below).
    priority: 20
    conda:
        TOOLS_ENV
    resources:
        gpu=1,
    params:
        prev_args=lambda wc: (
            f"--prev {TRAIN}/n{PREV[int(wc.n)]}/state " f"--prev-n {PREV[int(wc.n)]}"
            if PREV[int(wc.n)]
            else ""
        ),
        # The released exclusions of the rounds this segment trains on.
        release_args=lambda wc: (
            f"--release-rounds {release_rounds(int(wc.n))} --repo {SIM['repo']}"
            if FT["sims_from"] == "release"
            else ""
        ),
        # Where the QC exclusions are: this machine's rounds, or the
        # rounds host's mirrored here.
        rounds=MIRROR if FT["sims_from"] == "remote" else ROUNDS_DIR,
        held=lambda wc: f"{held_seeds()[0]} {held_seeds()[-1]}",
        prefixes=" ".join(str(p) for p in FT["prefixes"]),
        stop_from=FT["outbreaks"] + 1,
        blocks=FT["cuts"] * FT["draws"],
        sources=" ".join(FT["sources"]),
        real=(
            f"--real-validation {FT['real_validation']}"
            if FT["real_validation"]
            else ""
        ),
        bias=f"--series-bias --bias-lr {FT['bias_lr']}" if FT["series_bias"] else "",
        device=machine.device(LOCAL),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        watermark=LOCAL.get("MPS_HIGH_WATERMARK", "0.85"),
        # The run's folders, as functions: they hold the outputs.
        train=lambda wc: TRAIN,
        run=lambda wc: RUN,
        grid=FT["grid"],
        holdout=FT["holdout"],
        outbreaks=FT["outbreaks"],
        draws=FT["draws"],
        subset_seed=FT["subset_seed"],
        learning_rate=FT["learning_rate"],
        univariate_share=FT["univariate_share"],
        note=FT["note"],
        chunk=FT["chunk"],
        eval_every=FT["eval_every"],
        by_source_every=FT["by_source_every"],
        eval_mode=FT["eval"],
        # Files publishing reads beside the outputs, copied back with them
        # from a worker elsewhere (flowlib.compute).
        copy_back=lambda wc: [
            f"{TRAIN}/n{wc.n}/{name}"
            for name in (
                "outbreaks.txt",
                "consumed.txt",
                "finetuned-ckpt/adapter_config.json",
            )
        ],
    shell:
        # The run's own state resumes it; the state kept with the prefix
        # before replaces it if missing or behind (flowlib.segments).
        """
        uv run python workflow/scripts/prepare_segment.py {params.train} {params.run} \
            {params.rounds} {params.prev_args} {params.release_args} >{log}
        DEVICE={params.device} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 \
            PYTORCH_MPS_HIGH_WATERMARK_RATIO={params.watermark} \
            PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 MPLCONFIGDIR="${{TMPDIR:-/tmp}}" \
            uv run --group chronos python scripts/finetune_stream.py \
            {params.run}/sims {params.train} --grid {params.grid} \
            --seed-list {input.seeds} --held-out-seeds {params.held} \
            --holdout {params.holdout} --outbreaks {params.outbreaks} \
            --prefixes {params.prefixes} \
            --stop-from {params.stop_from} \
            --blocks-per-simulation {params.blocks} \
            --draws-per-simulation {params.draws} \
            --subset-seed {params.subset_seed} \
            --learning-rate {params.learning_rate} \
            --sources {params.sources} \
            --univariate-share {params.univariate_share} {params.real} \
            --note "{params.note}" --chunk {params.chunk} \
            --eval-every {params.eval_every} \
            --by-source-every {params.by_source_every} \
            --eval {params.eval_mode} \
            --exclude {params.run}/excluded.csv {params.bias} \
            --until {wildcards.n} >>{log} 2>&1
        # An optional notification line; it never fails the rule.
        uv run python workflow/scripts/segment_saved.py {output.settings} \
            {wildcards.n} || true
        """


rule checkpoint_check:
    # A report, never a gate: does checkpoint `check_at` beat zero-shot on
    # the held-out simulations and the BDBV pseudo-origins?
    input:
        f"{TRAIN}/n{CHECK}/settings.json",
    output:
        f"{RUN}/reports/n{CHECK}-check.txt",
    log:
        f"{RUN}/logs/n{CHECK}-check.log",
    localrule: True
    params:
        zero_shot=FT["zero_shot_real"],
    run:
        check_report(input, output, params, wildcards)


rule finetune:
    input:
        lambda wc: expand(f"{TRAIN}/n{{n}}/settings.json", n=to_train()),
        # A machine that publishes as it trains (PUBLISH in local.env).
        lambda wc: (
            expand(f"{TRAIN}/n{{n}}/published.json", n=to_train())
            if LOCAL["PUBLISH"]
            else []
        ),
        # The check once its checkpoint is asked for, so one segment a run
        # does not train on to it.
        lambda wc: [f"{RUN}/reports/n{CHECK}-check.txt"] if CHECK in to_train() else [],


rule dashboard:
    # A static render of the dashboard for the configured runs, for a
    # finished run or the record; `task dashboard` serves it live instead.
    output:
        f"{RUN}/reports/dashboard.html",
    log:
        f"{RUN}/logs/dashboard.log",
    localrule: True
    conda:
        TOOLS_ENV
    shell:
        "uv run --group paper python workflow/scripts/dashboard.py "
        "--once {output} > {log} 2>&1"
