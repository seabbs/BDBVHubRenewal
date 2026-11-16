# Fine-tuning.
#
# One continuous run of scripts/finetune_stream.py from base Chronos-2,
# cut into one segment per checkpoint prefix: segment N resumes from the
# state kept with the prefix before and stops once n<N> is saved
# (`--until N`). Its inputs are the batches that hold the first N
# outbreaks of the seed list, one spare, and the held-out batches,
# gathered in <RUN>/sims/ from this machine's rounds or the sims-r<N>
# releases (`fetch_batch`).
#
# Targets: `finetune` (every prefix), or one checkpoint,
# <RUN>/train/n<N>/settings.json. Publishing is the publish module's.

FT = config["finetune"]
# Where batches come from is this machine's choice when set.
if LOCAL["SIMS_FROM"]:
    FT["sims_from"] = LOCAL["SIMS_FROM"]
RUN = LOCAL["RUN"]
TRAIN = f"{RUN}/train"

segments.check_chunks(FT["prefixes"], FT["chunk"])
PREV = dict((n, p) for p, n in segments.segments(FT["prefixes"]))


def round_of(seed):
    return (int(seed) - SIM["seed_base"]) // 100


def training_seeds():
    """The run's training seeds, in the order it trains on them."""
    if FT["seed_file"]:
        return [int(s) for s in Path(FT["seed_file"]).read_text().split()]
    return [s for r in FT["rounds"] for s in seeds_of(r)[0]]


def held_seeds():
    return seeds_of(FT["held_round"])[1]


def known_exclusions():
    """The sim-QC exclusions known now."""
    paths = [Path(f"{RUN}/excluded.csv")]
    paths += Path(ROUNDS_DIR).glob("r*/qc/excluded.csv")
    paths += Path(f"{RUN}/excluded-releases").glob("*.csv")
    return set().union(*(segments.excluded(p) for p in paths))


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
    order = training_seeds()
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
        "sims": expand(
            f"{RUN}/sims/settings-{{seed}}.csv", seed=segment_seeds(n)
        ),
        "seeds": f"{RUN}/seeds.txt",
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


rule fetch_batch:
    input:
        unpack(fetch_source),
    output:
        counts=f"{RUN}/sims/counts-{{seed}}.csv.gz",
        settings=f"{RUN}/sims/settings-{{seed}}.csv",
    params:
        local=FT["sims_from"] == "local",
        repo=SIM["repo"],
        tag=lambda wc: f"sims-r{round_of(wc.seed)}",
        sims=f"{RUN}/sims",
        src=lambda wc, input: input.get("counts", ""),
        src_settings=lambda wc, input: input.get("settings", ""),
    localrule: True
    shell:
        # Hard links where the rounds are on this machine, else the batch's
        # assets of its round's release.
        """
        if [ "{params.local}" = True ]; then
          ln -f {params.src} {output.counts} 2> /dev/null || cp {params.src} {output.counts}
          ln -f {params.src_settings} {output.settings} 2> /dev/null || cp {params.src_settings} {output.settings}
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
    localrule: True
    run:
        Path(output[0]).write_text("".join(f"{s}\n" for s in training_seeds()))


rule real_validation:
    input:
        FT["delays"] or [],
    output:
        FT["real_validation"] or "no-real-validation",
    shell:
        "uv run python scripts/finetune_real_validation.py {input} {output}"


rule train_segment:
    input:
        unpack(segment_inputs),
    output:
        settings=protected(f"{TRAIN}/n{{n}}/settings.json"),
        adapter=protected(
            f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors"
        ),
    log:
        f"{TRAIN}/n{{n}}.log",
    params:
        prev_args=lambda wc: (
            f"--prev {TRAIN}/n{PREV[int(wc.n)]}/state "
            f"--prev-n {PREV[int(wc.n)]}"
            if PREV[int(wc.n)]
            else ""
        ),
        # The released exclusions of the rounds this segment trains on.
        release_args=lambda wc: (
            "--release-rounds "
            + " ".join(
                str(r)
                for r in sorted({round_of(s) for s in segment_seeds(int(wc.n))})
            )
            + f" --repo {SIM['repo']}"
            if FT["sims_from"] == "release"
            else ""
        ),
        held=lambda wc: f"{held_seeds()[0]} {held_seeds()[-1]}",
        prefixes=" ".join(str(p) for p in FT["prefixes"]),
        stop_from=FT["outbreaks"] + 1,
        blocks=16 * FT["draws"],
        sources=" ".join(FT["sources"]),
        real=f"--real-validation {FT['real_validation']}"
        if FT["real_validation"]
        else "",
        slots=f"--series-slots {FT['series_slots']}"
        if FT["series_slots"]
        else "",
        device=LOCAL.get("DEVICE") or os.environ.get("DEVICE", "auto"),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        watermark=LOCAL.get("MPS_HIGH_WATERMARK", "0.85"),
    resources:
        gpu=1,
    shell:
        # The run's own state resumes it; the state kept with the prefix
        # before replaces it if missing or behind (flowlib.segments).
        """
        uv run python workflow/scripts/prepare_segment.py {TRAIN} {RUN} \
          {ROUNDS_DIR} {params.prev_args} {params.release_args} > {log}
        DEVICE={params.device} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 \
        PYTORCH_MPS_HIGH_WATERMARK_RATIO={params.watermark} \
        PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 MPLCONFIGDIR="${{TMPDIR:-/tmp}}" \
        uv run --group chronos python scripts/finetune_stream.py \
          {RUN}/sims {TRAIN} --grid {FT[grid]} --seed-list {input.seeds} \
          --held-out-seeds {params.held} --holdout {FT[holdout]} \
          --outbreaks {FT[outbreaks]} --prefixes {params.prefixes} \
          --stop-from {params.stop_from} \
          --blocks-per-simulation {params.blocks} \
          --draws-per-simulation {FT[draws]} --subset-seed {FT[subset_seed]} \
          --learning-rate {FT[learning_rate]} --lr-steps {FT[lr_steps]} \
          --sources {params.sources} \
          --univariate-share {FT[univariate_share]} {params.real} \
          --note "{FT[note]}" --chunk {FT[chunk]} \
          --exclude {RUN}/excluded.csv {params.slots} \
          --until {wildcards.n} >> {log} 2>&1
        uv run python workflow/scripts/segment_saved.py {output.settings} \
          {wildcards.n}
        """


CHECK = FT.get("check_at", 1000)
AFTER = [n for n in sorted(FT["prefixes"]) if n > CHECK]


rule checkpoint_check:
    # A report, never a gate: does checkpoint `check_at` beat zero-shot on
    # the held-out simulations and the BDBV pseudo-origins?
    input:
        f"{TRAIN}/n{CHECK}/settings.json",
    output:
        f"{RUN}/reports/n{CHECK}-check.txt",
    params:
        zero_shot=FT["zero_shot_real"],
    localrule: True
    run:
        import json

        s = json.loads(Path(input[0]).read_text())
        real = s.get("real_validation_loss", float("inf"))
        held = s["held_out_loss"] < s["held_out_loss_start"]
        ok = held and real < params.zero_shot
        text = (
            f"n{CHECK} {'beats' if ok else 'does NOT beat'} zero-shot: "
            f"held-out {s['held_out_loss']:.4f} vs "
            f"{s['held_out_loss_start']:.4f}, real {real:.4f} vs "
            f"{params.zero_shot}"
        )
        Path(output[0]).write_text(text + "\n")
        nxt = f", check before n{AFTER[0]}" if AFTER else ""
        say(text if ok else f"ALERT {text}; training goes on{nxt}")


def trainable():
    """The prefixes done or within reach of the seed list."""
    skip = known_exclusions()
    order = training_seeds()
    return [
        n
        for n in FT["prefixes"]
        if Path(f"{TRAIN}/n{n}/settings.json").exists()
        or segments.reachable(
            n + SIM["batch"], order, lambda s: kept_or_full(s, skip)
        )
    ]


rule finetune:
    input:
        lambda wc: expand(f"{TRAIN}/n{{n}}/settings.json", n=trainable()),
        # A machine that publishes as it trains (PUBLISH in local.env).
        lambda wc: expand(f"{TRAIN}/n{{n}}/published.json", n=trainable())
        if LOCAL["PUBLISH"]
        else [],
        [f"{RUN}/reports/n{CHECK}-check.txt"] if CHECK in FT["prefixes"] else [],
