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

from flowlib import hosts, manifest, releases

FT = config["finetune"]
# Where batches come from is this machine's choice when set.
if LOCAL["SIMS_FROM"]:
    FT["sims_from"] = LOCAL["SIMS_FROM"]
RUN = LOCAL["RUN"]
TRAIN = f"{RUN}/train"

segments.check_chunks(FT["prefixes"], FT["chunk"], FT["outbreaks"])
PREV = dict((n, p) for p, n in segments.segments(FT["prefixes"]))


def round_of(seed):
    return (int(seed) - SIM["seed_base"]) // 100


def training_seeds():
    """The run's training seeds, in the order it trains on them."""
    if FT["seed_file"]:
        return [int(s) for s in Path(FT["seed_file"]).read_text().split()]
    return [s for r in FT["rounds"] for s in seeds_of(r)]


def held_seeds():
    return seeds_of(FT["held_round"])


ELSEWHERE = FT["sims_from"] in ("release", "remote")
# The rounds host's rounds mirrored here: manifests and QC exclusions.
MIRROR = f"{RUN}/rounds"


def available_seeds():
    """The training seeds whose batches can be had now, in order.

    All of them from this machine's rounds, which the workflow simulates.
    From the rounds host or releases, those before the first batch not
    yet listed, and none until the held-out round is all listed.
    """
    order = training_seeds()
    if not ELSEWHERE:
        return order
    path = checkpoints.available_batches.get().output[0]
    listed = {int(s) for s in Path(path).read_text().split()}
    if not set(held_seeds()) <= listed:
        return []
    for i, seed in enumerate(order):
        if seed not in listed:
            return order[:i]
    return order


checkpoint available_batches:
    output:
        f"{RUN}/available-batches.txt",
    params:
        repo=SIM["repo"],
        rounds=lambda wc: sorted(
            {FT["held_round"], *(round_of(s) for s in training_seeds())}
        ),
    localrule: True
    run:
        import json

        seeds = set()
        for r in params.rounds:
            if FT["sims_from"] == "remote":
                # The manifest and QC exclusions of a round there so far.
                here = Path(f"{MIRROR}/r{r}")
                (here / "qc").mkdir(parents=True, exist_ok=True)
                for name in ("manifest.csv", "qc/excluded.csv"):
                    there = f"{REMOTE_ROUNDS}/r{r}/{name}"
                    got = subprocess.run(
                        ["rsync", "-a", hosts._at(REMOTE, there), str(here / name)],
                        capture_output=True,
                        text=True,
                    )  # fmt: skip
                    # A round with no QC yet has no exclusions; a round not
                    # started has no manifest, and nor has an unreachable
                    # host, so that is said (the mirror kept so far stays).
                    if got.returncode and name == "manifest.csv":
                        print(
                            f"could not copy {there}: {got.stderr.strip()[:200]}",
                            file=sys.stderr,
                        )
                seeds |= manifest.seeds(here / "manifest.csv")
                continue
            # A round not released yet has no release.
            got = subprocess.run(
                ["gh", "release", "view", f"sims-r{r}", "-R", params.repo,
                 "--json", "assets"],
                capture_output=True,
                text=True,
            )  # fmt: skip
            if got.returncode == 0:
                names = [a["name"] for a in json.loads(got.stdout)["assets"]]
                seeds |= releases.released_seeds(names)
        Path(output[0]).write_text("".join(f"{s}\n" for s in sorted(seeds)))


def known_exclusions():
    """The sim-QC exclusions known now."""
    paths = [Path(f"{RUN}/excluded.csv")]
    paths += Path(ROUNDS_DIR).glob("r*/qc/excluded.csv")
    paths += Path(MIRROR).glob("r*/qc/excluded.csv")
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
    order = available_seeds()
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
        source=FT["sims_from"],
        repo=SIM["repo"],
        tag=lambda wc: f"sims-r{round_of(wc.seed)}",
        sims=f"{RUN}/sims",
        src=lambda wc, input: input.get("counts", ""),
        src_settings=lambda wc, input: input.get("settings", ""),
        there=lambda wc: hosts._at(
            REMOTE, f"{REMOTE_ROUNDS}/r{round_of(wc.seed)}/sims"
        ),
    localrule: True
    shell:
        # Hard links where the rounds are on this machine, a copy from the
        # rounds host, else the batch's assets of its round's release.
        """
        if [ "{params.source}" = local ]; then
          ln -f {params.src} {output.counts} 2> /dev/null || cp {params.src} {output.counts}
          ln -f {params.src_settings} {output.settings} 2> /dev/null || cp {params.src_settings} {output.settings}
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
        # Where the QC exclusions are: this machine's rounds, or the
        # rounds host's mirrored here.
        rounds=MIRROR if FT["sims_from"] == "remote" else ROUNDS_DIR,
        held=lambda wc: f"{held_seeds()[0]} {held_seeds()[-1]}",
        prefixes=" ".join(str(p) for p in FT["prefixes"]),
        stop_from=FT["outbreaks"] + 1,
        blocks=FT["cuts"] * FT["draws"],
        sources=" ".join(FT["sources"]),
        real=f"--real-validation {FT['real_validation']}"
        if FT["real_validation"]
        else "",
        bias=f"--series-bias --bias-lr {FT['bias_lr']}"
        if FT["series_bias"]
        else "",
        device=machine.device(LOCAL),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        watermark=LOCAL.get("MPS_HIGH_WATERMARK", "0.85"),
    resources:
        gpu=1,
    # In one pool the trainer, the run's critical path, goes first: before
    # simulation (15) and checkpoint scoring (10 and below).
    priority: 20
    shell:
        # The run's own state resumes it; the state kept with the prefix
        # before replaces it if missing or behind (flowlib.segments).
        """
        uv run python workflow/scripts/prepare_segment.py {TRAIN} {RUN} \
          {params.rounds} {params.prev_args} {params.release_args} > {log}
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
          --learning-rate {FT[learning_rate]} \
          --sources {params.sources} \
          --univariate-share {FT[univariate_share]} {params.real} \
          --note "{FT[note]}" --chunk {FT[chunk]} \
          --exclude {RUN}/excluded.csv {params.bias} \
          --until {wildcards.n} >> {log} 2>&1
        # An optional notification line; it never fails the rule.
        uv run python workflow/scripts/segment_saved.py {output.settings} \
          {wildcards.n} || true
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
    order = available_seeds()
    # A spare batch past each prefix but the run's last, which takes
    # every training simulation.
    return [
        n
        for n in FT["prefixes"]
        if Path(f"{TRAIN}/n{n}/settings.json").exists()
        or segments.reachable(
            n + SIM["batch"] * (n < FT["outbreaks"]),
            order,
            lambda s: kept_or_full(s, skip),
        )
    ]


def to_train():
    """Prefixes `finetune` asks for."""
    # The no-bias control is another run: it must not train into the slots
    # run's folder (RUN's default).
    if not FT["series_bias"] and Path(RUN) == Path(LOCAL["ROOT"], "checkpoints/slots"):
        raise ValueError(
            f"a no-bias run (series_bias false) needs its own RUN, not {RUN}"
        )
    prefixes = trainable()
    if not config.get("next_segment_only"):
        return prefixes
    saved = [n for n in prefixes if Path(f"{TRAIN}/n{n}/settings.json").exists()]
    rest = [n for n in prefixes if n not in saved]
    return saved + rest[:1]


rule finetune:
    input:
        lambda wc: expand(f"{TRAIN}/n{{n}}/settings.json", n=to_train()),
        # A machine that publishes as it trains (PUBLISH in local.env).
        lambda wc: expand(f"{TRAIN}/n{{n}}/published.json", n=to_train())
        if LOCAL["PUBLISH"]
        else [],
        # The check once its checkpoint is asked for, so one segment a run
        # does not train on to it.
        lambda wc: [f"{RUN}/reports/n{CHECK}-check.txt"]
        if CHECK in to_train()
        else [],


rule dashboard:
    # A static render of the dashboard for the configured runs, for a
    # finished run or the record; `task dashboard` serves it live instead.
    output:
        f"{RUN}/reports/dashboard.html",
    localrule: True
    shell:
        "uv run --group paper python workflow/scripts/dashboard.py "
        "--once {output}"
