# The slots checkpoints' follow-ups, each after that checkpoint's main
# scores.
#
# Targets: part of `score`; one checkpoint's slots check,
# <SCORE_CACHE>/slots-check/<label>/summary.csv, or aggregation rows,
# <SCORE_CACHE>/agg/<label>/aggregation-scores.csv.gz.

SLC = config["slots_check"]
AGG = config["agg"]
AGG_ROOT = f"{CACHE}/agg"
HEAVY = (
    ROOM + "{params.prefix} nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
)


def slots_labels():
    return [lb for lb in checkpoint_labels() if lb.startswith("slots-")]


def main_scores(wc):
    return [
        f"{CACHE}/bdbv/{wc.label}.done",
        f"{CACHE}/germany/{wc.label}.done",
    ]


wildcard_constraints:
    label=r"slots-[0-9a-f]+|[0-9a-f]+|reference[a-z-]*",
    shard=r"\d+",


rule fetch_adapter:
    # A slots checkpoint's adapter from its release, checked against its
    # sha256sums.txt.
    output:
        directory(f"{CACHE}/adapters/{{label}}"),
    localrule: True
    params:
        repo=SIM["repo"],
        tag=lambda wc: "chronos-slots-" + wc.label.removeprefix("slots-"),
    shell:
        """
        gh release download {params.tag} -R {params.repo} -D {output} \
            -p adapter_config.json -p adapter_model.safetensors \
            -p settings.json -p sha256sums.txt
        cd {output} && grep -E 'adapter_(config|model)' sha256sums.txt \
            | sha256sum -c --quiet
        """


rule slots_check_shard:
    input:
        main_scores,
        adapter=rules.fetch_adapter.output,
    output:
        csv=f"{CACHE}/slots-check/{{label}}/shard-{{shard}}.csv",
    log:
        f"{CACHE}/slots-check/{{label}}/shard-{{shard}}.log",
    threads: 1
    resources:
        mem_mb=SLC["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        shards=SLC["shards"],
        blocks=f"{CACHE}/slots-check/blocks.pkl",
        held=lambda wc: f"{held_seeds()[0]} {held_seeds()[-1]}",
        sims=f"{ROUNDS_DIR}/r{FT['held_round']}/sims",
        noslots=LOCAL["NOSLOTS_RUN"],
        germany=LOCAL["GERMANY_BLOCKS"],
    shell:
        HEAVY + "uv run --group chronos python -u scripts/slots_check.py score "
        "{output.csv} --slots {input.adapter} --label {wildcards.label} "
        "--shard {wildcards.shard} {params.shards} --cache {params.blocks} "
        "--noslots-root {params.noslots} --sim-dir {params.sims} "
        "--held {params.held} --germany {params.germany} > {log} 2>&1"


rule slots_check_summary:
    input:
        lambda wc: expand(
            f"{CACHE}/slots-check/{wc.label}/shard-{{i}}.csv",
            i=range(SLC["shards"]),
        ),
    output:
        csv=f"{CACHE}/slots-check/{{label}}/summary.csv",
        blocks=f"{CACHE}/slots-check/{{label}}/blocks.csv.gz",
    localrule: True
    params:
        run=f"{CACHE}/slots-check/{{label}}",
    shell:
        "uv run python scripts/slots_check.py summarise {params.run} "
        "--label {wildcards.label} --commit $(git rev-parse HEAD) "
        "--csv {output.csv} --blocks {output.blocks}"


rule agg_data:
    # The data the zero-shot rows used, pinned, fetched once.
    output:
        target=f"{AGG_ROOT}/data/target.csv",
        germany=f"{AGG_ROOT}/germany/COVID-19_hospitalizations.csv",
        zero_shot=f"{AGG_ROOT}/zero-shot-scores.csv.gz",
        existing=f"{AGG_ROOT}/existing-scores.csv.gz",
    localrule: True
    params:
        ref=AGG["hub_ref"],
        zero_shot=AGG["zero_shot"],
        existing=AGG["existing"],
    shell:
        """
        export GITHUB_TOKEN=${{GITHUB_TOKEN:-$(gh auth token)}}
        uv run python -m bdbvhubrenewal.cli data \
            --raw {AGG_ROOT}/data/raw --out {AGG_ROOT}/data --ref {params.ref}
        uv run python -m bdbvhubrenewal.cli germany-fetch \
            --out {AGG_ROOT}/germany
        git show {params.zero_shot} >{output.zero_shot}
        git show {params.existing} >{output.existing}
        """


rule agg_sanity:
    # A zero-shot arm reproduces its existing rows, once per package
    # version (as the driver keyed it, by the tree of src/).
    input:
        target=rules.agg_data.output.target,
        zero_shot=rules.agg_data.output.zero_shot,
    output:
        f"{AGG_ROOT}/sanity/{{code}}/result.txt",
    resources:
        mem_mb=AGG["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        sane=f"{AGG_ROOT}/sanity/{{code}}",
    shell:
        """
        run() {{ """
        + HEAVY
        + """uv run --group chronos python scripts/agg/finetuned.py "$@"; }}
        run bdbv-fit --data {AGG_ROOT}/data --model fm_chronos_daily_ragged \
            --groupings joint --origins 2026-10-17 2026-11-01 \
            --out {params.sane}/fits
        run bdbv-score --data {AGG_ROOT}/data --fits {params.sane}/fits \
            --out {params.sane}/scores.csv
        run sanity --existing {input.zero_shot} \
            --scores {params.sane}/scores.csv --out {params.sane}/compare.csv \
            --tolerance 0 >{output}
        """


def src_tree():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD:src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


rule agg_fit:
    input:
        main_scores,
        data=rules.agg_data.output,
        sanity=lambda wc: f"{AGG_ROOT}/sanity/{src_tree()}/result.txt",
    output:
        f"{AGG_ROOT}/{{label}}/fits-{{model}}.done",
    log:
        f"{AGG_ROOT}/{{label}}/fits-{{model}}.log",
    resources:
        mem_mb=AGG["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        cache=f"{AGG_ROOT}/{{label}}",
        origins=" ".join(AGG["origins"]),
        dates=f"--dates {' '.join(AGG['dates'])}" if AGG["dates"] else "",
    shell:
        """
        export BDBV_CHRONOS_FT={wildcards.label}
        run() {{ """
        + HEAVY
        + """uv run --group chronos python scripts/agg/finetuned.py "$@"; }}
        run bdbv-fit --data {AGG_ROOT}/data --model {wildcards.model} \
            --checkpoint {wildcards.label} --origins {params.origins} \
            --out {params.cache}/bdbv-fits >{log} 2>&1
        run germany-fit --data {AGG_ROOT}/germany --model {wildcards.model} \
            --checkpoint {wildcards.label} {params.dates} \
            --out {params.cache}/germany-samples >>{log} 2>&1
        touch {output}
        """


rule agg_rows:
    # The paired arm, the scores and the checkpoint's rows.
    input:
        lambda wc: expand(
            f"{AGG_ROOT}/{wc.label}/fits-{{model}}.done", model=AGG["models"]
        ),
    output:
        f"{AGG_ROOT}/{{label}}/aggregation-scores.csv.gz",
    log:
        f"{AGG_ROOT}/{{label}}/rows.log",
    resources:
        mem_mb=AGG["mem_mb"],
    params:
        prefix=lambda wc, resources: job_prefix(resources.mem_mb),
        cache=f"{AGG_ROOT}/{{label}}",
        existing=f"{AGG_ROOT}/existing-scores.csv.gz",
    shell:
        """
        export BDBV_CHRONOS_FT={wildcards.label}
        run() {{ """
        + HEAVY
        + """uv run --group chronos python scripts/agg/finetuned.py "$@"; }}
        rm -rf {params.cache}/bdbv-paired
        """
        + HEAVY
        + """uv run --group chronos python scripts/agg/bdbv_paired.py \
            --data {AGG_ROOT}/data --fits {params.cache}/bdbv-fits \
            --out {params.cache}/bdbv-paired >{log} 2>&1
        mkdir -p {params.cache}/scores
        run bdbv-score --data {AGG_ROOT}/data --fits {params.cache}/bdbv-fits \
            {params.cache}/bdbv-paired --out {params.cache}/scores/bdbv.csv \
            >>{log} 2>&1
        run germany-score --data {AGG_ROOT}/germany \
            --samples {params.cache}/germany-samples \
            --out {params.cache}/scores/germany.csv >>{log} 2>&1
        run export --existing {params.existing} \
            --checkpoint {wildcards.label} \
            --bdbv {params.cache}/scores/bdbv.csv \
            --germany {params.cache}/scores/germany.csv --out {output} \
            >>{log} 2>&1
        """


def followups(wc=None):
    labels = slots_labels()
    out = []
    if SLC["run"]:
        out += expand(f"{CACHE}/slots-check/{{label}}/summary.csv", label=labels)
    if AGG["run"]:
        out += expand(f"{AGG_ROOT}/{{label}}/aggregation-scores.csv.gz", label=labels)
    return out


rule followups:
    input:
        followups,
