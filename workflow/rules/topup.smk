# Top-up of the training rounds' missing cells (flowlib.topup).
#
# Target `topup`: the grid's training rounds (grid_rounds.<grid>.rounds)
# must be simulated and QC'd. Each missing cell (QC-excluded, or failed
# by the simulator) is simulated again with a fresh seed in the grid's
# `topup_round`, in tries: try t plans the cells still missing after the
# sim QC of try t - 1 (`topup_plan`), simulates them in batches of at
# most `simulate.batch` with the usual batch rule (ENW_CELLS), and checks
# the top-up round with the sim QC (`topup_qc`). A try with nothing left
# ends it; after `simulate.topup_tries` tries, cells still missing are
# reported. `topup` then writes the top-up round's manifest, seed list
# and READY. A run trains on the top-ups by listing the top-up round's
# seeds after the training rounds' in its `seed_file`.


ruleorder: topup > round_ready
ruleorder: topup > round_manifest


checkpoint topup_plan:
    input:
        lambda wc: topup_plan_inputs(int(wc.attempt)),
    output:
        f"{TOPUP_DIR}/topup/plan-{{attempt}}.txt",
    log:
        f"{TOPUP_DIR}/logs/topup-plan-{{attempt}}.log",
    localrule: True
    wildcard_constraints:
        attempt=r"\d+",
    run:
        topup_write_plan(int(wildcards.attempt), output[0], log[0])


rule topup_qc:
    # The sim QC of the whole top-up round once a try's batches are in.
    input:
        lambda wc: topup_try_records(int(wc.attempt)),
    output:
        touch(f"{TOPUP_DIR}/qc/try-{{attempt}}.done"),
    log:
        f"{TOPUP_DIR}/logs/topup-qc-{{attempt}}.log",
    wildcard_constraints:
        attempt=r"\d+",
    conda:
        TOOLS_ENV
    params:
        round_dir=lambda wc, output: Path(output[0]).parents[1],
        limits=" ".join(
            f"--{k.replace('_', '-')} {v}" for k, v in SIM.get("qc", {}).items()
        ),
    shell:
        "uv run python scripts/sim_qc.py {params.round_dir} {params.limits} "
        "> {log} 2>&1"


rule topup:
    input:
        topup_done,
    output:
        manifest=f"{TOPUP_DIR}/manifest.csv",
        seeds=f"{TOPUP_DIR}/seeds.txt",
        ready=f"{TOPUP_DIR}/READY",
    log:
        f"{TOPUP_DIR}/logs/topup.log",
    localrule: True
    run:
        topup_finish(input, output, log[0])
