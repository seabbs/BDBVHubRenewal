# Offloaded held-out evaluation (WP10): with `finetune.eval: offload`,
# score each segment's step snapshots and its checkpoint after it ends, in
# the `evaluate` group (WORKER_GROUPS places it), on the GPU below the
# trainer. Files and readers: workflow/flowlib/evaluations.py.


rule evaluate_steps:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        # Its steps start after the previous checkpoint's.
        prev=lambda wc: evaluate_prev(wc.n),
    output:
        touch(f"{TRAIN}/evals/n{{n}}.done"),
    log:
        f"{TRAIN}/logs/evaluate-steps-n{{n}}.log",
    priority: EVALUATE_PRIORITY
    conda:
        TOOLS_ENV
    resources:
        gpu=1,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        prev=lambda wc, input: (
            f"--prev {Path(input.prev).parent}" if input.prev else ""
        ),
        real=evaluate_real,
        device=machine.device(LOCAL),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=[
            f"{TRAIN}/evals/",
            f"{TRAIN}/{evaluations.HELD_OUT_BLOCKS}",
            *([FT["real_validation"]] if FT["real_validation"] else []),
        ],
        copy_back=[f"{TRAIN}/evals/"],
    shell:
        "DEVICE={params.device} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} steps {{params.where}} {{params.prev}} "
        "--device {params.device} "
        "{params.real} > {log} 2>&1"


rule evaluate_checkpoint:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
        # Step 0's result, base Chronos-2's loss, is scored with the steps.
        steps=f"{TRAIN}/evals/n{{n}}.done",
    output:
        f"{TRAIN}/n{{n}}/{evaluations.SIDECAR}",
    log:
        f"{TRAIN}/logs/evaluate-checkpoint-n{{n}}.log",
    priority: EVALUATE_PRIORITY
    conda:
        TOOLS_ENV
    resources:
        gpu=1,
    params:
        where=lambda wc, input: Path(input.settings).parent,
        real=evaluate_real,
        device=machine.device(LOCAL),
        hf_home=LOCAL.get("HF_HOME", "output/finetune/hf"),
        copy_in=lambda wc: evaluate_inputs_from(wc.n),
    shell:
        "DEVICE={params.device} HF_HOME={params.hf_home} HF_HUB_OFFLINE=1 "
        f"{EVALUATE_PY} checkpoint {{params.where}} --device {{params.device}} "
        "{params.real} > {log} 2>&1"
