# Offloaded held-out evaluation (WP10).
#
# With `finetune.eval: offload` the trainer only trains: it saves an
# adapter snapshot and a request at step 0 and each evaluation step
# (`train/evals/step-<k>/`), and writes its checkpoints without held-out
# losses. `evaluate_steps` scores the snapshots once their segment ends
# (`train/evals/step-<k>.json`); `evaluate_checkpoint` scores a
# checkpoint's adapter and writes its sidecar `n<N>/held-out.json`, beside
# the segment's protected settings. Both use the trainer's own loss
# functions on its saved held-out blocks
# (workflow/scripts/evaluate_offloaded.py), take the GPU below the
# trainer's priority and are in the `evaluate` group, so WORKER_GROUPS
# places them on any machine (`evaluate=archie`). Snapshots and the blocks
# are no rule's outputs, so they go over to a worker as `copy_in`.
# Readers merge the results (flowlib.evaluations).


rule evaluate_steps:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
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
        train=lambda wc, input: Path(input.settings).parents[1],
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
        f"{EVALUATE_PY} steps {{params.train}} --device {{params.device}} "
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
