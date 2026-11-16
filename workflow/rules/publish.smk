# Publishing checkpoints (was the trainer's `--publish`).
#
# `verify_checkpoint` checks that a checkpoint's hash recomputes from its
# commit (scripts/finetune_verify.py); `publish_checkpoint` releases a
# verified one (workflow/scripts/publish_checkpoint.py). Publishing writes
# GitHub releases, so it is the opt-in target `publish`, never part of
# `finetune` or `all`.


rule verify_checkpoint:
    input:
        settings=f"{TRAIN}/n{{n}}/settings.json",
        adapter=f"{TRAIN}/n{{n}}/finetuned-ckpt/adapter_model.safetensors",
    output:
        f"{TRAIN}/n{{n}}/verified.txt",
    params:
        where=f"{TRAIN}/n{{n}}",
    localrule: True
    shell:
        "uv run python scripts/finetune_verify.py {params.where} > {output}"


rule publish_checkpoint:
    input:
        rules.verify_checkpoint.output,
    output:
        f"{TRAIN}/n{{n}}/published.json",
    params:
        where=f"{TRAIN}/n{{n}}",
    localrule: True
    shell:
        "uv run python workflow/scripts/publish_checkpoint.py "
        "{params.where} {output}"


rule publish:
    input:
        expand(f"{TRAIN}/n{{n}}/published.json", n=FT["prefixes"]),
