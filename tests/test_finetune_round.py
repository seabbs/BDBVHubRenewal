"""One round of fine-tuning on the laptop, from archie's round directory."""

import subprocess
from pathlib import Path

ROUND = Path(__file__).parents[1] / "scripts" / "finetune_round.sh"


def test_round_trains_its_seed_list_from_the_given_adapter(tmp_path):
    local = tmp_path / "rounds" / "r1"
    (local / "sims").mkdir(parents=True)
    (local / "seeds.txt").write_text("20001\n20002\n")
    (local / "held-out.txt").write_text("29901 29904\n")
    (local / "real-training.pkl").write_bytes(b"x")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ROUND}"; ROUND_LOCAL={local}; '
            'trainer_args 1 prev/finetuned-ckpt; echo "${args[@]}"',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    args = out.stdout
    assert f"--seed-list {local}/seeds.txt" in args
    assert "--held-out-seeds 29901 29904" in args
    assert "--init-adapter prev/finetuned-ckpt" in args
    # Real-data blocks train only when asked, as a separate arm.
    assert "--real-training" not in args
    assert "--learning-rate 1e-05" in args or "--learning-rate 1e-5" in args
    assert "--publish" in args
    # Round 0 has no adapter to continue from.
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ROUND}"; ROUND_LOCAL={local}; '
            'trainer_args 0 none; echo "${args[@]}"',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--init-adapter" not in out.stdout


def test_real_data_blocks_train_only_when_asked(tmp_path):
    local = tmp_path / "rounds" / "r2"
    (local / "sims").mkdir(parents=True)
    (local / "seeds.txt").write_text("20001\n")
    (local / "held-out.txt").write_text("29901 29904\n")
    (local / "real-training.pkl").write_bytes(b"x")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ROUND}"; ROUND_LOCAL={local}; USE_REAL_TRAINING=1; '
            'trainer_args 2 prev/finetuned-ckpt; echo "${args[@]}"',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"--real-training {local}/real-training.pkl" in out.stdout
