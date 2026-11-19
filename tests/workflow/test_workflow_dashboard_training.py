"""A run's schedule and losses, for the status tiles and loss plots."""

import json

from flowlib.dashboard import logs, runs, training


def test_the_log_reads_any_source_names(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(
        "2026-11-17 10:00:00 [train] step 200: loss 0.5, held-out (first 100)"
        " 0.4, by source 3day 0.31, daily+univariate 0.29, weekly 0.12\n"
    )
    row = logs.Log(path).rows()[0]
    assert row["src"] == {
        "3day": 0.31,
        "daily+univariate": 0.29,
        "weekly": 0.12,
    }


def test_run_params_from_the_command_line_settings_and_history(tmp_path):
    n = tmp_path / "train" / "n250"
    n.mkdir(parents=True)
    (n / "settings.json").write_text(json.dumps({"lr_steps": 999}))
    args = (
        "python scripts/finetune_stream.py --learning-rate 1e-5 "
        "--lr-steps 36500 --bias-lr 1e-3 --prefixes 250 500 8000"
    )
    hist = [{"step": 100, "lr": {"LoRA": 9e-6, "bias": 9e-4}}]
    got = training.run_params(tmp_path, hist, 100, args, (250,))
    assert got["lr_steps"] == 36500
    assert got["groups"] == {"adapter": 1e-5, "bias": 1e-3, "LoRA": 9e-6}
    assert got["current"]["bias"] == 9e-4
    assert got["prefixes"] == [250, 500, 8000]


def test_run_params_decay_linearly_without_history(tmp_path):
    args = "x --learning-rate 1e-5 --lr-steps 1000 --prefixes 250"
    got = training.run_params(tmp_path, [], 250, args, (250,))
    assert got["current"]["adapter"] == 1e-5 * 0.75


def test_run_params_decay_over_the_outbreaks_without_a_step_count(tmp_path):
    args = "x --learning-rate 1e-5 --outbreaks 8400 --prefixes 250 8400"
    got = training.run_params(tmp_path, [], 0, args, (250,))
    assert got["lr_steps"] == 0
    assert got["lr_outbreaks"] == 8400


def test_steps_per_simulation_follow_the_schedule_or_the_last_checkpoint():
    assert training.steps_per_simulation(36500, [250, 8000], []) == 4.5625
    cks = [("n500", {"steps": 2000, "n_outbreaks": 500}, None)]
    assert training.steps_per_simulation(0, [250, 500], cks) == 4.0
    assert training.steps_per_simulation(0, [250], []) == 1.0


def test_loss_runs_take_held_out_by_source_from_history(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {"step": 0, "held_out_loss": 0.5, "by_source": {"weekly": 0.3}},
                {
                    "step": 200,
                    "loss": 0.45,
                    "by_source": {"weekly": 0.25, "3day": 0.4},
                },
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    got = training.loss_run(
        run, rows=[], checkpoints=[], spo=4.0, base=None, colour="#111"
    )
    assert got["bysrc"] == {"weekly": [[200, 0.25]], "3day": [[200, 0.4]]}
    assert got["full0"] == 0.5
    assert got["src0"] == {"weekly": 0.3}
    assert got["train"] == [[200, 0.45]]


def test_loss_runs_carry_the_nowcast_and_forecast_parts(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "held_out_loss": 0.5,
                    "held_out_loss_nowcast": 0.2,
                    "held_out_loss_forecast": 0.3,
                },
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    rows = [
        {
            "step": 100,
            "loss": 0.5,
            "held": 0.4,
            "parts": {"nowcast": 0.1, "forecast": 0.3},
        },
        {"step": 200, "loss": 0.5, "held": 0.39, "parts": {}},
    ]
    cks = [
        (
            "n250",
            {
                "steps": 1000,
                "held_out_loss": 0.35,
                "held_out_loss_nowcast": 0.1,
                "held_out_loss_forecast": 0.25,
            },
            None,
        ),
        ("n500", {"steps": 2000, "held_out_loss": 0.3}, None),
    ]
    got = training.loss_run(
        run, rows=rows, checkpoints=cks, spo=4.0, base=None, colour="#111"
    )
    assert got["parts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["bparts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["held_parts"] == {
        "nowcast": [[100, 0.1]],
        "forecast": [[100, 0.3]],
    }
    assert got["ck_parts"] == {
        "nowcast": [[1000, 0.1]],
        "forecast": [[1000, 0.25]],
    }
