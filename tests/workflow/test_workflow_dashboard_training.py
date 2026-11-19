"""A run's schedule and losses, for the status tiles and loss plots."""

import json

import pytest
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
        run,
        rows=[],
        checkpoints=[],
        sims=lambda s: s / 4,
        base=None,
        colour="#111",
    )
    # Against simulations trained: 4 steps a simulation.
    assert got["bysrc"] == {"weekly": [[50, 0.25]], "3day": [[50, 0.4]]}
    assert got["full0"] == 0.5
    assert got["src0"] == {"weekly": 0.3}
    assert got["train"] == [[50, 0.45]]


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
        run,
        rows=rows,
        checkpoints=cks,
        sims=lambda s: s / 4,
        base=None,
        colour="#111",
    )
    assert got["parts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["bparts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["held_parts"] == {
        "nowcast": [[25, 0.1]],
        "forecast": [[25, 0.3]],
    }
    # Checkpoints are at their own simulations.
    assert got["ck_parts"] == {
        "nowcast": [[250, 0.1]],
        "forecast": [[250, 0.25]],
    }
    assert got["ck"] == [[250, 0.35], [500, 0.3]]


def test_steps_map_to_simulations_from_checkpoints_and_chunks():
    cks = [("n250", {"steps": 1500, "n_outbreaks": 250}, None)]
    # Chunks: 250 sims in 1500 steps (also the checkpoint), then 250 in
    # 1000; a chunk after a gap in the logs is not used.
    chunks = {0: (250, 1500), 250: (500, 1000), 750: (1000, 900)}
    pairs = training.anchors(cks, chunks)
    assert pairs == [(0, 0), (1500, 250), (2500, 500)]
    assert training.sims_at(pairs, 750) == 125
    assert training.sims_at(pairs, 2000) == 375
    # Past the last pair, at the mean rate so far: 5 steps a simulation.
    assert training.sims_at(pairs, 3000) == 600


def test_a_live_run_before_its_first_checkpoint_maps_through_its_chunks():
    pairs = training.anchors([], {0: (250, 2000)})
    assert training.sims_at(pairs, 1000) == 125


def test_with_nothing_known_steps_map_at_the_given_rate():
    pairs = training.anchors([], {})
    assert training.sims_at(pairs, 600, spo=6.0) == 100


def test_a_run_from_the_workflow_decays_over_its_simulations(tmp_path):
    # Snakemake starts the trainer: no pid file, so no command line. The
    # settings say the rate falls over the run's simulations (`default`).
    n = tmp_path / "train" / "n250"
    n.mkdir(parents=True)
    (n / "settings.json").write_text(
        json.dumps(
            {
                "learning_rate": 1e-5,
                "lr_decay": "linear to zero over the outbreaks",
            }
        )
    )
    got = training.run_params(tmp_path, [], 3000, "", (250, 8400), sims=2100)
    assert got["lr_outbreaks"] == 8400
    assert got["current"]["adapter"] == pytest.approx(1e-5 * 0.75)


def test_the_step_a_checkpoint_is_saved_at_comes_from_planned_chunks():
    cks = [("n250", {"steps": 1500, "n_outbreaks": 250}, None)]
    pairs = training.anchors(cks, {0: (250, 1500), 250: (500, 1400)})
    # n500's chunk is planned: its step is known.
    assert training.step_at(pairs, 500) == (2900, True)
    # n750 is not: at the run's mean rate so far, 5.8 steps a simulation.
    assert training.step_at(pairs, 750) == (4350, False)
    assert training.step_at([(0, 0)], 250, spo=4.0) == (1000, False)


def test_the_rate_uses_logged_steps_and_the_status_over_a_long_enough_window():
    # Evaluations every 400 steps leave one logged step in 30 minutes; the
    # status file's step and time make the second point.
    t0 = 1_000_000.0
    points = [(t0, 800), (t0 + 2400, 1200), (t0 + 2400 + 900, 1350)]
    rate, span = training.recent_rate(points)
    assert rate == pytest.approx(550 / 3300)
    assert span == pytest.approx(3300)
    # With points only far apart, the last two are used.
    rate, _ = training.recent_rate([(t0, 0), (t0 + 20000, 4000)])
    assert rate == pytest.approx(0.2)
    # One point, or none: no rate.
    assert training.recent_rate([(t0, 5)]) == (None, None)
    assert training.recent_rate([]) == (None, None)


def test_a_falling_logged_rate_is_a_decay_even_without_lr_decay(tmp_path):
    n = tmp_path / "train" / "n250"
    n.mkdir(parents=True)
    (n / "settings.json").write_text(json.dumps({"learning_rate": 1e-5}))
    hist = [
        {"step": 400, "lr": {"lora": 9.9e-6}},
        {"step": 800, "lr": {"lora": 9.7e-6}},
    ]
    got = training.run_params(tmp_path, hist, 800, "", (250, 8400), sims=160)
    assert got["lr_outbreaks"] == 8400
