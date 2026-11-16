import numpy as np
import pandas as pd
import pytest
import yaml

from paperpipe.evidence import (
    EVIDENCE,
    aggregation_numbers,
    aggregation_table,
    copula_table,
    rho_medians,
    write_evidence,
)


def _copula() -> pd.DataFrame:
    rows = []
    for arm, es in (("rolling", 0.3), ("unpaired", 0.5), ("fixed", 0.31)):
        for origin in ("2026-10-04", "2026-10-11"):
            rows.append(
                {
                    "study": "ebola",
                    "strata": "provinces",
                    "model": "randomwalknowcast",
                    "arm": arm,
                    "origin_date": origin,
                    "energy_log": es,
                    "variogram_log": 40.0,
                    "cover50_weeks": 0.9,
                    "cover90_weeks": 0.95,
                    "cover50_places": 0.6,
                    "cover90_places": 0.7,
                    "wis_strata": 1.0,
                    "wis_national": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_copula_table_averages_and_orders_the_arms() -> None:
    t = copula_table(_copula(), "ebola")
    assert t["Pairing"].tolist() == ["Unpaired", "Fixed", "Rolling"]
    assert t["ES (log)"].tolist() == [0.5, 0.31, 0.3]
    assert t["Places 50% / 90%"].iloc[0] == "0.60 / 0.70"


def test_rho_medians_by_strata_and_model() -> None:
    rho = pd.DataFrame(
        {
            "study": "germany",
            "strata": "states",
            "model": "randomwalknowcast",
            "arm": "rolling",
            "rho_t": [0.5, 0.6, 0.7],
            "rho_p": [0.0, 0.04, 0.1],
            "rho_n": np.nan,
        }
    )
    # The fixed arm's prior is not an estimate.
    rho = pd.concat([rho, rho.assign(arm="fixed", rho_t=0.9, rho_p=0.7)])
    out = rho_medians(rho)
    assert out["germany"]["states"]["randomwalknowcast"] == {
        "rho_t": 0.6,
        "rho_p": 0.04,
    }


def test_aggregation_is_relative_to_the_reference_on_shared_targets() -> None:
    base = {
        "study": "ebola",
        "level": "national",
        "location": "CD",
        "origin_date": "2026-10-04",
        "cover50": 1,
        "cover95": 1,
    }
    rows = [
        {
            **base,
            "model": "randomwalknowcast",
            "option": "reference",
            "target_date": t,
            "wis": 10.0,
            "log_wis": 1.0,
        }
        for t in ("2026-09-27", "2026-10-04")
    ] + [
        {
            **base,
            "model": "fm",
            "option": "joint",
            "target_date": "2026-09-27",
            "wis": 5.0,
            "log_wis": 0.5,
        },
        {
            **base,
            "model": "fm",
            "option": "level",
            "target_date": "2026-09-27",
            "wis": 20.0,
            "log_wis": 2.0,
        },
    ]
    t = aggregation_table(pd.DataFrame(rows), "ebola")
    assert t["Option"].tolist() == ["Joint block", "Level by level"]
    assert t["Relative WIS (log)"].tolist() == ["0.50 (0.50)", "2.00 (2.00)"]
    assert t["n"].tolist() == [1, 1]
    numbers = aggregation_numbers(pd.DataFrame(rows))
    assert numbers["ebola"]["national"]["fm"] == {"joint": 0.5, "level": 2.0}


def test_missing_evidence_leaves_a_note_and_no_numbers(tmp_path) -> None:
    out = write_evidence(
        dict.fromkeys(EVIDENCE),
        tmp_path / "si",
        tmp_path / "fig",
        tmp_path / "e.yml",
    )
    assert "waits on" in (tmp_path / "si" / "copula-ebola.md").read_text()
    assert (tmp_path / "fig" / "sim-realism.png").is_file()
    doc = yaml.safe_load((tmp_path / "e.yml").read_text())
    rho = doc["evidence"]["rho"]["germany"]["states"]["randomwalknowcast"]
    assert rho["rho_t"] == "(not yet available)"
    assert len(out["evidence"]["waiting"]) == len(EVIDENCE)


def test_a_paper_run_fails_while_evidence_is_missing(tmp_path) -> None:
    from paperpipe.__main__ import main

    with pytest.raises(ValueError, match="waits on"):
        main(
            [
                "evidence",
                "--mode",
                "paper",
                "--si-dir",
                str(tmp_path),
                "--fig-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "e.yml"),
            ]
        )


def test_a_study_without_rows_still_waits(tmp_path) -> None:
    rows = tmp_path / "agg.csv"
    pd.DataFrame(
        {
            "study": "ebola",
            "model": ["randomwalknowcast", "fm"],
            "option": ["reference", "joint"],
            "level": "national",
            "location": "CD",
            "origin_date": "2026-10-04",
            "target_date": "2026-09-27",
            "wis": 1.0,
            "log_wis": 1.0,
            "cover50": 1,
            "cover95": 1,
        }
    ).to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"aggregation_scores": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    assert (
        "Joint block" in (tmp_path / "si" / "aggregation-ebola.md").read_text()
    )
    assert (
        "waits on" in (tmp_path / "si" / "aggregation-germany.md").read_text()
    )


def test_lag_numbers_pair_each_model_with_its_longer_horizon() -> None:
    from paperpipe.evidence import lag_numbers

    germany = pd.DataFrame(
        {
            "pool": "national",
            "model": ["fm_chronos_daily_ragged", "fm_chronos_daily_ragged_fix"],
            "rel_wis_randomwalknowcast": [2.9, 3.8],
        }
    )

    def bdbv(value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scale": "all",
                "target": "complete",
                "weeks": "nowcast",
                "model": ["fm_chronos_weekly_ragged", "randomwalk"],
                "rel_wis": [value, 1.4],
            }
        )

    out = lag_numbers(germany, bdbv(1.3), bdbv(3.2))
    assert out["germany"]["fm_chronos_daily_ragged"] == {
        "capped": 2.9,
        "longer": 3.8,
    }
    assert out["bdbv"] == {
        "fm_chronos_weekly_ragged": {"capped": 1.3, "longer": 3.2}
    }


def test_checkpoint_panels_show_n_and_grey_thin_ones() -> None:
    from paperpipe.evidence import checkpoint_figure, checkpoint_numbers

    rows = pd.DataFrame(
        {
            "study": ["ebola"] * 3 + ["germany"] * 3,
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "outbreaks": [0, 1000, 2000] * 2,
            "n": [1] * 3 + [400] * 3,
            "rel_wis": [2.0, 1.5, 1.2] * 2,
        }
    )
    fig = checkpoint_figure(rows)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == [
        "ebola, national (n = 1)",
        "germany, national (n = 400)",
    ]
    assert "too few cells" in axes[0].texts[0].get_text()
    assert not axes[1].texts
    numbers = checkpoint_numbers(rows)
    assert numbers["germany"]["national"]["n"] == 400
    assert numbers["ebola"]["national"]["fm_chronos_daily_ragged"] == {
        "zero-shot": 2.0,
        "n1000": 1.5,
        "n2000": 1.2,
    }


def test_scale_floor_numbers_split_simulations_and_bdbv() -> None:
    from paperpipe.evidence import scale_floor_numbers

    rows = pd.DataFrame(
        {
            "floor": [0, 1, 0, 1],
            "model": "n4000",
            "size": ["small (<=5)", "small (<=5)", "provinces", "provinces"],
            "median_loss": [3.311, 4.814, None, None],
            "share_q99_over_100x": [0.1067, 0.2845, None, None],
            "source": [None, None, "weekly", "weekly"],
            "relative_wis": [None, None, 4.2, 2.94],
        }
    )
    out = scale_floor_numbers(rows)
    small = out["sims"]["n4000"]["small"]
    assert small["median_loss"] == {"none": 3.3, "floor": 4.8}
    assert small["q99_over_100x"] == {"none": 11, "floor": 28}
    assert out["bdbv"]["provinces"]["weekly"]["n4000"] == {
        "none": 4.2,
        "floor": 2.94,
    }


def test_pair_is_blank_when_either_value_is_missing() -> None:
    from paperpipe.evidence import _pair

    a = pd.Series([0.5, np.nan, 0.5])
    b = pd.Series([0.25, 0.25, np.nan])
    assert _pair(a, b) == ["0.50 / 0.25", "", ""]


def test_checkpoint_models_keep_one_colour_across_panels() -> None:
    from paperpipe.evidence import checkpoint_figure

    rows = pd.DataFrame(
        {
            "study": ["ebola"] * 4 + ["germany"] * 2,
            "level": "national",
            "model": ["a", "a", "b", "b", "b", "b"],
            "outbreaks": [0, 1000] * 3,
            "n": 400,
            "rel_wis": [2.0, 1.5, 1.8, 1.2, 1.1, 0.9],
        }
    )
    fig = checkpoint_figure(rows)
    colours = {
        (ax.get_title(), line.get_label()): line.get_color()
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_label() in {"a, fine-tuned", "b, fine-tuned"}
    }
    assert (
        colours[("ebola, national (n = 400)", "b, fine-tuned")]
        == colours[("germany, national (n = 400)", "b, fine-tuned")]
    )


def test_checkpoint_rows_keep_one_grid_and_the_current_rule() -> None:
    from paperpipe.evidence import CHECKPOINT_RULE, checkpoint_rows

    base = {"level": "national", "model": "fm_chronos_daily_ragged", "n": 9}
    rows = pd.DataFrame(
        [
            {
                **base,
                "study": "germany",
                "checkpoint": "zero-shot",
                "outbreaks": 0,
                "grid": None,
                "rule": CHECKPOINT_RULE,
                "rel_wis": 2.0,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "a",
                "outbreaks": 1000,
                "grid": "r3",
                "rule": CHECKPOINT_RULE,
                "rel_wis": 1.5,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "b",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": 1.2,
                "rel_crps": None,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "b",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": "samples-100",
                "rel_wis": 9.0,
                "rel_crps": None,
            },
            {
                **base,
                "study": "ebola",
                "checkpoint": "b",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": None,
                "rel_crps": 0.8,
            },
            {
                **base,
                "study": "held_out",
                "checkpoint": "b",
                "outbreaks": 1000,
                "grid": "r4",
                "rule": CHECKPOINT_RULE,
                "rel_wis": None,
                "rel_crps": None,
            },
        ]
    )
    out = checkpoint_rows(rows, checkpoints=("b",))
    assert set(out["study"]) == {"germany", "ebola"}
    assert sorted(out["checkpoint"]) == ["b", "b", "zero-shot"]
    assert out.set_index("study").loc["ebola", "rel_wis"] == 0.8
    # Old tables, with no grid or rule, give the zero-shot rows only.
    old = rows.drop(columns=["grid", "rule", "rel_crps"]).iloc[:2]
    assert list(checkpoint_rows(old, checkpoints=("b",))["checkpoint"]) == [
        "zero-shot"
    ]


def test_checkpoint_rows_of_other_runs_leave_only_zero_shot() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = pd.DataFrame(
        {
            "study": ["germany", "germany"],
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "checkpoint": ["zero-shot", "6b7aaa8d4f55"],
            "outbreaks": [0, 1000],
            "n": 9,
            "rel_wis": [2.0, 1.5],
        }
    )
    out = checkpoint_rows(rows, checkpoints=("b",))
    assert list(out["checkpoint"]) == ["zero-shot"]


def test_checkpoint_rows_keep_only_the_final_run() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = pd.DataFrame(
        {
            "study": "germany",
            "level": "national",
            "model": "fm_chronos_daily_ragged",
            "checkpoint": ["a", "b", "c"],
            "outbreaks": [1000, 2000, 4000],
            "n": 9,
            "rel_wis": 1.0,
            "grid": ["r4", "r4.1", "r3"],
        }
    )
    # Only the final run's checkpoints, whatever their grid.
    assert list(checkpoint_rows(rows, checkpoints=("b",))["checkpoint"]) == [
        "b"
    ]


def test_joint_evidence_leaves_out_fine_tuned_rows_of_earlier_runs() -> None:
    from paperpipe.evidence import drop_finetuned

    rows = pd.DataFrame(
        {
            "model": [
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged_ft",
                "fm_chronos_daily_ragged_ft_n1000",
                "randomwalknowcast",
            ]
        }
    )
    assert list(drop_finetuned(rows)["model"]) == [
        "fm_chronos_daily_ragged",
        "randomwalknowcast",
    ]


def test_aggregation_options_name_the_lowest_level_not_zones() -> None:
    # Germany's lowest level is states or age groups, not zones.
    from paperpipe.evidence import OPTIONS

    assert OPTIONS["zones_summed"] == "Lowest level summed"
    assert OPTIONS["zones_copula"] == "Lowest level summed after pairing"


def test_checkpoint_curves_show_the_untuned_models_as_lines() -> None:
    # The fine-tuned checkpoints are the curve; zero-shot Chronos-2 on the
    # ragged layout and on complete data only are horizontal lines, and
    # randomwalknowcast is the line at 1.
    from paperpipe.evidence import checkpoint_figure

    rows = pd.DataFrame(
        {
            "study": "germany",
            "level": "national",
            "model": [
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_ragged",
                "fm_chronos_daily_complete",
            ],
            "checkpoint": ["zero-shot", "a", "b", "zero-shot"],
            "outbreaks": [0, 1000, 2000, 0],
            "n": 400,
            "rel_wis": [2.0, 1.5, 1.2, 1.7],
        }
    )
    fig = checkpoint_figure(rows)
    ax = next(a for a in fig.axes if a.get_visible())
    lines = {line.get_label(): line for line in ax.get_lines()}
    curve = lines["fm_chronos_daily_ragged, fine-tuned"]
    assert list(curve.get_xdata()) == [1000, 2000]
    zero = lines["fm_chronos_daily_ragged, zero-shot"]
    complete = lines["fm_chronos_daily_complete, complete data only"]
    assert set(zero.get_ydata()) == {2.0}
    assert set(complete.get_ydata()) == {1.7}
    # One colour per source, the references in other line styles.
    assert zero.get_color() == curve.get_color() == complete.get_color()
    assert len({zero.get_linestyle(), complete.get_linestyle()}) == 2
    assert set(lines["randomwalknowcast"].get_ydata()) == {1}
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "fm_chronos_daily_complete, complete data only" in legend
    assert "randomwalknowcast" in legend


def _slots_check() -> pd.DataFrame:
    """Rows in the slots check's layout: slots against zero-shot and no
    slots, and the lag-swap test."""
    base = {"window": "nowcast", "ci_low": 0.8, "ci_high": 0.95}
    return pd.DataFrame(
        [
            {
                **base,
                "set": "sims weekly",
                "model": "slots",
                "condition": "given",
                "against": "zero-shot",
                "loss": 8.0,
                "against_loss": 10.0,
                "ratio": 0.8,
            },
            {
                **base,
                "set": "sims weekly",
                "model": "slots",
                "condition": "given",
                "against": "no-slots",
                "loss": 8.0,
                "against_loss": 9.0,
                "ratio": 0.89,
            },
            {
                **base,
                "set": "sims 3day",
                "model": "slots",
                "condition": "given",
                "against": "no-slots",
                "loss": 4.0,
                "against_loss": 5.0,
                "ratio": 0.8,
            },
            {
                **base,
                "set": "sims weekly",
                "model": "slots",
                "condition": "swap_places",
                "against": "given",
                "loss": 8.4,
                "against_loss": 8.0,
                "ratio": 1.05,
            },
            {
                **base,
                "set": "sims weekly",
                "model": "zero-shot",
                "condition": "swap_places",
                "against": "given",
                "loss": 10.0,
                "against_loss": 10.0,
                "ratio": 1.0,
            },
        ]
    )


def test_slots_loss_leaves_out_3_day_blocks_against_no_slots() -> None:
    # The no-slots run built its 3-day blocks as daily, so only weekly and
    # daily blocks compare cleanly with it.
    from paperpipe.evidence import slots_loss_table

    out = slots_loss_table(_slots_check())
    assert list(out["Against"]) == ["zero-shot", "no-slots"]
    assert set(out["Set"]) == {"sims weekly"}
    assert out["Ratio (90% CI)"].iloc[0] == "0.80 (0.80 to 0.95)"


def test_slots_swap_gives_each_model_s_lag_swap_ratio() -> None:
    from paperpipe.evidence import slots_swap_table

    out = slots_swap_table(_slots_check())
    assert list(out["Model"]) == ["slots", "zero-shot"]
    assert list(out["Ratio (90% CI)"]) == [
        "1.05 (0.80 to 0.95)",
        "1.00 (0.80 to 0.95)",
    ]


def test_slots_scores_pair_the_runs_at_matched_checkpoints() -> None:
    from paperpipe.evidence import CHECKPOINT_RULE, slots_scores_table

    base = {
        "level": "national",
        "model": "fm_chronos_daily_ragged",
        "rule": CHECKPOINT_RULE,
        "grid": "r4",
        "rel_crps": None,
    }
    rows = pd.DataFrame(
        [
            {
                **base,
                "study": "germany",
                "checkpoint": "slots-aaa",
                "outbreaks": 1000,
                "rel_wis": 0.9,
            },
            {
                **base,
                "study": "germany",
                "checkpoint": "bbb",
                "outbreaks": 1000,
                "rel_wis": 1.0,
            },
            # No no-slots checkpoint at 250, so no pair.
            {
                **base,
                "study": "germany",
                "checkpoint": "slots-ccc",
                "outbreaks": 250,
                "rel_wis": 1.1,
            },
            {
                **base,
                "study": "ebola",
                "checkpoint": "slots-aaa",
                "outbreaks": 1000,
                "rel_wis": None,
                "rel_crps": 0.7,
            },
            {
                **base,
                "study": "ebola",
                "checkpoint": "bbb",
                "outbreaks": 1000,
                "rel_wis": None,
                "rel_crps": 0.75,
            },
        ]
    )
    out = slots_scores_table(rows)
    assert list(out["Outbreaks"]) == [1000, 1000]
    assert list(out["Slots"]) == [0.7, 0.9]
    assert list(out["No slots"]) == [0.75, 1.0]


def test_slots_scores_keep_only_the_final_run_s_checkpoints() -> None:
    # An earlier slots run's checkpoints are not the paper's.
    from paperpipe.evidence import CHECKPOINT_RULE, slots_scores_table

    base = {
        "study": "germany",
        "level": "national",
        "model": "fm_chronos_daily_ragged",
        "rule": CHECKPOINT_RULE,
        "outbreaks": 1000,
    }
    rows = pd.DataFrame(
        [
            {**base, "checkpoint": "slots-old", "rel_wis": 0.8},
            {**base, "checkpoint": "slots-new", "rel_wis": 0.9},
            {**base, "checkpoint": "plain", "rel_wis": 1.0},
        ]
    )
    out = slots_scores_table(rows, ("slots-new",))
    assert list(out["Slots"]) == [0.9]
    assert slots_scores_table(rows, ()).empty
