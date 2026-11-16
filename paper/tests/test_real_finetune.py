"""The SI comparison of Chronos-2 fine-tuned on real data with the model
fine-tuned on simulations."""

from pathlib import Path

import pandas as pd

from paperfigs.checkpoints import drop_real_data
from paperpipe.outputs import real_finetune_table, score_tables

SPLIT = pd.Timestamp("2022-02-10")
MODELS = {
    "randomwalknowcast": 2.0,
    "fm_chronos_daily_ragged": 3.0,
    "fm_chronos_daily_ragged_ft": 1.5,
    "fm_chronos_daily_ragged_real": 1.0,
}


def _scores() -> pd.DataFrame:
    rows = []
    for model, score in MODELS.items():
        for origin in ("2022-01-10", "2022-03-10"):
            for location, stratum in (
                ("DE", "all"),
                ("DE-BY", "all"),
                ("DE", "00-04"),
            ):
                # The real-data model is best in-sample.
                held = origin >= "2022-02-10"
                rows.append(
                    {
                        "study": "germany",
                        "model": model,
                        "location": location,
                        "stratum": stratum,
                        "origin_date": pd.Timestamp(origin),
                        "target_date": pd.Timestamp(origin),
                        "horizon": 0,
                        "metric": "wis",
                        "score": score if held else score / 10,
                        "log_score": 0.1,
                        "bias": 0.0,
                        "coverage_50": 1.0,
                        "coverage_95": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_real_data_models_are_dropped_from_the_other_outputs() -> None:
    df = pd.DataFrame(
        {"model": ["fm_chronos_daily_ragged_real", "fm_chronos_daily_ragged"]}
    )
    assert drop_real_data(df)["model"].tolist() == ["fm_chronos_daily_ragged"]


def test_the_comparison_is_on_the_held_out_half_by_level() -> None:
    table = real_finetune_table(_scores(), SPLIT)
    assert set(table["level"]) == {"all", "national", "states", "age groups"}
    assert set(table["model"]) == set(MODELS)
    de = table.loc[
        (table["model"] == "fm_chronos_daily_ragged_real")
        & (table["level"] == "all")
    ]
    # Held-out scores only: 1.0 against randomwalknowcast's 2.0.
    assert de["rel_score"].item() == 0.5
    assert de["n"].item() == 3


def test_german_score_tables_add_the_comparison_and_leave_it_out_elsewhere(
    tmp_path: Path,
) -> None:
    written = score_tables(
        "germany",
        _scores(),
        pd.DataFrame(),
        tmp_path / "main",
        tmp_path / "si",
        split=SPLIT,
        baseline="randomwalknowcast",
    )
    table = tmp_path / "si" / "germany-real-finetune.csv"
    assert table in written
    assert table.with_suffix(".md").exists()
    for path in written:
        if path != table:
            models = set(pd.read_csv(path)["model"])
            assert "fm_chronos_daily_ragged_real" not in models


def _ebola_scores() -> pd.DataFrame:
    rows = []
    for model, score in MODELS.items():
        for origin in ("2026-11-08", "2026-11-15"):
            for location in ("CD", "CD-NK", "CD-NK-14"):
                rows.append(
                    {
                        "study": "ebola",
                        "model": model,
                        "location": location,
                        "stratum": "all",
                        "origin_date": pd.Timestamp(origin),
                        "target_date": pd.Timestamp(origin),
                        "horizon": 0,
                        "metric": "crps",
                        "score": score if origin > "2026-11-08" else 0.1,
                        "log_score": 0.1,
                        "bias": 0.0,
                        "coverage_50": 1.0,
                        "coverage_90": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_bdbv_scores_only_rounds_after_the_last_release_trained_on() -> None:
    table = real_finetune_table(_ebola_scores(), None, after="2026-11-08")
    assert set(table["level"]) == {"all", "national", "provinces", "zones"}
    real = table.loc[
        (table["model"] == "fm_chronos_daily_ragged_real")
        & (table["level"] == "all")
    ]
    assert real["rel_score"].item() == 0.5
    assert real["n"].item() == 3


def test_ebola_score_tables_add_the_comparison_pending_until_rounds_score(
    tmp_path: Path,
) -> None:
    scores = _ebola_scores()
    early = scores.loc[scores["origin_date"] <= "2026-11-08"]
    written = score_tables(
        "ebola",
        early,
        pd.DataFrame(),
        tmp_path / "main",
        tmp_path / "si",
        baseline="randomwalknowcast",
        after="2026-11-08",
    )
    table = tmp_path / "si" / "ebola-real-finetune.csv"
    assert table in written
    assert "Pending" in table.with_suffix(".md").read_text()


def test_the_si_text_states_the_bdbv_cut_in_config() -> None:
    import tomllib

    root = Path(__file__).parents[2]
    cut = tomllib.loads((root / "config" / "finetune.toml").read_text())[
        "realdata_bdbv"
    ]
    text = (root / "paper" / "si.qmd").read_text()
    for key in ("release", "cutoff", "truth_until"):
        assert cut[key] in text
    assert f"{cut['truth_delay']} days" in text
