"""Tests of the live fine-tuning run's checkpoint selection and tables."""

import pandas as pd
import pytest

RULE = "own-quantile-wis_crps-1000-common"


def _rows() -> pd.DataFrame:
    """Checkpoint scores of a live run, another run and zero-shot."""
    rows = []
    for ck, run, n in [
        ("zero-shot", "zero-shot", 0),
        ("slots-a", "one-model", 125),
        ("slots-b", "one-model", 250),
        ("slots-z", "slots-zeroinit", 500),
    ]:
        for model in ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"):
            rows.append(
                {
                    "study": "ebola",
                    "level": "national",
                    "model": model,
                    "checkpoint": ck,
                    "run": run,
                    "outbreaks": n,
                    "rule": RULE,
                    "target": "7day",
                    "rel_crps": 2.0 - n / 1000,
                    "rel_log_crps": 1.5,
                    "last_origin": "2026-11-08",
                }
            )
            if ck in ("zero-shot", "slots-a"):
                rows.append(
                    {
                        "study": "germany",
                        "level": "national",
                        "model": model,
                        "checkpoint": ck,
                        "run": run,
                        "outbreaks": n,
                        "rule": RULE,
                        "target": "7day",
                        "rel_wis": 3.0,
                    }
                )
        if n:
            rows.append(
                {
                    "study": "held_out",
                    "level": "all",
                    "model": "fm_chronos",
                    "checkpoint": ck,
                    "run": run,
                    "outbreaks": n,
                    "rule": RULE,
                    "target": "7day",
                    "loss": 0.4 - n / 10000,
                    "loss_zero_shot": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_newest_is_per_study_within_the_live_run() -> None:
    from bdbvhubrenewal.reporting.live_checkpoint import newest

    got = newest(_rows(), "one-model")
    assert got["ebola"] == ("slots-b", 250)
    assert got["germany"] == ("slots-a", 125)
    assert got["held_out"] == ("slots-b", 250)


def test_the_label_names_n_of_the_planned_simulations() -> None:
    from bdbvhubrenewal.reporting.live_checkpoint import label

    assert label(1000) == (
        "Preliminary: fine-tuned on 1,000 of 8,000 simulations."
    )


def test_scored_bdbv_rounds_are_dropped_with_a_warning() -> None:
    from bdbvhubrenewal.reporting.live_checkpoint import tidy

    late = _rows().assign(last_origin="2026-11-28")
    with pytest.warns(UserWarning, match="scored"):
        out = tidy(late)
    assert not (out["study"] == "ebola").any()
    assert (out["study"] == "germany").any()


def test_the_checkpoint_table_has_zero_shot_first_then_each_checkpoint() -> (
    None
):
    from bdbvhubrenewal.reporting.live_checkpoint import checkpoint_table

    t = checkpoint_table(_rows(), "one-model")
    assert list(t["Simulations"]) == ["Zero-shot", "125", "250"]
    assert len(t.columns) <= 5
    assert t.iloc[1]["German WIS"] == "3.00"
    # A checkpoint not yet scored on a study is left blank.
    assert t.iloc[2]["German WIS"] == ""
    assert t.iloc[2]["Held-out loss"] == "0.38"
    assert t.iloc[0]["Held-out loss"] == "0.50"


def test_the_summary_has_zero_shot_and_the_newest_checkpoint() -> None:
    from bdbvhubrenewal.reporting.live_checkpoint import summary_table

    t = summary_table(_rows(), "one-model")
    assert len(t) == 2
    assert t.iloc[1]["German WIS"] == "3.00 (125)"
    assert t.iloc[1]["BDBV CRPS"] == "1.75 (250)"


def test_the_docs_page_gets_the_label_and_summary(tmp_path) -> None:
    from bdbvhubrenewal.cli import main

    scores = tmp_path / "scores.csv"
    _rows().to_csv(scores, index=False)
    out = tmp_path / "ft-preliminary.md"
    args = ["live-checkpoint", "--scores", str(scores), "--out", str(out)]
    assert main([*args, "--run", "one-model"]) == 0
    text = out.read_text()
    assert "Preliminary" in text
    assert "Newest checkpoint" in text


def test_the_docs_page_waits_without_scores(tmp_path) -> None:
    from bdbvhubrenewal.cli import main

    out = tmp_path / "ft-preliminary.md"
    missing = str(tmp_path / "none.csv")
    assert (
        main(["live-checkpoint", "--scores", missing, "--out", str(out)]) == 0
    )
    assert "not yet" in out.read_text()


def test_an_empty_scores_file_also_waits(tmp_path) -> None:
    from bdbvhubrenewal.cli import main

    empty = tmp_path / "scores.csv"
    empty.write_text("")
    out = tmp_path / "ft-preliminary.md"
    assert (
        main(["live-checkpoint", "--scores", str(empty), "--out", str(out)])
        == 0
    )
    assert "not yet" in out.read_text()


def test_live_tags_map_the_run_s_labels_to_release_tags() -> None:
    from bdbvhubrenewal.reporting.live_checkpoint import live_tags

    rows = _rows().replace(
        {"checkpoint": {"slots-a": "slots-aaaaaa", "slots-b": "slots-bbbbbb"}}
    )
    assert live_tags(rows, "one-model") == {
        "chronos-slots-aaaaaa": 125,
        "chronos-slots-bbbbbb": 250,
    }
