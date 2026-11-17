"""The live fine-tuning run's checkpoints in the paper's evidence."""

import pandas as pd
import yaml

from paperpipe.evidence import EVIDENCE, write_evidence

RULE = "own-quantile-wis_crps-1000-common"
MODELS = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")


def _scores() -> pd.DataFrame:
    """Checkpoint scores: zero-shot, the live run, no biases, another run."""
    rows = []
    for ck, run, n in [
        ("zero-shot", "zero-shot", 0),
        ("slots-a", "one-model", 125),
        ("slots-b", "one-model", 1000),
        ("plain", "no-slots", 1000),
        ("slots-z", "slots-zeroinit", 500),
    ]:
        for model in MODELS:
            for level in ("national", "provinces"):
                rows.append(
                    {
                        "study": "ebola",
                        "level": level,
                        "model": model,
                        "checkpoint": ck,
                        "run": run,
                        "outbreaks": n,
                        "n": 10,
                        "rule": RULE,
                        "target": "7day",
                        "rel_crps": 2.0 - n / 1000,
                        "rel_log_crps": 1.5,
                        "last_origin": "2026-11-08",
                    }
                )
            rows.append(
                {
                    "study": "germany",
                    "level": "national",
                    "model": model,
                    "checkpoint": ck,
                    "run": run,
                    "outbreaks": n,
                    "n": 10,
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


def test_the_curve_keeps_zero_shot_and_the_live_run_by_level() -> None:
    from paperpipe.evidence import checkpoint_rows

    out = checkpoint_rows(_scores(), "one-model")
    assert set(out["checkpoint"]) == {"zero-shot", "slots-a", "slots-b"}
    # Each level is kept, for its own panel.
    assert set(out["level"]) == {"national", "provinces"}


def test_the_attention_bias_comparison_comes_from_checkpoint_scores() -> None:
    from paperpipe.evidence import slots_rows

    out = slots_rows(_scores(), "one-model")
    assert set(out["run"]) == {"attention biases", "no biases"}
    # One set per study and level, not pooled.
    assert {"BDBV national, daily", "BDBV provinces, daily"} <= set(out["set"])
    assert "Held-out loss, all" not in set(out["set"])
    assert set(out.columns) >= {"set", "run", "outbreaks", "score"}


def test_the_evidence_has_the_compact_tables_and_labels(tmp_path) -> None:
    rows = tmp_path / "scores.csv"
    _scores().to_csv(rows, index=False)
    paths = dict.fromkeys(EVIDENCE) | {"checkpoint_scores": rows}
    write_evidence(paths, tmp_path / "si", tmp_path / "fig", tmp_path / "e.yml")
    table = (tmp_path / "si" / "checkpoint-table.md").read_text()
    assert "Zero-shot" in table
    assert "slots-z" not in table
    summary = (tmp_path / "si" / "slots-summary.md").read_text()
    assert "no biases" in summary
    assert (tmp_path / "si" / "live-summary.md").is_file()
    assert (tmp_path / "fig" / "compare-slots.png").is_file()
    live = yaml.safe_load((tmp_path / "e.yml").read_text())["evidence"]["live"]
    assert live["ebola"]["outbreaks"] == 1000
    assert live["germany"]["label"] == (
        "Preliminary: fine-tuned on 1,000 of 8,000 simulations."
    )


def test_the_runs_are_compared_at_matched_sizes() -> None:
    from paperpipe.evidence import slots_summary

    extra = _scores()
    extra = extra.loc[extra["run"] == "no-slots"].assign(
        checkpoint="plain2", outbreaks=2000, rel_crps=0.1, loss=0.01
    )
    out = slots_summary(pd.concat([_scores(), extra]), "one-model")
    # Both runs reached 1,000, so both rows give 1,000, not 2,000.
    assert out.iloc[2]["BDBV CRPS"] == "1.00 / 1.00 (1,000)"
    assert out.iloc[1]["BDBV CRPS"] == "1.00 / 1.00 (1,000)"


def test_the_curve_keeps_the_hub_target_only() -> None:
    from paperpipe.evidence import checkpoint_rows

    rows = _scores()
    daily = rows.assign(target="daily", rel_crps=9.0, rel_wis=9.0)
    out = checkpoint_rows(pd.concat([rows, daily]), "one-model")
    assert (out["rel_wis"] < 9).all()
    assert not out.duplicated(["study", "level", "model", "checkpoint"]).any()
