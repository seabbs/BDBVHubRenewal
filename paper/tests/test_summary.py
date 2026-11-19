import pandas as pd
import yaml

from paperpipe.meta import DUMMY_LABEL, read_meta, write_meta
from paperpipe.outputs import to_markdown
from paperpipe.summary import study_summary, write_summary


def _scores(**extra) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["KIT-frozen_baseline", "a", "b"],
            "rel_score": [1.0, 0.8, 0.6],
            "rel_log_score": [1.0, 0.9, 0.7],
            **extra,
        }
    )


def test_summary_names_the_best_model_against_the_baseline() -> None:
    out = study_summary({"study": "germany", "source": "real"}, _scores())
    assert out["best_model"] == "b"
    assert out["best_rel_score"] == 0.6
    assert out["baseline"] == "KIT-frozen_baseline"
    assert out["label"] == ""


def test_summary_uses_the_heldout_half_and_labels_dummy_data() -> None:
    # Held-out rows first, as a table sorted by model may have them.
    scores = pd.concat(
        [
            _scores(period="held out").assign(rel_score=[1.0, 0.5, 0.9]),
            _scores(period="development"),
        ]
    )
    out = study_summary({"study": "germany", "source": "dummy"}, scores)
    assert out["best_model"] == "a"
    assert out["label"] == DUMMY_LABEL


def test_meta_and_summary_round_trip(tmp_path) -> None:
    write_meta(tmp_path / "m.json", "ebola", "dummy")
    assert read_meta(tmp_path / "m.json")["source"] == "dummy"
    write_summary(tmp_path / "r.yml", "development", {"ebola": {"n": 1}})
    doc = yaml.safe_load((tmp_path / "r.yml").read_text())
    assert doc["results"]["mode"] == "development"
    assert doc["results"]["ebola"] == {"n": 1}


def test_markdown_tables_round_numbers_and_escape_pipes() -> None:
    md = to_markdown(pd.DataFrame({"model": ["a|b"], "x": [0.123456]}))
    lines = md.splitlines()
    assert lines[0] == "| model | x |"
    assert lines[2] == "| a\\|b | 0.123 |"


def test_markdown_tables_leave_missing_models_blank() -> None:
    md = to_markdown(pd.DataFrame({"model": ["randomwalk", None], "x": [1, 2]}))
    lines = md.splitlines()
    assert lines[2].startswith("| Random walk |")
    assert "nan" not in md.lower()


def test_summary_says_when_the_scored_part_has_no_scores_yet() -> None:
    scores = _scores(period="development")
    out = study_summary({"study": "ebola", "source": "real"}, scores)
    assert out["best_model"] == ""
    assert out["n_models"] == 0
    assert "no scored forecasts yet" in out["status"].lower()
    scored = study_summary({"study": "germany", "source": "real"}, _scores())
    assert scored["status"] == ""


def test_summary_says_which_part_its_numbers_come_from() -> None:
    germany = {"study": "germany", "source": "real"}
    ebola = {"study": "ebola", "source": "real"}
    assert study_summary(germany, _scores())["scope"] == "development half"
    held = _scores(period="held out")
    assert study_summary(germany, held)["scope"] == "held-out half"
    assert study_summary(ebola, _scores())["scope"] == (
        "release backtest before the scored rounds"
    )
    assert study_summary(ebola, held)["scope"] == "scored rounds"
    synthetic = {"study": "synthetic", "source": "real"}
    assert study_summary(synthetic, _scores(), mode="development")["scope"] == (
        "development simulations"
    )
    assert study_summary(synthetic, _scores(), mode="paper")["scope"] == (
        "held-out simulations"
    )


def test_summary_records_the_fine_tuned_checkpoint(tmp_path) -> None:
    path = tmp_path / "results.yml"
    write_summary(path, "paper", {}, finetuned="e8a1f5ae338b")
    doc = yaml.safe_load(path.read_text())["results"]
    assert doc["finetuned"] == "e8a1f5ae338b"
    write_summary(path, "paper", {})
    assert yaml.safe_load(path.read_text())["results"]["finetuned"] == "pending"


def test_summary_marks_each_study_it_has_as_shown() -> None:
    # The paper hides a study's results when its run has no entry for it.
    out = study_summary({"study": "germany", "source": "real"}, _scores())
    assert out["shown"] is True
