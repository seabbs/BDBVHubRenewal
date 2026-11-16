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
