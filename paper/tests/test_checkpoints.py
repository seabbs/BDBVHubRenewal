import pandas as pd

from paperfigs.checkpoints import checkpoint_of, drop_checkpoints


def test_checkpoint_names_split_into_model_and_outbreaks() -> None:
    assert checkpoint_of("fm_chronos_daily_ragged_ft@4000") == (
        "fm_chronos_daily_ragged_ft",
        4000,
    )
    assert checkpoint_of("fm_chronos_daily_ragged_ft") is None
    assert checkpoint_of("team@x") is None


def test_drop_checkpoints_keeps_the_other_models() -> None:
    df = pd.DataFrame({"model": ["a", "a@1000", "b"]})
    assert drop_checkpoints(df)["model"].tolist() == ["a", "b"]
