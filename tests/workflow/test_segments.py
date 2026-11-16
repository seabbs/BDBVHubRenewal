"""Training segments: one per checkpoint prefix, resumed in turn."""

import pytest
from flowlib import segments


def test_each_prefix_resumes_from_the_one_before():
    assert segments.segments([250, 500, 1000]) == [
        (None, 250),
        (250, 500),
        (500, 1000),
    ]


def test_prefixes_are_sorted_and_unique():
    assert segments.segments([1000, 250, 250]) == [(None, 250), (250, 1000)]


def test_prefixes_must_end_chunks():
    segments.check_chunks([250, 500, 750, 1000], 250)
    with pytest.raises(ValueError, match="600"):
        segments.check_chunks([250, 600], 250)


def test_round_seeds_split_training_and_held_out():
    train, held = segments.round_seeds(4, base=30000, train=40, held=4)
    assert train == list(range(30400, 30440))
    assert held == [30440, 30441, 30442, 30443]


def test_seeds_for_counts_kept_outbreaks_in_list_order():
    kept = {30101: 25, 30100: 20, 30103: 25, 30102: 25}
    order = [30101, 30100, 30103, 30102]
    assert segments.seeds_for(45, order, kept.__getitem__) == [30101, 30100]
    # An exclusion pushes the prefix into the next batch.
    assert segments.seeds_for(46, order, kept.__getitem__) == [
        30101,
        30100,
        30103,
    ]


def test_seeds_for_stops_asking_once_it_has_enough():
    asked = []

    def kept(seed):
        asked.append(seed)
        return 25

    segments.seeds_for(50, [1, 2, 3, 4], kept)
    assert asked == [1, 2]


def test_seeds_for_fails_when_the_list_is_too_short():
    with pytest.raises(ValueError, match="only 50"):
        segments.seeds_for(60, [1, 2], lambda _s: 25)


def test_kept_outbreaks_leave_out_exclusions(tmp_path):
    settings = tmp_path / "settings-30101.csv"
    settings.write_text(
        "outbreak,r0\n30101-0001,1\n30101-0002,1\n30101-0003,1\n"
    )
    assert segments.kept_outbreaks(settings, {"30101-0002"}) == 2
    assert segments.kept_outbreaks(settings, set()) == 3


def test_excluded_reads_a_list_or_a_csv(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("30101-0002\n\n30101-0003\n")
    table = tmp_path / "table.csv"
    table.write_text("simulation,reason\n30101-0004,flat\n")
    assert segments.excluded(plain) == {"30101-0002", "30101-0003"}
    assert segments.excluded(table) == {"30101-0004"}
    assert segments.excluded(tmp_path / "none.txt") == set()


def _state(path, done):
    path.mkdir(parents=True)
    (path / "state.json").write_text(f'{{"done": {done}}}')


def test_a_segment_starts_from_the_prefix_before(tmp_path):
    run, prev = tmp_path / "state", tmp_path / "n250" / "state"
    _state(prev, 250)
    assert segments.resume_state(run, prev, 250) == "copied"
    assert (run / "state.json").read_text() == '{"done": 250}'


def test_a_state_further_on_is_kept(tmp_path):
    # A segment that stopped part way resumes from its own saves.
    run, prev = tmp_path / "state", tmp_path / "n250" / "state"
    _state(prev, 250)
    _state(run, 375)
    assert segments.resume_state(run, prev, 250) == "kept"
    assert (run / "state.json").read_text() == '{"done": 375}'


def test_a_state_behind_is_replaced(tmp_path):
    run, prev = tmp_path / "state", tmp_path / "n500" / "state"
    _state(prev, 500)
    _state(run, 250)
    assert segments.resume_state(run, prev, 500) == "copied"
    assert (run / "state.json").read_text() == '{"done": 500}'


def test_a_run_state_alone_is_enough(tmp_path):
    # The live run's checkpoints have no state of their own.
    run = tmp_path / "state"
    _state(run, 875)
    assert segments.resume_state(run, tmp_path / "n750" / "state", 750) == (
        "kept"
    )


def test_no_state_to_resume_from_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="n750"):
        segments.resume_state(
            tmp_path / "state", tmp_path / "n750" / "state", 750
        )


def test_the_first_segment_starts_from_base(tmp_path):
    assert segments.resume_state(tmp_path / "state", None, None) == "base"


def test_reachable_prefixes_fit_the_seed_list():
    assert segments.reachable(500, [1] * 21, lambda _s: 25)
    assert not segments.reachable(2000, [1] * 21, lambda _s: 25)


def test_exclusions_are_gathered_into_one_file(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("simulation,reason,evidence\n30101-0002,flat,x\n")
    b = tmp_path / "b.csv"
    b.write_text("simulation,reason,evidence\n")
    out = tmp_path / "excluded.csv"
    segments.gather_exclusions([a, b, tmp_path / "none.csv"], out)
    assert segments.excluded(out) == {"30101-0002"}
