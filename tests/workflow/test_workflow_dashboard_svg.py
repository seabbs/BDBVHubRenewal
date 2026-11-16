"""The dashboard's SVG line charts."""

from flowlib.dashboard import svg


def test_a_chart_draws_each_series_and_its_legend():
    out = svg.chart(
        "Loss",
        [
            ("run a", "#111", [(0, 1.0), (10, 0.5)], "pts"),
            ("run b", "#222", [(0, 1.2), (10, 0.7)], "line", "dash"),
        ],
        marks=[(5, "n250")],
        ylog=True,
        hlines=(1.0,),
    )
    assert out.count("<polyline") == 2
    assert "run b (dashed)" in out
    assert "n250" in out
    assert "<title>run a step 10: 0.5000</title>" in out


def test_a_chart_with_no_points_says_so():
    assert "no data" in svg.chart("Loss", [("a", "#111", [(0, None)], "line")])


def test_a_log2_axis_takes_the_given_ticks():
    out = svg.chart(
        "Held-out",
        [("a", "#111", [(250, 0.5), (1000, 0.4)], "pts")],
        xticks=[(125, "zero-shot"), (250, "250"), (1000, "1000")],
    )
    assert "zero-shot" in out


def test_rolling_means_and_nice_ticks():
    assert svg.rolling([(0, 1), (1, 3), (2, 5)], 2) == [(0, 1), (1, 2), (2, 4)]
    assert svg.nice(0, 1)[:3] == [0, 0.2, 0.4]
