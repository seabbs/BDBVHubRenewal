from make_figures import render


def test_render_writes_every_figure(tmp_path) -> None:
    paths = render(tmp_path, ["ebola"])
    assert {p.name for p in paths} == {
        "ebola-results.png",
        "ebola-si-location-horizon.png",
        "ebola-si-location-horizon-log.png",
        "ebola-si-wis-over-time.png",
    }
    assert all(p.stat().st_size > 0 for p in paths)
