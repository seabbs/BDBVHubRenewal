from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdbvhubrenewal.reporting import plots

WEEKS = ["2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31"]


def _target(vintages: list[str]) -> pd.DataFrame:
    rows = [
        {
            "kind": "real",
            "vintage": v,
            "location": "CD",
            "location_name": "DRC",
            "target_end_date": w,
            "total": 10 + 5 * i,
        }
        for v in vintages
        for i, w in enumerate(WEEKS[:4])
    ]
    return pd.DataFrame(rows)


def _forecasts(vintage: str, ref: str, top: float) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 100
    values = rng.poisson(20, size=(n, 2)).astype(float)
    values[0, 1] = top  # one path far above the band
    return pd.DataFrame(
        {
            "model": "m",
            "kind": "real",
            "vintage": vintage,
            "reference_date": ref,
            "horizon": np.tile([0, 1], n),
            "location": "CD",
            "target_end_date": np.tile(WEEKS[3:], n),
            "sample": np.repeat(np.arange(1, n + 1), 2),
            "value": values.reshape(-1),
        }
    )


def test_grid_has_no_dashed_line_when_origin_is_the_truth() -> None:
    fig = plots.plot_location_grid(
        _forecasts("2026-10-24", "2026-10-31", 20),
        _target(["2026-10-24"]),
        ["CD"],
        truth_vintage="2026-10-24",
    )
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "Data at the origin" not in labels
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert "Data at the origin" not in texts
    plt.close(fig)


def test_grid_axis_ignores_a_single_far_path() -> None:
    fig = plots.plot_location_grid(
        _forecasts("2026-10-24", "2026-10-31", 5000),
        _target(["2026-10-24"]),
        ["CD"],
        truth_vintage="2026-10-24",
    )
    assert fig.axes[0].get_ylim()[1] < 100
    assert fig.axes[0].get_ylim()[0] == 0
    plt.close(fig)


def test_backtest_panels_are_titled_by_origin() -> None:
    fc = pd.concat(
        [
            _forecasts("2026-10-17", "2026-10-24", 20),
            _forecasts("2026-10-24", "2026-10-31", 20),
        ]
    )
    fig = plots.plot_forecasts(
        fc, _target(["2026-10-17", "2026-10-24"]), "CD", models=["m"]
    )
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["CD, origin 2026-10-17", "CD, origin 2026-10-24"]
    plt.close(fig)


def test_delay_check_draws_growth_dispersion_and_removals(
    daily: pd.DataFrame,
) -> None:
    fig = plots.plot_delay_check(daily, "2026-10-24", ["CD", "CD-NK"])
    left, mid, right = fig.axes
    assert left.get_yscale() == "log"
    assert len(left.get_lines()) >= 2
    assert len(mid.get_lines()) >= 1
    assert len(right.get_lines()) >= 1
    plt.close(fig)


def test_plot_downward_updates_has_three_scales() -> None:
    target = _target(["2026-10-24", "2026-10-31"])
    target = pd.concat(
        [target, target.assign(location="CD-IT", location_name="Ituri")]
    )
    zones = target.loc[target["location"] == "CD-IT"].assign(
        location="CD-IT-01"
    )
    fig = plots.plot_downward_updates(target, zones, mode="dark")
    titles = [ax.get_title() for ax in fig.axes[:3]]
    assert titles == ["Country", "Provinces", "Health zones"]
    assert fig.legends[0].get_texts()[0].get_text() == (
        "2026-10-24 to 2026-10-31"
    )
    plt.close(fig)


def test_plot_imputed_by_release_one_line_per_place() -> None:
    target = _target(["2026-10-24", "2026-10-31"]).assign(imputed=2)
    fig = plots.plot_imputed_by_release(target)
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)
