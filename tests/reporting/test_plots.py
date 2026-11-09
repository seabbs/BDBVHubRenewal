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
            "cutoff": v,
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


def test_model_styles_are_fixed_by_name() -> None:
    from bdbvhubrenewal.forecasters import list_models

    registered = {s.name for s in list_models()}
    assert set(plots.MODEL_STYLE) == registered
    # Each colour and dash pair belongs to one model.
    pairs = list(plots.MODEL_STYLE.values())
    assert len(set(pairs)) == len(pairs)
    # Main models have a colour of their own; variants share it, dashed.
    solid = [s for s, d in pairs if d == "solid"]
    assert len(set(solid)) == len(solid)
    assert {d for _, d in pairs} <= set(plots.DASHES)
    # The same style whichever models are shown with it.
    for mode in ("light", "dark"):
        alone = plots.model_colour("renewal", ["renewal"], mode)
        among = plots.model_colour("renewal", sorted(registered), mode)
        assert alone == among == plots.MODEL_COLOURS[mode][2]
    # A stand-in outside the table takes the first colour it leaves free.
    used = {s for s, _ in pairs}
    flat = plots.model_colour("flat", ["flat"])
    assert flat == plots.MODEL_COLOURS["light"][max(used) + 1]
    assert plots.model_dash("flat") == ""


def test_mpl_dash_reads_the_svg_dash() -> None:
    assert plots.mpl_dash("") == "-"
    assert plots.mpl_dash("6 3") == (0, (6.0, 3.0))


def test_variants_are_hatched_and_hollow() -> None:
    assert plots.bar_style("#123456", "") == {"color": "#123456"}
    assert plots.bar_style("#123456", "6 3")["hatch"]
    assert plots.bar_style("#123456", "6 3")["facecolor"] == "none"
    assert plots.point_style("#123456", "")["mfc"] == "#123456"
    assert plots.point_style("#123456", "6 3")["mfc"] == "none"


def _learnt(model: str, vintage: str, location: str) -> pd.DataFrame:
    days = [
        {"x": str(d), "quantity": "reported_share", "q50": d / 10}
        for d in range(11)
    ]
    weeks = [{"x": w, "quantity": "mean_delay", "q50": 6.0} for w in WEEKS]
    rows = pd.DataFrame(days + weeks)
    return rows.assign(
        model=model,
        vintage=vintage,
        location=location,
        q5=rows["q50"] * 0.9,
        q95=rows["q50"] * 1.1,
    )


def test_renewal_delay_has_a_row_per_fit_at_its_latest_origin() -> None:
    learnt = pd.concat(
        [
            _learnt("renewal", "2026-10-24", "shared"),
            _learnt("renewal", "2026-11-01", "shared"),
            _learnt("renewalbvd", "2026-11-01", "shared"),
            _learnt("renewalsp", "2026-11-01", "CD-NK"),
            _learnt("renewalsp", "2026-11-01", "CD-IT"),
        ],
        ignore_index=True,
    )
    baseline = pd.DataFrame(
        {"location": ["CD", "CD"], "age": [0, 1], "share": [0.1, 0.2]}
    )
    names = {"CD-IT": "Ituri", "CD-NK": "Nord Kivu"}
    fig = plots.plot_renewal_delay(learnt, baseline, names)
    left = fig.axes[0::2]
    titles = [ax.get_title(loc="left") for ax in left]
    assert titles == [
        "Shared delay: renewal, renewalbvd",
        "Ituri: renewalsp",
        "Nord Kivu: renewalsp",
    ]
    # The baseline and two models in the shared row; the share stops the
    # day before the longest delay.
    assert len(left[0].lines) == 3
    assert max(left[1].lines[-1].get_xdata()) == 9
    assert fig.get_suptitle() == "Fits at 2026-11-01"
    plt.close(fig)
