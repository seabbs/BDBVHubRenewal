from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_history.py"


@pytest.fixture(scope="module")
def sh():
    spec = importlib.util.spec_from_file_location("score_history", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_history"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_source_sha_is_read_from_main_deploys_only(sh) -> None:
    sha = "0996814" + "0" * 33
    msg = f"Deploying to gh-pages from @ seabbs/BDBVHubRenewal@{sha} 🚀"
    assert sh.source_sha(msg) == sha
    assert sh.source_sha("Deploy preview for PR 102 🛫") is None
    assert sh.source_sha("Remove preview for PR 93 🛬") is None


def _summary() -> dict[str, object]:
    # Models: randomwalk (baseline), good, "as reported".
    # Places: CD (national), CD-IT and CD-NK (provinces).
    # Rows: model, origin, release, place, week, wis, cov50; `horizons`
    # gives each row's horizon.
    return {
        "models": [
            {"name": "randomwalk"},
            {"name": "good"},
            {"name": "as reported"},
        ],
        "baseline": 0,
        "metrics": [
            {"key": "wis", "relative": True},
            {"key": "coverage_50", "relative": False},
        ],
        "origins": ["2026-10-24", "2026-10-31"],
        "releases": ["2026-11-01"],
        "weeks": ["2026-10-24"],
        "places": [
            {"id": "CD", "pool": "national"},
            {"id": "CD-IT", "pool": "provinces"},
            {"id": "CD-NK", "pool": "provinces"},
        ],
        "rows": [
            [0, 0, 0, 0, 0, 10.0, 1.0],
            [0, 1, 0, 0, 0, 30.0, 0.0],
            [1, 0, 0, 0, 0, 5.0, 1.0],
            [1, 1, 0, 0, 0, 30.0, 1.0],
            [0, 0, 0, 1, 0, 4.0, 1.0],
            [1, 0, 0, 1, 0, 2.0, 0.0],
            # No baseline forecast for CD-NK: left out.
            [1, 0, 0, 2, 0, 100.0, 0.0],
            [2, 0, 0, 0, 0, 1.0, 1.0],
        ],
        "horizons": [0, 1, 0, 1, 0, 0, 0, 0],
    }


def _full(df):
    """Every forecast, over all horizons."""
    return df.loc[(df["horizon"] == "all") & ~df["common"]]


def test_summary_scores_match_on_baseline_forecasts(sh) -> None:
    every = sh.summary_scores(_summary())
    assert set(every["baseline"]) == {"randomwalk", "good"}
    out = every.loc[every["baseline"] == "randomwalk"]
    out = out.set_index(["pool", "model"])
    assert set(out.index.get_level_values("model")) == {"randomwalk", "good"}
    nat = out.loc[("national", "good")]
    assert nat["n"] == 2
    assert nat["wis"] == pytest.approx(17.5)
    assert nat["rel_wis"] == pytest.approx(35 / 40)
    assert nat["base_wis"] == pytest.approx(20.0)
    assert nat["coverage_50"] == pytest.approx(1.0)
    prov = out.loc[("provinces", "good")]
    assert prov["n"] == 1
    assert prov["rel_wis"] == pytest.approx(0.5)
    assert out.loc[("national", "randomwalk"), "rel_wis"] == 1.0
    assert "rel_coverage_50" not in out.columns
    # Relative to good, randomwalk is twice as bad nationally.
    vs_good = every.loc[every["baseline"] == "good"].set_index(
        ["pool", "model"]
    )
    assert vs_good.loc[("national", "randomwalk"), "rel_wis"] == pytest.approx(
        40 / 35
    )


def test_history_flags_changes_in_what_is_scored(sh) -> None:
    base = _summary()
    later = _summary()
    later["origins"] = ["2026-10-31", "2026-11-07"]
    # Same origins and releases, but fewer of good's forecasts scored.
    fewer = _summary()
    fewer["origins"] = later["origins"]
    keep = [r[:2] != [1, 1] for r in fewer["rows"]]
    fewer["rows"] = [r for r, k in zip(fewer["rows"], keep, strict=True) if k]
    fewer["horizons"] = [
        h for h, k in zip(fewer["horizons"], keep, strict=True) if k
    ]
    deploys = [
        ("a" * 40, "2026-11-04T10:00:00+00:00", "p1", "latest", base),
        ("a" * 40, "2026-11-04T11:00:00+00:00", "p2", "latest", base),
        ("b" * 40, "2026-11-05T10:00:00+00:00", "p3", "latest", later),
        ("c" * 40, "2026-11-06T10:00:00+00:00", "p4", "latest", fewer),
    ]
    every = _full(sh.history(deploys))
    out = every.loc[every["baseline"] == "randomwalk"]
    first = out.loc[(out["model"] == "good") & (out["pool"] == "national")]
    assert list(first["source_sha"].str[:1]) == ["a", "a", "b", "c"]
    assert list(first["origins"])[2] == "2026-10-31;2026-11-07"
    assert list(first["break"]) == [True, False, True, True]
    assert list(first["releases"]) == ["2026-11-01"] * 4
    # The baseline's forecasts are unchanged at the last deploy.
    rw = out.loc[(out["model"] == "randomwalk") & (out["pool"] == "national")]
    assert list(rw["break"]) == [True, False, True, False]
    # Relative to good, whose forecasts changed, randomwalk breaks too.
    vs_good = every.loc[
        (every["baseline"] == "good")
        & (every["model"] == "randomwalk")
        & (every["pool"] == "national")
    ]
    assert list(vs_good["break"]) == [True, False, True, True]


def test_history_splits_by_horizon(sh) -> None:
    out = sh.history([("a" * 40, "t", "p1", "latest", _summary())])
    out = out.loc[
        ~out["common"]
        & (out["baseline"] == "randomwalk")
        & (out["model"] == "good")
        & (out["pool"] == "national")
    ].set_index("horizon")
    assert set(out.index) == {"all", "0", "1"}
    assert out.loc["0", "rel_wis"] == pytest.approx(0.5)
    assert out.loc["1", "rel_wis"] == pytest.approx(1.0)
    assert out.loc["1", "n"] == 1


def _good_national(df):
    keep = (
        (df["horizon"] == "all")
        & (df["baseline"] == "randomwalk")
        & (df["model"] == "good")
        & (df["pool"] == "national")
    )
    return df.loc[keep]


def test_common_forecasts_keep_the_scored_set_fixed(sh) -> None:
    more = _summary()
    # A third origin, scored by both, and good's first forecast refitted.
    more["origins"] = ["2026-10-24", "2026-10-31", "2026-11-07"]
    more["rows"] = [*more["rows"], [0, 2, 0, 0, 0, 8.0, 1.0]]
    more["rows"].append([1, 2, 0, 0, 0, 2.0, 1.0])
    more["rows"][2] = [1, 0, 0, 0, 0, 7.0, 1.0]
    more["horizons"] = [*more["horizons"], 0, 0]
    out = _good_national(
        sh.history(
            [
                ("a" * 40, "t1", "p1", "latest", _summary()),
                ("b" * 40, "t2", "p2", "latest", more),
            ]
        )
    )
    full = out.loc[~out["common"]]
    common = out.loc[out["common"]]
    assert list(full["n"]) == [2, 3]
    assert list(common["n"]) == [2, 2]
    assert list(common["wis"]) == pytest.approx([17.5, 18.5])
    assert list(common["break"]) == [True, False]


def test_common_window_starts_after_the_last_disjoint_deploy(sh) -> None:
    moved = _summary()
    moved["origins"] = ["2026-11-14", "2026-11-21"]
    out = _good_national(
        sh.history(
            [
                ("a" * 40, "t1", "p1", "latest", _summary()),
                ("b" * 40, "t2", "p2", "latest", _summary()),
                ("c" * 40, "t3", "p3", "latest", moved),
            ]
        )
    )
    common = out.loc[out["common"]]
    assert list(common["source_sha"].str[:1]) == ["c"]
    assert list(common["n"]) == [2]


def test_baselines_can_be_restricted(sh) -> None:
    out = sh.history(
        [("a" * 40, "t", "p1", "latest", _summary())], baselines=["randomwalk"]
    )
    assert set(out["baseline"]) == {"randomwalk"}


def _two_deploys(sh):
    later = _summary()
    later["origins"] = ["2026-10-31", "2026-11-07"]
    return sh.history(
        [
            ("a" * 40, "2026-11-04T10:00:00+00:00", "p1", "latest", _summary()),
            ("b" * 40, "2026-11-05T10:00:00+00:00", "p2", "latest", later),
        ]
    )


def test_chart_marks_breaks_and_names_commits(sh) -> None:
    charts = sh.chart_data(_two_deploys(sh), {"good": ("#111", "#222", "")})
    assert set(charts) == {"latest"}
    d = charts["latest"]
    assert d["kind"] == "line"
    assert [m["name"] for m in d["models"]] == ["randomwalk", "good"]
    assert d["models"][1]["light"] == "#111"
    assert d["scales"] == ["national", "provinces"]
    assert d["baseline"] == 0
    assert {
        f"{b}|{p}|all horizons"
        for b in ("randomwalk", "good")
        for p in ("national", "provinces")
    } <= set(d["sets"])
    assert d["common"] is True
    assert d["baselines"] == ["randomwalk", "good"]
    assert d["splits"] == ["all horizons", "horizon 0", "horizon 1"]
    assert d["split"] == "horizon 0"
    assert "randomwalk|national|horizon 1|common" in d["sets"]
    nat = d["sets"]["randomwalk|national|all horizons"]
    assert nat["x"] == ["11-04 10:00 aaaaaaa *", "11-05 10:00 bbbbbbb *"]
    assert nat["marks"] == [0, 1]
    good = [r for r in nat["rows"] if r[0] == 1]
    assert good[0][:3] == [1, 0, 2]
    keys = [m["key"] for m in d["metrics"]]
    assert good[0][3 + keys.index("rel_wis")] == pytest.approx(35 / 40)
    rw = d["sets"]["good|national|all horizons"]["rows"][0]
    assert rw[0] == 0
    assert rw[3 + keys.index("rel_wis")] == pytest.approx(40 / 35, rel=1e-3)


def test_page_has_chart_table_and_commit_links(sh) -> None:
    page = sh.page_markdown(_two_deploys(sh), "seabbs/BDBVHubRenewal")
    assert 'class="checkschart"' in page
    assert "../../figures/scorehistory-latest.json" in page
    assert 'class="tablefilter"' in page
    assert (
        "[aaaaaaa](https://github.com/seabbs/BDBVHubRenewal/commit/"
        + "a" * 40
        + ")"
    ) in page


def test_page_says_when_history_is_unavailable(sh) -> None:
    page = sh.page_markdown(None, "seabbs/BDBVHubRenewal")
    assert "unavailable" in page
    assert "checkschart" not in page


def test_current_build_is_read_from_its_figures(sh, tmp_path: Path) -> None:
    (tmp_path / "scoresummary-backtest-complete.json").write_text(
        json.dumps(_summary())
    )
    out = list(sh.current_deploys(tmp_path, "c" * 40, "2026-11-09T16:00"))
    assert [(d[0], d[2], d[3]) for d in out] == [("c" * 40, "", "complete")]


def test_script_does_not_import_the_package() -> None:
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('s', {str(SCRIPT)!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print('bdbvhubrenewal' in sys.modules, 'jax' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


def test_common_window_keeps_the_most_forecast_deploys(sh) -> None:
    # B adds four national forecasts by both models: B alone has 7
    # common forecasts (7 forecast-deploys) and A to B 3 (6), so the
    # window is B alone.
    more = _summary()
    more["origins"] = [f"2026-1{i}-01" for i in range(6)]
    more["origins"][:2] = ["2026-10-24", "2026-10-31"]
    for o in (2, 3, 4, 5):
        more["rows"].append([0, o, 0, 0, 0, 8.0, 1.0])
        more["rows"].append([1, o, 0, 0, 0, 4.0, 1.0])
        more["horizons"] += [0, 0]
    out = _good_national(
        sh.history(
            [
                ("a" * 40, "t1", "p1", "latest", _summary()),
                ("b" * 40, "t2", "p2", "latest", more),
            ]
        )
    )
    common = out.loc[out["common"]]
    assert list(common["source_sha"].str[:1]) == ["b"]
    assert list(common["n"]) == [6]
