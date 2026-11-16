"""The German nowcast benchmark page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.germany import (
    SCORE_COLUMNS,
    load_germany_config,
)
from bdbvhubrenewal.reporting.germany import germany_page

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_germany_config(ROOT / "config" / "germany.toml")


def _rows(models: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    rows = []
    for model, wis in models:
        for place, pool in (
            ("DE", "national"),
            ("DE-BY", "states"),
            ("80+", "age"),
        ):
            for h in (0, -7):
                end = pd.Timestamp("2021-12-01") + pd.Timedelta(days=h)
                rows.append(
                    {
                        "model": model,
                        "target": "40-day",
                        "origin": "2021-12-01",
                        "truth_vintage": str(end + pd.Timedelta(days=40))[:10],
                        "location": place,
                        "target_end_date": str(end)[:10],
                        "horizon": h,
                        "truth": 100.0,
                        **dict.fromkeys(SCORE_COLUMNS, 0.5),
                        "wis": wis,
                        "pool": pool,
                    }
                )
    return pd.DataFrame(rows)


LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


def _forecasts(models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model in models:
        for origin in ("2021-11-22", "2021-11-23", "2022-02-14"):
            for place in ("DE", "DE-BY", "80+"):
                for h in (0, -7):
                    end = pd.Timestamp(origin) + pd.Timedelta(days=h)
                    for i, q in enumerate(LEVELS):
                        rows.append(
                            {
                                "model": model,
                                "origin": origin,
                                "location": place,
                                "target_end_date": str(end)[:10],
                                "horizon": h,
                                "quantile": q,
                                "value": 10.0 * (i + 1),
                                "truth": 42.0,
                                "truth_vintage": "2022-03-01",
                            }
                        )
    return pd.DataFrame(rows)


def _scores(path: Path) -> None:
    path.mkdir(parents=True)
    ours = ("randomwalk", "randomwalknowcast")
    _forecasts(ours).to_csv(path / "forecasts.csv.gz", index=False)
    teams = ("KIT", "KIT-frozen_baseline")
    _forecasts(teams).to_csv(path / "teams-forecasts.csv.gz", index=False)
    pd.DataFrame(
        {
            "origin": "2021-11-22",
            "location": "DE",
            "target_end_date": ["2021-11-21", "2021-11-22"],
            "total": [30, 20],
        }
    ).to_csv(path / "asof.csv.gz", index=False)
    ours = _rows((("randomwalk", 10.0), ("randomwalknowcast", 5.0)))
    held = ours.assign(origin="2022-03-01")
    pd.concat([ours, held]).to_csv(path / "scores.csv.gz", index=False)
    _rows((("KIT", 4.0),)).to_csv(path / "teams.csv.gz", index=False)


def test_page_tables_and_summaries(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    table = (gen / "germany-scores.md").read_text()
    assert CONFIG.split_date in table
    assert "randomwalknowcast" in table
    assert "| KIT " in table
    assert "rel_wis" in table
    assert "| wis " not in table
    assert "0.8" in table
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    assert [p["id"] for p in data["pools"]] == ["national", "states", "age"]
    names = {p["id"]: p["name"] for p in data["places"]}
    assert names["DE-BY"] == "Bavaria"
    assert names["80+"] == "Age 80+"
    assert data["models"][data["baseline"]]["name"] == "randomwalknowcast"
    assert data["origins"] == ["2021-12-01"]
    assert [m["name"] for m in data["models"]][-1] == "KIT"
    keys = [m["key"] for m in data["metrics"]]
    assert keys == ["wis", "ae_median", "coverage_50", "coverage_95"]
    assert CONFIG.commit[:7] in (gen / "germany-sources.md").read_text()


def test_hidden_models_start_switched_off(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    assert CONFIG.hidden == ("randomwalk", "KIT-frozen_baseline")
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    on = {m["name"]: m["on"] for m in data["models"]}
    assert not on["randomwalk"]
    assert on["randomwalknowcast"]
    assert on["KIT"]


def test_nowcast_explorer_shows_development_plot_dates(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    assert "explorer-germany.json" in (gen / "germany-nowcasts.md").read_text()
    data = json.loads((fig / "explorer-germany.json").read_text())
    # Every 7th development date from the first; later dates are held out.
    assert data["rounds"] == ["2021-11-22"]
    on = {m["name"]: m["on"] for m in data["models"]}
    assert on == {
        "randomwalk": False,
        "randomwalknowcast": True,
        "KIT": True,
        "KIT-frozen_baseline": False,
    }
    assert data["levels"] == ["National", "States", "Age groups"]
    assert data["outer"] == "95%"
    # q05 and q95 carry the 2.5% and 97.5% quantiles.
    assert {(r[4], r[8]) for r in data["q"]} == {(10.0, 70.0)}
    assert data["data"]["DE"][0][1] == 42
    assert data["asof"] == [
        [0, "DE", "2021-11-21", 30],
        [0, "DE", "2021-11-22", 20],
    ]


def test_page_says_when_not_run(tmp_path: Path) -> None:
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, tmp_path / "missing", CONFIG)
    assert "Not run yet" in (gen / "germany-scores.md").read_text()


def test_every_model_has_its_own_colour_and_dash() -> None:
    from bdbvhubrenewal.evaluation.germany import TEAM_NAMES
    from bdbvhubrenewal.reporting.germany import model_styles

    models = [*CONFIG.models, *TEAM_NAMES.values()]
    styles = model_styles(models, CONFIG.models)
    assert set(styles) == set(models)
    pairs = [(light, dash) for light, _, dash in styles.values()]
    assert len(set(pairs)) == len(models)


FT = "fm_chronos_daily_ragged_ft"


def _with_ft(res: Path, tag: str | None = "chronos-ft-bbb") -> None:
    """Add the stored fine-tuned model, as `germany-ft-load` does."""
    fc = pd.read_csv(res / "forecasts.csv.gz", dtype=str)
    ft = _forecasts((FT,))
    ft = ft.loc[ft["origin"] < CONFIG.split_date]
    pd.concat([fc, ft.astype(str)]).to_csv(
        res / "forecasts.csv.gz", index=False
    )
    sc = pd.read_csv(res / "scores.csv.gz", dtype=str)
    pd.concat([sc, _rows(((FT, 2.5),)).astype(str)]).to_csv(
        res / "scores.csv.gz", index=False
    )
    meta = {"tag": tag, "pinned": "chronos-ft-bbb", "model": FT}
    (res / "ft-checkpoint.json").write_text(json.dumps(meta))


def test_the_stored_ft_model_is_scored_on_the_baselines_cells(
    tmp_path: Path,
) -> None:
    res = tmp_path / "germany"
    _scores(res)
    _with_ft(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    table = (gen / "germany-scores.md").read_text()
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in table.splitlines()
        if line.startswith("|") and "---" not in line
    ]
    head, body = rows[0], rows[1:]
    got = pd.DataFrame(body, columns=head)
    ns = got.pivot(index="pool", columns="model", values="n")
    assert FT in ns.columns
    # Every model on the page has the same n at each level.
    assert (ns.nunique(axis=1) == 1).all(), ns
    rel = got.loc[got["model"] == FT].set_index("pool")["rel_wis"]
    assert set(rel.astype(float)) == {0.5}
    data = json.loads((fig / "explorer-germany.json").read_text())
    assert FT in {m["name"] for m in data["models"]}


def test_the_page_names_the_ft_checkpoint(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    _with_ft(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    note = (gen / "germany-ft.md").read_text()
    assert "chronos-ft-bbb" in note
    assert FT in note
    assert "releases/tag/chronos-ft-bbb" in note


def test_the_page_says_when_no_ft_checkpoint_is_stored(
    tmp_path: Path,
) -> None:
    res = tmp_path / "germany"
    _scores(res)
    _with_ft(res, tag=None)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    assert "not stored yet" in (gen / "germany-ft.md").read_text()


def test_the_page_writes_no_held_out_date(tmp_path: Path) -> None:
    from bdbvhubrenewal.reporting.germany import held_out_leaks

    res = tmp_path / "germany"
    _scores(res)
    _with_ft(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    files = [*fig.glob("*germany*.json"), *gen.glob("germany-*.md")]
    assert len(files) >= 7
    assert held_out_leaks(files, CONFIG.split_date) == []


def test_the_leak_check_finds_held_out_dates(tmp_path: Path) -> None:
    from bdbvhubrenewal.reporting.germany import held_out_leaks

    split = CONFIG.split_date
    ok = tmp_path / "ok.json"
    # Truth vintages (`releases`) are report dates of the truth of
    # development sums, so they may fall after the split.
    ok.write_text(
        json.dumps({"origins": ["2022-02-09"], "releases": ["2022-03-21"]})
    )
    bad = tmp_path / "scoresummary-germany-latest.json"
    bad.write_text(json.dumps({"origins": ["2022-02-09", split]}))
    page = tmp_path / "index.html"
    # The split date itself is named in the page's prose.
    page.write_text(f"<p>Nowcast dates before {split} only.</p>")
    leaky = tmp_path / "leaky.html"
    leaky.write_text("<td>2022-02-11</td>")
    assert held_out_leaks([ok, page], split) == []
    found = held_out_leaks([ok, bad, page, leaky], split)
    assert len(found) == 2
    assert any("2022-02-11" in f for f in found)
    assert any(split in f and "origins" in f for f in found)


def test_the_page_refuses_to_write_a_held_out_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal.reporting import germany as page

    res = tmp_path / "germany"
    _scores(res)
    # Were the development filter to fail, the guard stops the page.
    monkeypatch.setattr(
        page, "_development", lambda frame, _config: frame, raising=True
    )
    with pytest.raises(ValueError, match="held-out"):
        page.germany_page(
            tmp_path / "figures", tmp_path / "generated", res, CONFIG
        )


def test_every_row_scores_the_cells_all_its_models_share(
    tmp_path: Path,
) -> None:
    res = tmp_path / "germany"
    _scores(res)
    _with_ft(res)
    # A team missing one national cell: every model's national row drops
    # it, so each row of a level has the same n.
    teams = pd.read_csv(res / "teams.csv.gz", dtype=str)
    gap = _rows((("SU", 3.0),)).astype(str)
    gap = gap.loc[~((gap["pool"] == "national") & (gap["horizon"] == "-7"))]
    pd.concat([teams, gap]).to_csv(res / "teams.csv.gz", index=False)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    table = (gen / "germany-scores.md").read_text()
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in table.splitlines()
        if line.startswith("|") and "---" not in line
    ]
    got = pd.DataFrame(rows[1:], columns=rows[0])
    n = got.pivot(index="pool", columns="model", values="n").astype(int)
    assert "SU" in n.columns
    assert (n.nunique(axis=1) == 1).all(), n
    assert n.loc["national"].iloc[0] == 1
    assert n.loc["states"].iloc[0] == 2
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    pool = [p["pool"] for p in data["places"]]
    # The summary's national rows hold the shared cell only (horizon 0).
    national = {
        h
        for r, h in zip(data["rows"], data["horizons"], strict=True)
        if pool[r[3]] == "national"
    }
    assert national == {0}
