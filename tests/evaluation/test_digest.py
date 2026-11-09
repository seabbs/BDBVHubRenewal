from __future__ import annotations

import ast
import shutil
from pathlib import Path

from bdbvhubrenewal.evaluation.digest import REPORT_ONLY, backtest_digest

ROOT = Path(__file__).resolve().parents[2]


def _digest_tree(root: Path) -> dict[str, Path]:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, root / name)
    files = {
        "code": root / "src/bdbvhubrenewal/evaluation/a.py",
        "report": root / "src/bdbvhubrenewal/reporting/report.py",
        "data": root / "data/target.csv",
        "sources": root / "data/sources.json",
        "raw": root / "data/raw/x.csv",
        "results": root / "results/forecasts.csv.gz",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a = 1\n")
    return files


def test_backtest_digest_follows_what_a_backtest_reads(
    tmp_path: Path,
) -> None:
    files = _digest_tree(tmp_path)
    before = backtest_digest(tmp_path)
    # Report-only code, the download record and raw downloads are not read.
    for name in ("report", "sources", "raw", "results"):
        files[name].write_text("a = 2\n")
        assert backtest_digest(tmp_path) == before, name
    assert backtest_digest(tmp_path, text="other") != before
    assert backtest_digest(tmp_path, extra=["results"]) != before
    for name in ("code", "data"):
        files[name].write_text("a = 3\n")
        after = backtest_digest(tmp_path)
        assert after != before, name
        before = after


def _reporting_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        found |= {
            n.rsplit(".", 1)[1]
            for n in names
            if n.startswith("bdbvhubrenewal.reporting.")
        }
    return found


def test_backtest_code_never_imports_report_only_modules() -> None:
    src = ROOT / "src" / "bdbvhubrenewal"
    report_only = {Path(p).stem for p in REPORT_ONLY}
    assert report_only
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in REPORT_ONLY:
            continue
        used = _reporting_imports(path) & report_only
        # The `report` command, which no backtest runs.
        if rel == "src/bdbvhubrenewal/cli.py":
            used -= {"report"}
        assert not used, f"{rel} imports {used}"
