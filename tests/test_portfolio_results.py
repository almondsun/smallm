import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.render_results_chart import load_results, render_svg

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_results_match_published_contract():
    payload = load_results()

    assert [row["name"] for row in payload["corpora"]] == [
        "Alice",
        "Peter Pan",
        "Hamlet",
        "Art of War",
        "Lincoln",
        "Frankenstein",
        "Douglass",
        "Origin",
    ]
    assert [row["seeds"] for row in payload["corpora"]] == [1, 1, 1, 3, 3, 3, 3, 3]
    assert [row["control"] for row in payload["corpora"]][-3:] == ["char136"] * 3
    assert all(row["byte_bpe512"] < row["character"] for row in payload["corpora"])
    assert "ByteBPE512 lowers sealed-test mean BPC" in render_svg(payload)


def test_portfolio_results_reject_invalid_values(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric": "bits_per_character",
                "corpora": [
                    {
                        "name": "bad",
                        "character": float("inf"),
                        "byte_bpe512": 2,
                        "seeds": 1,
                        "control": "char128",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="character"):
        load_results(path)


def test_checked_in_chart_is_current():
    payload = load_results()
    chart = ROOT / "docs" / "assets" / "sealed-test-bpc.svg"

    assert chart.read_text(encoding="utf-8") == render_svg(payload)


def test_chart_script_supports_direct_execution():
    completed = subprocess.run(
        [sys.executable, "scripts/render_results_chart.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "sealed-test-bpc.svg" in completed.stdout
