from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "sealed_test_bpc.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "sealed-test-bpc.svg"


def load_results(path: Path = DATA_PATH) -> dict[str, Any]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("results must contain a JSON object")
    payload: dict[str, Any] = loaded
    if payload.get("schema_version") != 1 or payload.get("metric") != "bits_per_character":
        raise ValueError("results must use the sealed-test BPC schema")
    corpora = payload.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise ValueError("results must contain at least one corpus")
    names: set[str] = set()
    for row in corpora:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("each result must contain a corpus name")
        if row["name"] in names:
            raise ValueError("corpus names must be unique")
        names.add(row["name"])
        for key in ("character", "byte_bpe512"):
            value = row.get(key)
            if not isinstance(value, int | float) or isinstance(value, bool) or not 0 < value < 10:
                raise ValueError(f"{key} must be a finite positive BPC value")
        if not isinstance(row.get("seeds"), int) or row["seeds"] <= 0:
            raise ValueError("seeds must be a positive integer")
    return payload


def render_svg(payload: dict[str, Any]) -> str:
    rows = payload["corpora"]
    width, height = 900, 120 + 74 * len(rows)
    plot_left, plot_right = 190, 830
    minimum, maximum = 1.95, 2.36

    def x(value: float) -> float:
        return plot_left + (value - minimum) / (maximum - minimum) * (plot_right - plot_left)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Sealed-test bits per character by tokenizer</title>',
        '<desc id="desc">ByteBPE512 has lower bits per character than the character tokenizer on all five reported corpora.</desc>',
        '<rect width="100%" height="100%" fill="#0d1117" rx="12"/>',
        '<text x="32" y="42" fill="#f0f6fc" font-family="system-ui,sans-serif" font-size="24" font-weight="700">ByteBPE512 wins every sealed comparison</text>',
        '<text x="32" y="70" fill="#8b949e" font-family="system-ui,sans-serif" font-size="14">Lower bits per character is better · points are reported values or 3-seed means</text>',
    ]
    for tick in (2.0, 2.1, 2.2, 2.3):
        position = x(tick)
        lines.extend(
            [
                f'<line x1="{position:.1f}" y1="92" x2="{position:.1f}" y2="{height - 34}" stroke="#30363d"/>',
                f'<text x="{position:.1f}" y="{height - 12}" fill="#8b949e" font-family="system-ui,sans-serif" font-size="12" text-anchor="middle">{tick:.1f}</text>',
            ]
        )
    for index, row in enumerate(rows):
        y = 118 + index * 74
        character_x = x(float(row["character"]))
        bpe_x = x(float(row["byte_bpe512"]))
        lines.extend(
            [
                f'<text x="32" y="{y + 8}" fill="#f0f6fc" font-family="system-ui,sans-serif" font-size="15">{row["name"]}</text>',
                f'<line x1="{bpe_x:.1f}" y1="{y}" x2="{character_x:.1f}" y2="{y}" stroke="#484f58" stroke-width="3"/>',
                f'<circle cx="{character_x:.1f}" cy="{y}" r="7" fill="#f0883e"/>',
                f'<circle cx="{bpe_x:.1f}" cy="{y}" r="7" fill="#3fb950"/>',
                f'<text x="{character_x:.1f}" y="{y + 25}" fill="#f0883e" font-family="ui-monospace,monospace" font-size="11" text-anchor="middle">{row["character"]:.3f}</text>',
                f'<text x="{bpe_x:.1f}" y="{y - 13}" fill="#3fb950" font-family="ui-monospace,monospace" font-size="11" text-anchor="middle">{row["byte_bpe512"]:.3f}</text>',
            ]
        )
    lines.extend(
        [
            '<circle cx="650" cy="42" r="6" fill="#f0883e"/><text x="664" y="47" fill="#c9d1d9" font-family="system-ui,sans-serif" font-size="13">Character</text>',
            '<circle cx="752" cy="42" r="6" fill="#3fb950"/><text x="766" y="47" fill="#c9d1d9" font-family="system-ui,sans-serif" font-size="13">ByteBPE512</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_svg(load_results())
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            parser.error("checked-in chart is missing or stale; run make chart")
        print(f"current {OUTPUT_PATH.relative_to(ROOT)}")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
