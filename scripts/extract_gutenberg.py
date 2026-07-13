from __future__ import annotations

import argparse
from pathlib import Path

from smallm.data import extract_gutenberg_body
from smallm.utils.io import atomic_write_text

_MAX_INPUT_BYTES = 20_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-characters", type=int)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if input_path.stat().st_size > _MAX_INPUT_BYTES:
        parser.error(f"input exceeds {_MAX_INPUT_BYTES} bytes")
    try:
        text = input_path.read_text(encoding="utf-8")
        body = extract_gutenberg_body(text, max_characters=args.max_characters)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    atomic_write_text(args.output, body)
    print(f"extracted {len(body)} characters to {args.output}")


if __name__ == "__main__":
    main()
