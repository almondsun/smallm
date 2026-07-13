from __future__ import annotations

import re
from pathlib import Path

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def main() -> None:
    files = [
        Path("README.md"),
        *Path("docs").rglob("*.md"),
        *Path("experiments").glob("*.md"),
        *Path("notes").rglob("*.md"),
    ]
    missing: list[tuple[Path, str]] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1)
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative_target = target.split("#", 1)[0]
            path = (file.parent / relative_target).resolve()
            if not path.exists():
                missing.append((file, target))
    if missing:
        for file, target in missing:
            print(f"missing: {file}: {target}")
        raise SystemExit(1)
    print("markdown links resolve")


if __name__ == "__main__":
    main()
