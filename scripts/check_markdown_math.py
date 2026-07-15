"""Check theory-note math for GitHub-compatible delimiters and notation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "notes"
FORBIDDEN_COMMANDS = ("left", "operatorname", "right", "text", "texttt")
COMMAND_PATTERN = re.compile(
    r"\\(?:" + "|".join(re.escape(command) for command in FORBIDDEN_COMMANDS) + r")\b"
)


def math_expressions(markdown: str) -> list[tuple[int, str]]:
    """Return line-numbered math expressions outside fenced code blocks."""
    expressions: list[tuple[int, str]] = []
    in_fence = False
    delimiter: str | None = None
    start_line = 0
    chunks: list[str] = []

    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        index = 0
        while index < len(line):
            if delimiter is None:
                marker = line.find("$", index)
                if marker < 0:
                    break
                if marker > 0 and line[marker - 1] == "\\":
                    index = marker + 1
                    continue
                delimiter = "$$" if line.startswith("$$", marker) else "$"
                start_line = line_number
                chunks = []
                index = marker + len(delimiter)
            else:
                marker = line.find(delimiter, index)
                if marker < 0:
                    chunks.append(line[index:])
                    break
                if marker > 0 and line[marker - 1] == "\\":
                    chunks.append(line[index : marker + 1])
                    index = marker + 1
                    continue
                closing_length = len(delimiter)
                chunks.append(line[index:marker])
                expressions.append((start_line, "".join(chunks)))
                delimiter = None
                chunks = []
                index = marker + closing_length

    if delimiter is not None:
        raise ValueError(f"unclosed {delimiter} delimiter starting on line {start_line}")
    return expressions


def check_braces(expression: str) -> str | None:
    """Return an error for unmatched grouping braces, ignoring escaped set braces."""
    depth = 0
    for index, character in enumerate(expression):
        if character not in "{}":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and expression[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        depth += 1 if character == "{" else -1
        if depth < 0:
            return "unmatched closing brace"
    return "unmatched opening brace" if depth else None


def main() -> int:
    failures: list[str] = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        try:
            expressions = math_expressions(path.read_text(encoding="utf-8"))
        except ValueError as error:
            failures.append(f"{path.relative_to(ROOT)}: {error}")
            continue
        for line_number, expression in expressions:
            if match := COMMAND_PATTERN.search(expression):
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: unsupported GitHub math command "
                    f"{match.group(0)}"
                )
            if brace_error := check_braces(expression):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {brace_error}")

    if failures:
        print("\n".join(failures))
        return 1
    print("note math passes GitHub-compatibility checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
