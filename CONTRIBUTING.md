# Contributing

`smaLLM` is a frozen research artifact with no active feature roadmap. Issues and pull requests are
welcome as public discussion and as useful context for downstream forks, but no response, review,
merge, fix, or release timeline is promised. New modeling features and experiments should normally
continue in a fork rather than extend this completed evidence record.

## Development workflow

1. Create a focused branch from `main`.
2. Install dependencies with `uv sync --frozen --extra dev`.
3. Keep reusable behavior in `src/smallm/` and CLI adaptation in `scripts/`.
4. Add tests and update relevant contracts or notes; do not rewrite historical experiment reports.
5. Run `make check` and the closest smoke or experiment command.
6. Open a pull request using the repository template.

Do not commit corpora, tokenizers, checkpoints, run directories, generated samples, or local
paths. Never weaken causal masking, held-out evaluation, provenance verification, or safe
checkpoint loading to make a test pass.

If a fork resumes experimentation, record the hypothesis, controls, exact commands, observed
outputs, limitations, and corpus checksum. Negative results are valid outcomes.

Use short imperative commit subjects and keep unrelated changes separate.
