# Contributing

## Development workflow

1. Create a focused branch from `main`.
2. Install dependencies with `uv sync --frozen --extra dev`.
3. Keep reusable behavior in `src/smallm/` and CLI adaptation in `scripts/`.
4. Add tests and update relevant contracts, notes, or experiment reports.
5. Run `make check` and the closest smoke or experiment command.
6. Open a pull request using the repository template.

Do not commit corpora, tokenizers, checkpoints, run directories, generated samples, or local
paths. Never weaken causal masking, held-out evaluation, provenance verification, or safe
checkpoint loading to make a test pass.

Experiment changes must record the hypothesis, controls, exact commands, observed outputs,
limitations, and corpus checksum. Negative results are welcome.

Use short imperative commit subjects and keep unrelated changes separate.
