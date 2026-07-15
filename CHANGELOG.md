# Changelog

Notable user-visible and experiment-contract changes are recorded here. The project follows
[Semantic Versioning](https://semver.org/) for artifact and interface compatibility.

## [Unreleased]

## [1.0.2] - 2026-07-15

### Fixed

- Converted handbook mathematics to GitHub-compatible inline and display delimiters so all
  equations render on repository pages.
- Reworked the few formulas that conflicted with GitHub Markdown emphasis or inline-math parsing.

## [1.0.1] - 2026-07-14

### Added

- Rebuilt the theory handbook as a glossary-first, mathematically formal path from tensors and
  probability through the exact smaLLM implementation.
- Recorded the final evidence-preserving repository audit in the project-completion document.

### Changed

- Consolidated duplicate research references and stale operational result summaries into their
  canonical handbook, experiment, and completion records.
- Aligned contributor guidance and issue templates with the frozen-but-open lifecycle policy.

## [1.0.0] - 2026-07-14

### Added

- Completed the preregistered three-corpus, three-arm, three-seed capacity-controlled panel.
- Added the structured 27-cell result record, final evidence chart, and release-asset checksums.
- Added a permanent project-completion record and frozen-but-open lifecycle policy.

### Changed

- Declared the research and implementation scope complete and removed scheduled maintenance
  automation while retaining push and pull-request validation.
- Recorded the final outcome: ByteBPE512 beat near-parameter-matched char136 in eight of nine
  sealed pairs, with a negative mean contrast on all three corpora.

## [0.3.0] - 2026-07-14

### Added

- Added boundary-aware ByteBPE320/512, validation early stopping, multi-seed and cross-corpus
  robustness analysis, sealed chronological tests, and validated factorial-matrix aggregation.
- Added preregistered Hamlet, Art of War, and Lincoln external-distribution evidence through
  milestone 028.
- Added a clean-clone CPU demo, structured headline results, a deterministic result chart, and
  citation metadata.

### Changed

- Reframed the README as a concise research-engineering portfolio entry point.
- Made package version metadata authoritative from `smallm.__version__`.

## [0.2.0] - 2026-07-12

### Changed

- Hardened tokenizer fitting, provenance, evaluation, checkpoint loading, and run artifacts.
- Added a comprehensive theory handbook and professional quality gates.

### Security

- Restricted checkpoint loading to weights-only mode with bounded model and tokenizer schemas.
- Added atomic artifact writes, exact corpus verification, completed-run filtering, and dependency
  auditing.

## [0.1.0] - 2026-07-12

### Added

- Reproducible corpus preparation, character and educational BPE tokenizers, GPT training,
  baselines, controlled generation, run artifacts, and milestone reports through experiment 018.

[Unreleased]: https://github.com/almondsun/smallm/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/almondsun/smallm/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/almondsun/smallm/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/almondsun/smallm/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/almondsun/smallm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/almondsun/smallm/releases/tag/v0.2.0
