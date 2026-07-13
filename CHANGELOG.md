# Changelog

Notable user-visible and experiment-contract changes are recorded here. The project follows
[Semantic Versioning](https://semver.org/) for artifact and interface compatibility.

## [Unreleased]

### Added

- Added milestone 020's controlled BPE context-length and learning-rate study.

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

[Unreleased]: https://github.com/almondsun/smallm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/almondsun/smallm/releases/tag/v0.2.0
