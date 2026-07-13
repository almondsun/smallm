# Security Policy

Security fixes apply to the latest code on `main`.

Report checkpoint-deserialization, corpus-path, artifact-integrity, dependency, or data-exposure
issues through GitHub private vulnerability reporting. Include sanitized reproduction steps and
impact; never attach private corpora, checkpoints trained on private data, or local run artifacts.

Corpora, manifests, configs, tokenizers, and checkpoints are input boundaries. Checkpoints are
loaded in weights-only mode, but users should still obtain artifacts from trusted sources and
verify provenance. Runtime artifacts may reproduce corpus fragments and must be handled as
potentially sensitive data.
