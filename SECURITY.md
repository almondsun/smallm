# Security Policy

`smaLLM` is a frozen, unmaintained research artifact. No version currently receives guaranteed
security updates, and no response, triage, fix, or release timeline is promised.

Report checkpoint-deserialization, corpus-path, artifact-integrity, dependency, or data-exposure
issues through GitHub private vulnerability reporting. Include sanitized reproduction steps and
impact; never attach private corpora, checkpoints trained on private data, or local run artifacts.

Reports are still useful to downstream users and forks, but users must evaluate and mitigate risk
themselves. Do not use this educational project as a production or security boundary.

Corpora, manifests, configs, tokenizers, and checkpoints are input boundaries. Checkpoints are
loaded in weights-only mode, but users should still obtain artifacts from trusted sources and
verify provenance. Runtime artifacts may reproduce corpus fragments and must be handled as
potentially sensitive data.
