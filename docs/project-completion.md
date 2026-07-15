# Project Completion

`smaLLM` reached permanent feature completion at version `1.0.0`. The repository is preserved as
an inspectable research-engineering portfolio artifact: it is complete, public, and open to
discussion, but it is not an actively maintained product.

## What Is Complete

- Reproducible corpus preparation with normalization, checksums, manifests, and chronological
  train/validation/test splits.
- Character, educational character-BPE, and lossless boundary-aware UTF-8 ByteBPE tokenization.
- A decoder-only Transformer, config-driven training, best-checkpoint selection, controlled
  generation, baselines, and run provenance.
- Correct character-normalized BPC evaluation, sealed-test handling, balanced multi-corpus and
  multi-seed analysis, and explicit capacity controls.
- A chronological record of implementation and experiments, including superseded contracts,
  negative results, preregistrations, and final conclusions.
- Repository-native formatting, lint, strict type checking, branch-covered tests, package builds,
  dependency auditing, and a clean-clone CPU demonstration.

## Final Research Question

Earlier panels found that ByteBPE512 beat a width-128 character model on sealed test segments, but
the larger tokenizer vocabulary also increased the embedding and output parameter count. The
final preregistered study therefore compared ByteBPE512 with both the historical width-128
character arm and a near-parameter-matched width-136 character arm across three new genres and
three fixed seeds. The protocol was committed before source access and the complete result is
reported in [experiment 030](../experiments/030-final-capacity-panel-and-project-completion.md).

## Deliberate Limits

This project does not claim production language-model quality, broad population inference,
state-of-the-art tokenization, distributed training, benchmark leadership, or safety for
deployment. Models are intentionally small, each corpus is one public-domain work, seeds and
terminal regions are fixed, and runtime corpora and checkpoints are not distributed.

These limits are part of the artifact's contract rather than unfinished roadmap items.

## Frozen-But-Open Policy

- No additional features, experiments, dependency refreshes, or compatibility work are planned.
- CI remains available on pushes and pull requests as reproducibility evidence.
- Issues and pull requests may remain open for discussion, but response, review, merge, release,
  and security-fix timelines are not promised.
- Forks may continue the work under the MIT license. Such work is independent unless explicitly
  incorporated into a separately identified fork or release.
- Runtime artifacts remain local and ignored. The durable record is source, tests, configs,
  structured aggregate results, experiment reports, release metadata, and checksums.

Version `1.0.0` is therefore the feature-completion release, not the beginning of an ongoing
maintenance series. Version `1.0.1` polishes the archived repository and learning material;
`1.0.2` only corrects GitHub rendering of the handbook mathematics.

## Final Archive Audit

The `1.0.1` audit reviewed every tracked top-level subsystem and preserved the material required to
inspect or reproduce the project: source, tests, thin CLI scripts, all experiment configurations,
the complete 001–030 report sequence, structured results, CI, and release metadata.

Cleanup was deliberately narrower than deletion by age:

- the theory handbook now forms one glossary-first sequence numbered 00–12;
- unique study-source revisions were moved into the handbook reference chapter;
- redundant pointer notes, a duplicated paper PDF, stale operational result narratives, and the
  feature-proposal issue surface were removed;
- current guidance now points to milestone 030 and the 182-test contract;
- ignored corpora, checkpoints, runs, cloned references, and other local artifacts were not
  inspected as durable source and were not modified.

The audit found no private source symbol that could be removed without weakening a public,
historical, or compatibility contract. Version metadata is the only runtime source change.

Final validation passed on 2026-07-14:

- `uv run make check`: Ruff format/lint, strict mypy over 47 source files, 182 tests, 90.49%
  branch coverage, compileall, and Markdown links;
- `uv run make audit`: no known dependency vulnerabilities; the unpublished local
  `smallm==1.0.1` package is correctly skipped;
- `uv run make demo`: corpus preparation, tokenizer fitting, baselines, five-step CPU training,
  run inspection, and deterministic generation;
- deterministic chart verification, wheel/source build, isolated wheel installation reporting
  `1.0.1`, distribution-content inspection, and all release checksums.
