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

Version `1.0.0` is therefore a completion release, not the beginning of an ongoing maintenance
series.
