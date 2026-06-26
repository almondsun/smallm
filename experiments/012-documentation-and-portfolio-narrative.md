# 012 Documentation and Portfolio Narrative

## Goal

Make `smaLLM` read like a serious small-scale language-model engineering project rather than an early toy README.

## Changes

- Rewrote `README.md` with positioning, capabilities, real quick start, current status, documentation links, and honest limitations.
- Expanded `docs/architecture.md` into a boundary-oriented system overview for data, model, evaluation, training, generation, and artifacts.
- Reworked `docs/training.md` into a concise workflow guide with command maps and artifact contracts.
- Added `docs/experiments.md` as a clean index and topic guide for experiments 001-011.
- Preserved the candid result from experiment 011: the larger corpus helped the bigram baseline more than the unchanged GPTiny model.

## Current Public Story

The repo now presents `smaLLM` as a reproducible tiny language-model lab built from scratch in PyTorch. It emphasizes what the project actually demonstrates:

- corpus preparation and provenance
- character tokenization
- causal self-attention and Transformer blocks
- training and validation loops
- non-neural baselines
- run artifacts and discovery
- controlled generation
- experiment reporting

It also states the current limits directly:

- greedy decoding still collapses
- generated text is still weak
- the larger-corpus run trailed the bigram baseline
- the next technical question is training budget, optimization, capacity, or tokenization

## Validation

Documentation was checked by reading the updated files and verifying links point to existing docs and experiment reports.

No code paths were changed in this milestone.

## Remaining Limitations

- The docs now describe the project accurately, but they do not include diagrams.
- There is still no generated API reference.
- The next technical milestone should return to experiments, likely a training budget and optimization study.
