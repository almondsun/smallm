# 021 — Boundary-Aware Byte BPE

## Goal

Test whether the corrected BPE128 deficit comes from its character-only alphabet and unrestricted
cross-boundary merges. The intervention is a lossless UTF-8 byte BPE with explicit whitespace
boundaries, evaluated against the milestone-019 character and BPE128 controls.

## Method

The tokenizer starts with all 256 bytes, so every UTF-8 input has fallback coverage and no `<unk>`.
It splits text into maximal whitespace/non-whitespace segments and learns/applies merges only within
segments. Fitting uses the 130,077-character training split only. Encoding records how many Unicode
scalars finish in each token, allowing exact BPC even when a scalar spans several bytes.

A byte vocabulary cannot contain 128 entries: the lossless base alphabet already has 256. The tested
caps are therefore 320 and 512. On the training split they compress to 93,304 and 75,103 tokens,
or 1.3941 and 1.7320 characters/token. Contexts 46 and 37 cover about 64 characters; batches 22 and
27 yield 1,012 and 999 token positions/update versus 1,024 for the character control.

Both runs use the same chronological 90/10 split, corpus checksum `a4c81ef23eb9…`, 4-layer,
4-head, 128-wide GPTiny, dropout 0.1, AdamW at `1e-3`, seed 1337, 5,000 steps, and full
non-overlapping validation every 250 steps. Vocabulary-dependent embeddings make parameter counts
881,472 and 929,664; this modest capacity change is a limitation of fixed-width token models.

## Results

| tokenizer | train / val tokens | best step | best loss | best BPC | final loss | final BPC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| character control (019) | 130,077 / 14,453 | 5,000 | 1.438960 | 2.075981 | 1.438960 | 2.075981 |
| BPE128 control (019) | — / 9,522 | 2,500 | 2.202047 | 2.097552 | — | — |
| ByteBPE320 | 93,304 / 10,324 | 3,250 | 1.968506 | **2.028567** | 2.031014 | 2.092982 |
| ByteBPE512 | 75,103 / 8,471 | 1,750 | 2.375155 | **2.008269** | 2.655081 | 2.244955 |

Both boundary-aware tokenizers beat both corrected controls. ByteBPE320 improves on character BPC by
2.28%; ByteBPE512 improves it by 3.26% and is the strongest corrected result in the repository.
The larger vocabulary also overfits much earlier and more severely: ByteBPE512 loses 0.2367 BPC
between its best and final checkpoints while training loss continues downward.

Token-level baselines are tokenizer-specific and not comparable across rows. For completeness,
ByteBPE320 uniform/unigram/bigram losses are 5.7683/3.9374/3.0450; ByteBPE512 losses are
6.2383/4.3271/3.6564. Both neural runs beat their own bigram baseline.

## Controlled Generation

Prompt `Once`, 100 new tokens. Seeded decoding uses temperature 0.8, top-k 10, seed 1337.

| tokenizer/checkpoint | greedy distinct-2 | seeded distinct-2 | qualitative reading |
| --- | ---: | ---: | --- |
| ByteBPE320 best | 0.5704 | 0.6338 | Locally fluent but repeats “what I … know”. |
| ByteBPE320 final | 0.6507 | 0.6241 | More varied surface form; syntax remains unstable. |
| ByteBPE512 best | 0.5217 | 0.5417 | Word-like dialogue with phrase reuse. |
| ByteBPE512 final | 0.6045 | 0.6901 | Highest seeded diversity, not highest validation quality. |

The best validation checkpoint does not maximize distinct-2, reinforcing that diversity is not a
surrogate for held-out likelihood or coherence. None of the samples sustains long-range meaning.

## Exact Commands

```bash
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_bytebpe320_5k_lr1e-3_ctx46.yaml
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37.yaml
uv run --frozen --extra dev python scripts/evaluate_baselines.py --config configs/gptiny_bytebpe320_5k_lr1e-3_ctx46.yaml
uv run --frozen --extra dev python scripts/evaluate_baselines.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe320_5k_lr1e-3_ctx46.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37.yaml
uv run --frozen --extra dev python scripts/show_run.py --run latest --run-name gptiny_bytebpe320_5k_lr1e-3_ctx46
uv run --frozen --extra dev python scripts/show_run.py --run latest --run-name gptiny_bytebpe512_5k_lr1e-3_ctx37
uv run --frozen --extra dev make check
```

Generation used `scripts/generate.py` for each run and `best`/`final` checkpoint kind, once with
`--greedy --diagnostics` and once with
`--temperature 0.8 --top-k 10 --seed 1337 --diagnostics`.

Run paths:

- `runs/gptiny_bytebpe320_5k_lr1e-3_ctx46/2026-07-12_21-35-03`
- `runs/gptiny_bytebpe512_5k_lr1e-3_ctx37/2026-07-12_21-43-55`

## Limitations And Next Question

- This is one corpus split and one seed; it does not establish a universal tokenizer ranking.
- Vocabulary size changes embedding/head parameter count and optimization geometry.
- Whitespace segmentation is intentionally simple and language-dependent.
- Invalid generated byte sequences decode with replacement characters, although encoded source text
  round-trips exactly.
- Distinct-n diagnostics are surface statistics, not human evaluation.

The next controlled experiment should keep ByteBPE512 fixed and address its early overfit with
validation-based early stopping as an orchestration feature, then compare modest dropout or weight
decay changes without extending the sweep or scaling the model.
