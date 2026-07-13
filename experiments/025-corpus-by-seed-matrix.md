# 025 — Corpus-by-Seed Matrix

## Goal

Estimate whether ByteBPE512's advantage depends on corpus or training seed using a balanced,
preregistered 2-tokenizer × 2-corpus × 3-seed design. Every completed cell is reported. The
candidate-minus-reference contrast is fixed as ByteBPE512 BPC minus character BPC, so negative
values favor ByteBPE512.

## Design

The matrix crosses Alice and Peter Pan with character and ByteBPE512 tokenizers at seeds 1337,
2027, and 4242. Both corpora use a chronological 90/10 split and approximately 144.5k prepared
characters. All runs use 4 layers, 4 heads, width 128, dropout 0.1, AdamW `lr=1e-3`, zero weight
decay, full validation every 250 steps, patience 3, and a 5,000-step ceiling.

Character runs use a 64-token context and batch 16. ByteBPE512 is fitted on training text only and
uses a 37-token context and batch 27. The average character context is about 64.08 on Alice and
63.90 on Peter Pan. Vocabulary-dependent embeddings leave ByteBPE512 with more parameters, so this
is context-matched but not parameter-matched.

Milestone 025 reused five compatible completed runs: the three Alice ByteBPE512 seeds and both
Peter Pan seed-1337 runs. Seven missing runs were launched only after the matrix, seeds, contrast,
and analysis rules were fixed. Alice character seed 1337 was rerun under the explicit patience-3
contract instead of silently mixing it with the older no-stopping config; it reproduced best BPC
`2.075981` exactly.

## Artifact Validation

`scripts/summarize_matrix.py` requires exactly two declared tokenizers and two distinct corpus
checksums, all four cells, the same distinct seed set in every cell, one experiment fingerprint per
cell, complete schema-v2 summaries, matching tokenizer labels, bounded numeric fields, and unique
run directories. It computes only paired same-seed contrasts. A direct-execution import failure
found during validation moved the shared bounded run loader into
`smallm.evaluation.run_observation`; both aggregation CLIs now use that package boundary.

## Cell Results

| corpus | tokenizer | mean best BPC | population SD | range | mean actual steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Alice | character | 2.084406 | 0.013990 | 2.073117–2.104122 | 4,916.7 |
| Alice | ByteBPE512 | **2.022544** | 0.012358 | 2.008269–2.038414 | 2,666.7 |
| Peter Pan | character | 2.179927 | 0.007258 | 2.172051–2.189565 | 5,000.0 |
| Peter Pan | ByteBPE512 | **2.154724** | 0.003452 | 2.150949–2.159293 | 3,000.0 |

## Paired Tokenizer Effects

| corpus | seed | character BPC | ByteBPE512 BPC | Byte − character |
| --- | ---: | ---: | ---: | ---: |
| Alice | 1337 | 2.075981 | 2.008269 | -0.067712 |
| Alice | 2027 | 2.104122 | 2.038414 | -0.065708 |
| Alice | 4242 | 2.073117 | 2.020948 | -0.052169 |
| Peter Pan | 1337 | 2.172051 | 2.153930 | -0.018121 |
| Peter Pan | 2027 | 2.178164 | 2.150949 | -0.027215 |
| Peter Pan | 4242 | 2.189565 | 2.159293 | -0.030273 |

| paired contrast | mean | population SD | minimum | maximum |
| --- | ---: | ---: | ---: | ---: |
| Alice Byte − character | **-0.061863** | 0.006903 | -0.067712 | -0.052169 |
| Peter Pan Byte − character | **-0.025203** | 0.005161 | -0.030273 | -0.018121 |
| Peter Pan effect − Alice effect | **+0.036660** | 0.011380 | +0.021896 | +0.049591 |

ByteBPE512 wins all six paired comparisons. The average reduction is about 2.97% on Alice and
1.16% on Peter Pan. The direction is robust across this seed and corpus set, while the positive
interaction shows that the magnitude is corpus-dependent: Peter Pan reduces the advantage by
0.03666 BPC on average.

Population SD describes the complete preregistered three-seed set. It is not a confidence interval,
hypothesis test, or estimate over all possible books and seeds. The six paired observations are also
not independent because each seed appears across corpora.

## Training Dynamics

Character models peak late: Alice best steps are 5,000, 4,000, and 4,500; Peter Pan best steps are
4,750 for all three seeds. ByteBPE512 peaks between 1,750 and 2,500 and stops between 2,500 and
3,250. Patience therefore saves about 46% of Alice ByteBPE steps and 40% of Peter Pan ByteBPE
steps relative to their character controls without imposing an artificial shared stopping point.

This milestone deliberately evaluates held-out likelihood rather than selecting samples by eye.
Milestones 023–024 retain the controlled generation diagnostics; no generation metric is promoted
to a matrix-level outcome because distinct-n is neither semantic evaluation nor a preregistered
endpoint here.

## Exact Commands

The seven new configs are the three `configs/gptiny_char_5k_lr1e-3_earlystop*.yaml` files, the two
new Peter Pan character seed configs, and the two new Peter Pan ByteBPE512 seed configs. Each was
run with:

```bash
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/<config>.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/<config>.yaml
```

The exact 12-directory aggregation command is intentionally verbose so labels are explicit:

```bash
uv run --frozen --extra dev python scripts/summarize_matrix.py \
  --reference-tokenizer char --candidate-tokenizer byte_bpe \
  --cell alice char runs/gptiny_char_5k_lr1e-3_earlystop/2026-07-12_23-54-27 \
  --cell alice char runs/gptiny_char_5k_lr1e-3_earlystop_seed2027/2026-07-13_00-02-48 \
  --cell alice char runs/gptiny_char_5k_lr1e-3_earlystop_seed4242/2026-07-13_00-10-47 \
  --cell alice byte_bpe runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop/2026-07-12_22-12-08 \
  --cell alice byte_bpe runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed2027/2026-07-12_22-37-30 \
  --cell alice byte_bpe runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed4242/2026-07-12_22-43-16 \
  --cell peter_pan char runs/gptiny_peterpan_char_5k_lr1e-3_earlystop/2026-07-12_23-16-33 \
  --cell peter_pan char runs/gptiny_peterpan_char_5k_lr1e-3_earlystop_seed2027/2026-07-13_00-19-24 \
  --cell peter_pan char runs/gptiny_peterpan_char_5k_lr1e-3_earlystop_seed4242/2026-07-13_00-28-31 \
  --cell peter_pan byte_bpe runs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop/2026-07-12_23-25-37 \
  --cell peter_pan byte_bpe runs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed2027/2026-07-13_00-37-31 \
  --cell peter_pan byte_bpe runs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed4242/2026-07-13_00-43-01
uv run --frozen --extra dev make check
```

## Limitations And Next Step

- Two books and three seeds are a small descriptive factorial, not broad distribution coverage.
- Both corpora are English literary prose from Project Gutenberg.
- The chronological validation tail is repeatedly used for stopping and checkpoint selection.
- There is no sealed test segment, so validation-guided research decisions can accumulate bias.
- Parameter counts differ with tokenizer vocabulary.
- Corpus and tokenizer effects may change with scale, context, or optimizer settings.

The next professional step is evaluation-contract hardening rather than another training sweep:
add a deterministic train/validation/test split, keep test unavailable to early stopping, and
evaluate the already-fixed tokenizer decision once on sealed terminal segments. After that, expand
to a stylistically different corpus family rather than another nineteenth-century novel.
