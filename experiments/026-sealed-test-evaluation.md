# 026 — Sealed Test Evaluation

## Goal

Harden the evaluation contract with a chronological train/validation/test split, freeze the
ByteBPE512 decision from milestone 025, and evaluate that decision once on terminal test segments
that were unavailable to tokenizer fitting, gradient updates, early stopping, and checkpoint
selection.

## Contract

`data.validation_split` is optional. When absent, legacy configs retain the existing two-way
train/remainder-validation behavior. When present, smaLLM creates three chronological regions:

1. training fits tokenizer merges and model parameters;
2. validation drives early stopping and selects `best_checkpoint.pt`;
3. test is sliced only to record its character count during training; it is not tokenized, scored,
   or used for selection until `scripts/evaluate_test.py` runs.

The four confirmatory configs use 80% train, 10% validation, and 10% test. Training summaries expose
only the test character count and the logical status `test_status: sealed_unread`; they contain no
test tokens, loss, or BPC. The status means unavailable to the modeling and selection workflow, not
that the containing corpus file was never opened. The evaluator requires a completed run with an explicit test split, verifies the copied
manifest and prepared-corpus checksum, loads the declared best checkpoint, verifies its step against
the summary, evaluates full coverage, hashes the checkpoint, writes `test_evaluation_best.json`, and
refuses to overwrite it.

The refusal is an operational guardrail, not tamper-proof enforcement: local artifacts can be
deleted. Scientific compliance still requires treating these four test results as consumed after
this milestone.

## Frozen Setup

The comparison was fixed before any test access: seed 1337, character versus ByteBPE512, 4 layers,
4 heads, width 128, dropout 0.1, AdamW `lr=1e-3`, zero weight decay, full validation every 250
steps, patience 3, and a 5,000-step ceiling. Character uses context 64 and batch 16; ByteBPE512 uses
context 37 and batch 27. No configuration was changed after observing validation or test results.

| corpus | prepared chars | train | validation | sealed test | checksum |
| --- | ---: | ---: | ---: | ---: | --- |
| Alice | 144,530 | 115,624 | 14,453 | 14,453 | `a4c81ef23eb9…` |
| Peter Pan | 144,489 | 115,591 | 14,449 | 14,449 | `16e4f26e7e52…` |

## Validation And Stopping

| corpus | tokenizer | actual steps | best step | best validation BPC |
| --- | --- | ---: | ---: | ---: |
| Alice | character | 5,000 | 4,250 | 2.119716 |
| Alice | ByteBPE512 | 2,250 | 1,500 | **2.076651** |
| Peter Pan | character | 5,000 | 4,750 | 2.195677 |
| Peter Pan | ByteBPE512 | 2,750 | 2,000 | **2.187737** |

The validation advantage is 0.043066 BPC on Alice and 0.007940 on Peter Pan. ByteBPE512 again
selects and stops much earlier, while character continues improving near the ceiling.

## One-Shot Test Results

All evaluations use the best-validation checkpoint and full non-overlapping coverage. One source
character is consumed as the autoregressive input before the first scored target, so target
characters equal test characters minus one.

| corpus | tokenizer | test tokens | target chars | test token loss | test BPC |
| --- | --- | ---: | ---: | ---: | ---: |
| Alice | character | 14,453 | 14,452 | 1.510502 | 2.179194 |
| Alice | ByteBPE512 | 8,471 | 14,452 | 2.504684 | **2.117790** |
| Peter Pan | character | 14,449 | 14,448 | 1.576362 | 2.274210 |
| Peter Pan | ByteBPE512 | 8,569 | 14,448 | 2.628048 | **2.248431** |

ByteBPE512 wins both sealed comparisons: 0.061404 BPC on Alice and 0.025780 on Peter Pan. These are
strikingly close to milestone 025's three-seed validation-matrix mean advantages of 0.061863 and
0.025203. The tokenizer decision therefore survives a genuinely untouched chronological segment on
both books.

Absolute test BPC is worse than validation BPC for every model: +0.059478 Alice character,
+0.041139 Alice ByteBPE512, +0.078533 Peter Pan character, and +0.060693 Peter Pan ByteBPE512. The
terminal text is harder than the middle validation region, which demonstrates why validation alone
was an optimistic description of forward generalization.

## Baselines

These are validation token losses under the new 80/10/10 training fit and are comparable only within
tokenizer:

| corpus | tokenizer | uniform | unigram | add-one bigram |
| --- | --- | ---: | ---: | ---: |
| Alice | character | 4.3307 | 3.2019 | 2.4569 |
| Alice | ByteBPE512 | 6.2383 | 4.3664 | 3.7471 |
| Peter Pan | character | 4.3820 | 3.0862 | 2.4118 |
| Peter Pan | ByteBPE512 | 6.2383 | 4.2748 | 3.7324 |

## Exact Commands

Each corpus was re-manifested without changing its prepared checksum:

```bash
python scripts/prepare_corpus.py ... --train-split 0.8 --validation-split 0.1
python scripts/prepare_data.py --config configs/gptiny_alice_char_sealed.yaml
python scripts/prepare_data.py --config configs/gptiny_alice_bytebpe512_sealed.yaml
python scripts/prepare_data.py --config configs/gptiny_peterpan_char_sealed.yaml
python scripts/prepare_data.py --config configs/gptiny_peterpan_bytebpe512_sealed.yaml
python scripts/evaluate_baselines.py --config configs/<sealed-config>.yaml
python scripts/train.py --config configs/gptiny_alice_char_sealed.yaml
python scripts/train.py --config configs/gptiny_alice_bytebpe512_sealed.yaml
python scripts/train.py --config configs/gptiny_peterpan_char_sealed.yaml
python scripts/train.py --config configs/gptiny_peterpan_bytebpe512_sealed.yaml
python scripts/evaluate_test.py --run runs/<sealed-run>/<run-id> --checkpoint-kind best
make check
make audit
```

Run paths are:

- `runs/gptiny_alice_char_sealed/2026-07-13_01-06-53`
- `runs/gptiny_alice_bytebpe512_sealed/2026-07-13_01-15-47`
- `runs/gptiny_peterpan_char_sealed/2026-07-13_01-19-42`
- `runs/gptiny_peterpan_bytebpe512_sealed/2026-07-13_01-28-21`

## Limitations And Next Step

- One seed is confirmatory for the fixed matrix-level decision, not a new seed-distribution study.
- Alice and Peter Pan are related English literary distributions.
- Parameter counts remain vocabulary-dependent.
- The terminal segment may differ in chapter structure and intrinsic difficulty.
- These test segments are consumed and cannot support further model or hyperparameter selection.
- The local one-shot guard is procedural rather than cryptographically enforced.

The next modeling experiment must define a new untouched evaluation distribution before training.
The strongest choice is a stylistically different public-domain corpus family—such as nonfiction,
speeches, or plays—with its own preregistered 80/10/10 split. The current Alice and Peter Pan test
results should remain final evidence for the frozen ByteBPE512 decision.
