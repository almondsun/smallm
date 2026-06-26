# 016 Tokenization Study

## Goal

Test whether a small educational BPE tokenizer improves GPTiny generation coherence or modeling efficiency compared with the current character tokenizer, while keeping the corpus, architecture, optimizer, training budget, prompt, and generation diagnostics controlled.

## Setup

- Corpus: Project Gutenberg ebook #11, *Alice's Adventures in Wonderland* by Lewis Carroll.
- Extraction: body text between the Gutenberg START/END markers, with boilerplate removed before corpus preparation.
- Prepared corpus: 144,530 characters.
- Prepared SHA-256: `a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`.
- Split: 90/10 chronological split.
- Train characters: 130,077.
- Validation characters: 14,453.
- Character control: `configs/gptiny_5k_lr1e-3_deep.yaml`.
- BPE comparison: `configs/gptiny_bpe128_5k_lr1e-3_deep.yaml`.
- Optional BPE256 config and tokenizer artifact were created, but BPE256 training was skipped after BPE128 already took about 8 minutes and showed a clear early-overfit signal.

Character-level baselines on the same prepared corpus:

| baseline | validation loss | perplexity | notes |
| --- | ---: | ---: | --- |
| uniform | 4.3175 | 75.00 | equal probability for every character |
| unigram | 3.1699 | 23.81 | train-set character frequencies |
| bigram | 2.4340 | 11.40 | add-1 smoothed character transitions |

## Implementation Changes

- Added `SimpleBPETokenizer` in `src/smallm/data/bpe_tokenizer.py`.
- Added tokenizer config fields: `data.tokenizer_type`, `data.bpe_vocab_size`, and `data.bpe_min_frequency`.
- Added tokenizer artifact loading that supports current char artifacts, legacy char artifacts, and BPE artifacts.
- Updated `prepare_data.py` and training to train/save the selected tokenizer type.
- Updated checkpoints to embed tokenizer state through a tokenizer-neutral `to_state()` path.
- Updated training to split the corpus at the shared character boundary before
  encoding each split, so tokenizers do not silently change the validation text.
- Added tokenizer metadata and bits-per-character estimates to new training summaries.
- Added tests for BPE config validation, BPE round-trip behavior, save/load, deterministic training, unknown-character handling, legacy char artifact loading, and a tiny BPE training run.

The BPE implementation is intentionally small and inspectable. It starts from individual characters plus `<unk>`, repeatedly merges the most frequent adjacent symbol pair, stops at the requested vocabulary size or minimum frequency, and serializes the vocabulary and merge list as JSON.

## Tokenizer Comparison

| tokenizer | config | vocab size | train tokens | validation tokens | validation chars/token |
| --- | --- | ---: | ---: | ---: | ---: |
| char | `gptiny_5k_lr1e-3_deep` | 75 | 130,077 | 14,453 | 1.00 |
| BPE128 | `gptiny_bpe128_5k_lr1e-3_deep` | 128 | 86,380 | 9,522 | 1.52 |

BPE128 reduced the validation sequence length by about 34.1%. The training path
splits the corpus at the shared character boundary before encoding each split,
so both tokenizers evaluate the same 14,453 validation characters.

## Training Results

| tokenizer | run | parameters | final train loss | final val loss | best val loss | best step | final ppl | best ppl | final bits/char | best bits/char | duration | tokens/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char | `runs/gptiny_5k_lr1e-3_deep/2026-06-26_15-01-40` | 820,811 | 1.2805 | 1.6447 | 1.4950 | 2500 | 5.18 | 4.46 | 2.3728 | 2.1569 | 442.20s | 11,590 |
| BPE128 | `runs/gptiny_bpe128_5k_lr1e-3_deep/2026-06-26_17-34-59` | 834,432 | 1.4121 | 2.8109 | 2.5727 | 1250 | 16.62 | 13.10 | 2.6717 | 2.4453 | 582.36s | 8,798 |

The BPE token-level losses are not comparable to the character losses because the prediction units and vocabulary differ. Bits per character is a fairer rough comparison here, and on that metric BPE128 underperformed the character control. BPE128 also reached its best validation loss early, at step 1250, while training loss continued improving through step 5000.

## Generation Diagnostics

Greedy decoding with prompt `Once`, `max_new_tokens=200`:

| tokenizer | repetition rate | distinct-1 | distinct-2 | longest repeated token run | note |
| --- | ---: | ---: | ---: | ---: | --- |
| char | 0.0099 | 0.1324 | 0.4039 | 2 | repetitive but partially sentence-like; loops around common phrases |
| BPE128 | 0.0131 | 0.1176 | 0.4918 | 2 | more word-like surface text, but phrase repetition remains obvious |

Seeded top-k decoding with prompt `Once`, `temperature=0.8`, `top_k=10`, `seed=1337`, `max_new_tokens=200`:

| tokenizer | repetition rate | distinct-1 | distinct-2 | longest repeated token run | note |
| --- | ---: | ---: | ---: | ---: | --- |
| char | 0.0345 | 0.1863 | 0.5862 | 1 | best 015 sampled texture; still semantically loose |
| BPE128 | 0.0171 | 0.1224 | 0.5222 | 1 | smoother word chunks in places, but lower diversity and incoherent continuation |

Greedy excerpts:

```text
char:
Once of the same to the whiting the same to say things and the reason of the sea.
```

```text
BPE128:
Once?”

“It didn’t know what are _you_ myself about it!” said the Mouse.
```

Seeded top-k excerpts:

```text
char:
Once, by the Dormouse.

“Yes!” said Alice.
```

```text
BPE128:
Once? Why, I don’t
have done when I heard used to dry pepper to it!
```

## Metric Caveats

Token-level validation loss and perplexity are not apples-to-apples across tokenizers. A BPE token predicts a different unit than a character token, with a different vocabulary size and sequence length. Bits per character is more comparable, but this implementation estimates it from validation loss and validation token/character counts rather than from a separate character-normalized evaluator.

The generation diagnostics are text-level and therefore comparable after decoding, but they remain simple proxies. In particular, `longest_repeated_token_run` catches repeated identical words but does not capture phrase loops such as repeated `said the Mock Turtle` patterns.

## Answers

Did BPE reduce sequence length?

Yes. BPE128 reduced the validation sequence from 14,453 character tokens to 9,522 BPE tokens, about 1.52 validation characters per token.

Did BPE improve generated text coherence?

Mixed to negative. Greedy BPE128 produced more word-like and quotation-shaped text than the character control, but it still repeated phrases and did not produce coherent prose. Seeded top-k BPE128 was not clearly better than the character control.

Did BPE improve text-level generation diagnostics?

Only partially. Greedy distinct-2 improved from 0.4039 to 0.4918, but greedy repetition rate increased and distinct-1 fell. Seeded top-k diagnostics got worse on distinct-1 and distinct-2, though repetition rate decreased.

Did BPE introduce new failure modes?

It introduced an early validation peak followed by worse final validation while train loss kept falling. It also produced some smoother word-level chunks but retained phrase-level incoherence.

Does tokenization look like a useful direction?

Yes, but simple BPE128 with the unchanged deep GPTiny setup did not beat the character control. The result suggests tokenization is worth studying, but the BPE setup likely needs its own tuning, such as a smaller learning rate, different block size in token units, stronger regularization, or a second vocab-size point before making stronger claims.

## Validation

Commands run before the experiment:

```bash
python -m pytest
python -m compileall src scripts
```

Observed result:

- `python -m pytest`: 54 passed.
- `python -m compileall src scripts`: passed.
- Markdown link/path check from `docs/codex/build-and-test.md`: passed.

Experiment commands run:

```bash
curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/raw/input.txt
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --manifest data/processed/corpus_manifest.json \
  --source-name "Larger public-domain prose corpus" \
  --source-note "Project Gutenberg ebook #11, Alice's Adventures in Wonderland by Lewis Carroll, body text between START/END markers; boilerplate removed after fetch from https://www.gutenberg.org/cache/epub/11/pg11.txt"
python scripts/prepare_data.py --config configs/gptiny_5k_lr1e-3_deep.yaml
python scripts/evaluate_baselines.py --config configs/gptiny_5k_lr1e-3_deep.yaml
python scripts/prepare_data.py --config configs/gptiny_bpe128_5k_lr1e-3_deep.yaml
python scripts/prepare_data.py --config configs/gptiny_bpe256_5k_lr1e-3_deep.yaml
python scripts/train.py --config configs/gptiny_bpe128_5k_lr1e-3_deep.yaml
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3_deep
python scripts/show_run.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
```

The character control training run was reused from experiment 015 because the run artifact was present and valid:

```text
runs/gptiny_5k_lr1e-3_deep/2026-06-26_15-01-40
```

## Limitations

- The BPE tokenizer is intentionally educational, not production-grade.
- Only BPE128 was trained; BPE256 was prepared but not run.
- The corpus is one public-domain book with a chronological split.
- The architecture and optimizer were copied from the best character setup, which may not be optimal for BPE.
- The block size stayed fixed in token units, so BPE sees a longer character span per context window than the character model.
- Bits per character is estimated from token validation loss and token/character counts.
- Generated text evaluation still relies on simple text diagnostics and qualitative inspection.
