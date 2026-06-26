# 008 Dataset Manifest and Checksums

## Goal

Make corpus preparation auditable by recording file hashes, normalization rules, source metadata, and the exact prepared corpus identity.

## Manifest Command

```bash
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --manifest data/processed/corpus_manifest.json \
  --source-name "Alice-style public-domain prose excerpt"
```

## Key Manifest Fields

- Source name: Alice-style public-domain prose excerpt.
- Raw path: `data/raw/input.txt`.
- Prepared path: `data/processed/corpus.txt`.
- Stats path: `data/processed/corpus_stats.json`.
- Raw characters: 4,838.
- Prepared characters: 4,838.
- Unique characters: 53.
- Train split: 0.9.
- Train characters: 4,354.
- Validation characters: 484.

Normalization rules:

```text
normalize CRLF/CR to LF
strip trailing whitespace
collapse repeated blank lines
ensure final newline
```

## Checksums

- Raw SHA-256: `1f7cc1920f8b1bed55b8bd84aa6df1cc0b4b04e6a972d048adf2f9c0b6219366`
- Prepared SHA-256: `1f7cc1920f8b1bed55b8bd84aa6df1cc0b4b04e6a972d048adf2f9c0b6219366`

The hashes match for this corpus because the current raw file already conforms to the preparation rules.

## Corpus Stats

- Total lines: 58.
- Non-empty lines: 48.
- Top characters:
  - space: 879
  - `e`: 503
  - `t`: 377
  - `o`: 300
  - `a`: 281

## Baselines

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        3.9703        53.00 equal probability for every character
unigram           inf          inf train-set character frequencies
bigram         2.5562        12.89 add-1 smoothed character transitions
```

## Tiny GPT Run

- Latest run path: `runs/tiny_gpt/2026-06-26_01-10-32`.
- Final train loss: 2.3241.
- Final validation loss: 2.4505.
- Parameters: 111,029.
- Duration: 14.5 seconds on CPU.

## Validation

```bash
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "Alice-style public-domain prose excerpt"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/evaluate_baselines.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/show_run.py --run latest --run-name tiny_gpt
python -m pytest
python -m compileall src scripts
```

- Manifest generation passed and wrote `data/processed/corpus_manifest.json`.
- Tokenizer preparation passed with vocab size 53.
- Baseline evaluation passed.
- Training passed and wrote `runs/tiny_gpt/2026-06-26_01-10-32`.
- Run inspection passed.
- `python -m pytest` passed: 25 tests.
- `python -m compileall src scripts` passed.

## Remaining Limitations

- The manifest is generated and ignored, so users must regenerate it locally.
- The raw source text is still local and ignored.
- There is no tracked corpus snapshot or remote source fetch step.
- The manifest records file identity, but training run summaries do not yet copy manifest fields into run artifacts.
