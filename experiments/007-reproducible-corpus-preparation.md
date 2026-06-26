# 007 Reproducible Corpus Preparation

## Goal

Make the small real-corpus experiments reproducible and auditable with a clear corpus preparation step, source metadata, and generated corpus statistics.

## Command

```bash
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --source-name "Alice-style public-domain prose excerpt"
```

## Corpus Stats

- Source name: Alice-style public-domain prose excerpt.
- Output path: `data/processed/corpus.txt`.
- Generated stats path: `data/processed/corpus_stats.json`.
- Total characters: 4,838.
- Total lines: 58.
- Non-empty lines: 48.
- Unique characters: 53.
- Train split: 0.9.
- Train characters: 4,354.
- Validation characters: 484.

Top character frequencies:

```text
" " 879
"e" 503
"t" 377
"o" 300
"a" 281
```

## Baselines

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        3.9703        53.00 equal probability for every character
unigram           inf          inf train-set character frequencies
bigram         2.5562        12.89 add-1 smoothed character transitions
```

## Tiny GPT Run

- Latest run path: `runs/tiny_gpt/2026-06-26_00-51-36`.
- Final train loss: 2.3241.
- Final validation loss: 2.4505.
- Parameters: 111,029.
- Duration: 13.0 seconds on CPU.

The tiny GPT run still beats the add-one smoothed bigram baseline on validation loss after switching training and baseline evaluation to the prepared corpus path.

## Validation

```bash
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --source-name "Alice-style public-domain prose excerpt"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/evaluate_baselines.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/show_run.py --run latest --run-name tiny_gpt
python -m pytest
python -m compileall src scripts
```

- Corpus preparation passed and wrote `data/processed/corpus.txt` plus `data/processed/corpus_stats.json`.
- Tokenizer preparation passed with vocab size 53.
- Baseline evaluation passed.
- Training passed and wrote `runs/tiny_gpt/2026-06-26_00-51-36`.
- Run inspection passed.
- `python -m pytest` passed: 23 tests.
- `python -m compileall src scripts` passed.

## Remaining Limitations

- Raw text is still local and ignored.
- The prepared corpus is generated and ignored, so users must run `prepare_corpus.py` before training or evaluation.
- Corpus metadata is local JSON, not a dataset manifest with checksums.
- Tokenization is still character-level.
