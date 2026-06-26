# 009 Run Dataset Provenance

## Goal

Make every training run self-contained from a data-provenance perspective by copying the generated corpus manifest into the run directory and recording key dataset identity fields in `summary.json`.

## Latest Run

- Run path: `runs/tiny_gpt/2026-06-26_01-36-39`.
- Copied manifest path: `runs/tiny_gpt/2026-06-26_01-36-39/dataset_manifest.json`.
- Final train loss: 2.3241.
- Final validation loss: 2.4505.

The run directory now contains:

```text
checkpoint.pt
config.yaml
dataset_manifest.json
metrics.jsonl
sample.txt
summary.json
```

## Summary Dataset Fields

`summary.json` now contains a `dataset` object with:

```json
{
  "manifest_path": "runs/tiny_gpt/2026-06-26_01-36-39/dataset_manifest.json",
  "prepared_characters": 4838,
  "prepared_sha256": "1f7cc1920f8b1bed55b8bd84aa6df1cc0b4b04e6a972d048adf2f9c0b6219366",
  "raw_characters": 4838,
  "raw_sha256": "1f7cc1920f8b1bed55b8bd84aa6df1cc0b4b04e6a972d048adf2f9c0b6219366",
  "source_name": "Alice-style public-domain prose excerpt",
  "source_note": null,
  "train_characters": 4354,
  "train_split": 0.9,
  "unique_characters": 53,
  "validation_characters": 484
}
```

Normalization rules are also preserved in the dataset object.

## `show_run` Dataset Output

```text
dataset:
  source: Alice-style public-domain prose excerpt
  prepared_sha256: 1f7cc1920f8b
  prepared_characters: 4838
  unique_characters: 53
  split_characters: train=4354 val=484
```

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

- Corpus preparation with manifest passed.
- Tokenizer preparation passed with vocab size 53.
- Baseline evaluation passed.
- Training passed and copied `dataset_manifest.json` into the run directory.
- Run inspection printed the dataset section.
- `python -m pytest` passed: 27 tests.
- `python -m compileall src scripts` passed.

## Remaining Limitations

- The run summary stores selected manifest fields, not the entire manifest object.
- Raw and prepared corpus files remain local and ignored.
- There is still no tracked public dataset snapshot or remote source fetch step.
