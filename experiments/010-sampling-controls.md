# 010 Sampling Controls

## Goal

Make text generation reproducible and comparable across runs by adding explicit sampling controls for CLI generation and training-time sample artifacts.

## Implemented Controls

Generation now supports:

- `max_new_tokens`
- `temperature`
- `top_k`
- `seed`
- `greedy`

The default generation behavior remains sampling with `temperature=1.0`, no `top_k`, no seed, and `greedy=False`.

Training sample generation now reads these config fields from `train`:

- `sample_max_new_tokens`
- `sample_temperature`
- `sample_top_k`
- `sample_seed`
- `sample_greedy`

Each run records the settings in `summary.json` under `generation`.

## Latest Run

- Run path: `runs/tiny_gpt/2026-06-26_11-52-05`.
- Final train loss: 2.3241.
- Final validation loss: 2.4505.
- Training sample settings:

```json
{
  "greedy": false,
  "max_new_tokens": 100,
  "prompt": "Once",
  "seed": 1337,
  "temperature": 1.0,
  "top_k": null
}
```

## Example Greedy Output

Command:

```bash
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --greedy --max-new-tokens 100
```

Output:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the
```

## Example Seeded Top-K Output

Command:

```bash
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
```

Output:

```text
Oncer on st.
" ind a wis he wheline then ang t arat wo the ithe shering ad s wing the te atre wato ht at
```

## Validation

```bash
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "Alice-style public-domain prose excerpt"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --greedy --max-new-tokens 100
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
python -m pytest
python -m compileall src scripts
```

- Corpus preparation with manifest passed.
- Tokenizer preparation passed with vocab size 53.
- Training passed and wrote controlled `sample.txt` settings to `summary.json`.
- Greedy generation from the latest run passed.
- Seeded top-k generation from the latest run passed.
- `python -m pytest` passed: 31 tests.
- `python -m compileall src scripts` passed.

## Remaining Limitations

- Sampling controls affect generation only; they do not change model architecture, tokenizer, or training dynamics.
- Greedy decoding can collapse into repetitive text for this tiny model.
- Seeded generation is reproducible for the same runtime/device path, but exact samples may still vary across different PyTorch versions or device backends.
- Training still writes a single sample per run rather than a comparison grid across multiple sampling settings.
