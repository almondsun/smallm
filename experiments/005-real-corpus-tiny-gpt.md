# 005 Real Corpus Tiny GPT

## Goal

Train the tiny GPT config on a small real prose corpus, preserve the run artifacts, and compare the behavior against the earlier smoke/toy-corpus runs.

## Corpus

- Source: local ignored `data/raw/input.txt`.
- Description: a small public-domain-style prose excerpt adapted from the opening Alice-in-Wonderland scene.
- Characters: 4,838.
- Tokenizer: character-level.
- Vocab size: 53.
- Train/validation split: 90/10 from `configs/tiny_gpt.yaml`.

## Run

- Config: `configs/tiny_gpt.yaml`.
- Run path: `runs/tiny_gpt/2026-06-26_00-10-31`.
- Model size: 111,029 parameters.
- Device: CPU.
- Max steps: 500.
- Duration: 16.2 seconds.

## Metrics

First logged metric:

```json
{"elapsed_seconds": 0.3546127620029438, "learning_rate": 0.0003, "step": 10, "tokens_per_second": 28876.56930946832, "train_loss": 3.8724772930145264, "val_loss": null}
```

First validation metric:

```json
{"elapsed_seconds": 1.6713057440028933, "learning_rate": 0.0003, "step": 50, "tokens_per_second": 30634.729871371375, "train_loss": 3.070948839187622, "val_loss": 3.073628330230713}
```

Final metric:

```json
{"elapsed_seconds": 16.045488155003113, "learning_rate": 0.0003, "step": 500, "tokens_per_second": 31909.281603274518, "train_loss": 2.3240978717803955, "val_loss": 2.4505045413970947}
```

The run shows observable learning: train loss moved from 3.8725 at step 10 to 2.3241 at step 500, and validation loss moved from 3.0736 at step 50 to 2.4505 at step 500.

## Generated Sample

Trainer-written sample:

```text
Once towa it waladit o, shepond w won gte, lle ikeunke tar sf kpeecef be mary il tto h f sharerad opelin
```

Manual generation from the preserved run:

```text
Oncedny wan ttooor the at the ony tound wn ashe f t certly tind o.

Sr nd owalisathagarin tong hh t ve s
```

The generated text is still incoherent, but it has more prose-like spacing and letter patterns than the smoke run.

## Validation

```bash
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/list_runs.py --run-name tiny_gpt
python scripts/show_run.py --run latest --run-name tiny_gpt
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once"
python -m pytest
python -m compileall src scripts
```

- Data preparation passed with tokenizer vocab size 53.
- Training passed and wrote `runs/tiny_gpt/2026-06-26_00-10-31`.
- `list_runs.py --run-name tiny_gpt` found the preserved run.
- `show_run.py --run latest --run-name tiny_gpt` printed artifact paths, final metric, and sample.
- `generate.py --run latest --run-name tiny_gpt --prompt "Once"` generated text from the preserved checkpoint.
- `python -m pytest` passed: 15 tests.
- `python -m compileall src scripts` passed.

## Limitations

- The corpus is still very small.
- The tokenizer is still character-level.
- The sample is not coherent text yet.
- The run proves that the infrastructure can preserve and inspect a slightly more realistic training run; it does not yet prove useful language modeling quality.
