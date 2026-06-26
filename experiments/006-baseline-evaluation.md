# 006 Baseline Evaluation

## Goal

Add simple character-level language-model baselines so tiny GPT validation loss can be interpreted against non-neural baselines.

## Corpus

- Corpus path: ignored local `data/raw/input.txt`.
- Characters: 4,838.
- Vocab size: 53.
- Split: same 90/10 train/validation split from `configs/tiny_gpt.yaml`.

## Baselines

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        3.9703        53.00 equal probability for every character
unigram           inf          inf train-set character frequencies
bigram         2.5562        12.89 add-1 smoothed character transitions
```

- Uniform is the expected `log(vocab_size)` baseline.
- Unigram is unsmoothed and is infinite on this split because at least one validation character is unseen in the training split.
- Bigram uses add-one smoothing and is the main finite non-neural baseline.

## Tiny GPT

- Run path: `runs/tiny_gpt/2026-06-26_00-30-47`.
- Final validation loss: 2.4505.
- Final validation perplexity: 11.59.
- Final train loss: 2.3241.
- Parameters: 111,029.

Tiny GPT beats the uniform baseline and the add-one smoothed bigram baseline on this run. The margin over bigram is small, but it is useful evidence that the neural model is learning more than simple character-transition statistics.

## Validation

```bash
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/evaluate_baselines.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/show_run.py --run latest --run-name tiny_gpt
python -m pytest
python -m compileall src scripts
```

- Data preparation passed with tokenizer vocab size 53.
- Baseline evaluation printed uniform, unigram, and bigram rows.
- Training passed and wrote `runs/tiny_gpt/2026-06-26_00-30-47`.
- `show_run.py --run latest --run-name tiny_gpt` inspected the latest preserved run.
- `python -m pytest` passed: 19 tests.
- `python -m compileall src scripts` passed.

## Limitations

- All baselines are character-level.
- The unigram baseline is unsmoothed, so unseen validation characters produce infinite loss.
- The corpus is still small, so results are directional rather than robust.
- The bigram baseline is simple add-one smoothing, not a tuned n-gram model.
