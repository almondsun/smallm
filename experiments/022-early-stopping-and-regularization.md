# 022 — Early Stopping and Regularization

## Goal

Turn milestone 021's early ByteBPE512 optimum into explicit, inspectable training behavior, then test
whether modest AdamW weight decay delays overfitting. The intervention stays narrow: the same seed,
tokenizer, model, data order, validation contract, and 5,000-step ceiling.

## Implementation

`TrainConfig` now accepts `early_stopping_patience` and `early_stopping_min_delta`. Patience counts
full validation events without a meaningful improvement. The lowest numerical validation checkpoint
is still saved independently. Run summaries add `actual_steps`, `stopped_early`, `stop_reason`, and
the terminal early-stopping state; the final checkpoint represents the actual stop step.

Both experiment configs use patience 3, minimum delta 0, full validation every 250 steps, and the
milestone-021 ByteBPE512 setup. One keeps weight decay 0; the other changes only weight decay to
`0.01`.

## Results

| setting | actual / ceiling steps | best step | best loss | best BPC | stopped-final BPC | duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 021 control, no stopping | 5,000 / 5,000 | 1,750 | 2.375155 | 2.008269 | 2.244955 | 422.7s |
| early stopping | 2,500 / 5,000 | 1,750 | 2.375155 | 2.008269 | 2.055447 | 209.3s |
| early stopping + WD 0.01 | 2,500 / 5,000 | 1,750 | 2.374858 | 2.008018 | 2.053365 | 204.6s |

The unregularized run reproduces every observed milestone-021 loss through step 2,500, showing that
the stopping feature does not perturb optimization. It cuts runtime by 50.5% and avoids most of the
late final-checkpoint degradation. This is an orchestration win, not a new best model.

Weight decay improves best BPC by only `0.000251` (0.0125%) and stopped-final BPC by `0.002082`.
With one deterministic seed, those differences are practically neutral and do not justify changing
the default research conclusion. Both runs stop after the three non-improving evaluations at steps
2,000, 2,250, and 2,500.

## Controlled Generation

Prompt `Once`, 100 new tokens. Seeded decoding uses temperature 0.8, top-k 10, seed 1337.

| setting/checkpoint | greedy distinct-2 | seeded distinct-2 |
| --- | ---: | ---: |
| early-stop best | 0.5217 | 0.5417 |
| early-stop final | 0.5723 | 0.5980 |
| WD 0.01 best | 0.4326 | 0.5439 |
| WD 0.01 final | 0.4253 | 0.6380 |

Best-checkpoint likelihood and distinct-2 again disagree. Weight decay does not visibly solve
coherence or repetition; the samples remain locally plausible but semantically unstable.

## Exact Commands

```bash
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_wd0.01_earlystop.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_wd0.01_earlystop.yaml
uv run --frozen --extra dev python scripts/show_run.py --run latest --run-name gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop
uv run --frozen --extra dev python scripts/show_run.py --run latest --run-name gptiny_bytebpe512_5k_lr1e-3_ctx37_wd0.01_earlystop
uv run --frozen --extra dev make check
```

Generation used `scripts/generate.py` for both runs and `best`/`final` checkpoint kinds, once with
`--greedy --diagnostics` and once with
`--temperature 0.8 --top-k 10 --seed 1337 --diagnostics`.

Run paths:

- `runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop/2026-07-12_22-12-08`
- `runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_wd0.01_earlystop/2026-07-12_22-16-28`

## Limitations And Next Step

- Patience is evaluated only every 250 steps, so stopping reacts with bounded delay.
- One seed and one chronological split cannot establish whether the tiny WD difference is stable.
- Early stopping observes the validation set repeatedly and is part of model selection.
- Generation diagnostics are surface statistics, not human evaluation.

The next useful milestone is robustness rather than another single-run hyperparameter: repeat the
ByteBPE512 early-stopping control across multiple seeds and report mean, spread, stop-step variation,
and generation diagnostics without selecting the best seed.
