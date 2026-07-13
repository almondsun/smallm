# 023 — Multi-Seed Robustness

## Goal

Measure whether ByteBPE512's advantage and early-stopping behavior survive training randomness.
Seeds 1337, 2027, and 4242 were fixed before the additional runs; all completed seeds are reported,
and no result is selected or discarded based on quality.

## Setup

Every run uses the milestone-022 unregularized configuration: the same training-only ByteBPE512
tokenizer, corpus checksum `a4c81ef23eb9…`, chronological 90/10 split, 37-token context, batch 27,
4-layer/4-head/128-wide GPTiny, dropout 0.1, AdamW `lr=1e-3`, zero weight decay, full validation
every 250 steps, and patience-3 early stopping under a 5,000-step ceiling. Training seed is the only
changed field. Generation sampling holds seed 1337 fixed to isolate model-training variation.

Population standard deviation describes this complete preregistered seed set; with only three seeds,
it is descriptive uncertainty, not a confidence interval or population estimate.
The aggregation CLI verified canonical experiment fingerprint
`ed9650e463830d0d85dbbc32a40164b6699d7033017386a2cb22f2ab035431b0` across all runs.

## Validation Results

| seed | actual steps | best step | best loss | best BPC | stopped-final BPC | duration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 2,500 | 1,750 | 2.375155 | 2.008269 | 2.055447 | 209.3s |
| 2027 | 3,000 | 2,250 | 2.410806 | 2.038414 | 2.099879 | 308.3s |
| 4242 | 2,500 | 1,750 | 2.390150 | 2.020948 | 2.052917 | 244.1s |

| metric | mean | population SD | minimum | maximum |
| --- | ---: | ---: | ---: | ---: |
| actual steps | 2,666.67 | 235.70 | 2,500 | 3,000 |
| best step | 1,916.67 | 235.70 | 1,750 | 2,250 |
| best BPC | **2.022544** | **0.012358** | 2.008269 | 2.038414 |
| stopped-final BPC | 2.069414 | 0.021567 | 2.052917 | 2.099879 |
| duration seconds | 253.93 | 41.01 | 209.32 | 308.33 |

All three seeds beat the corrected character control (`2.075981`) and BPE128 control (`2.097552`)
on best BPC. The worst tested ByteBPE512 seed retains a 0.03757 BPC advantage over character. The
tokenizer conclusion therefore survives this seed set, although its effect is smaller than the
single best seed suggested.

Seed-to-seed best-BPC SD (`0.012358`) is about 49 times milestone 022's apparent weight-decay gain
(`0.000251`). That comparison strengthens the earlier conclusion that weight decay `0.01` was
effectively neutral. Early stopping is stable in direction but not timing: seed 2027 improves later
and needs 500 additional steps.

## Controlled Generation

Prompt `Once`, 100 new tokens. Seeded decoding uses temperature 0.8, top-k 10, seed 1337.

| training seed | best greedy d-2 | best seeded d-2 | final greedy d-2 | final seeded d-2 |
| ---: | ---: | ---: | ---: | ---: |
| 1337 | 0.5217 | 0.5417 | 0.5723 | 0.5980 |
| 2027 | 0.5562 | 0.6836 | 0.5595 | 0.6591 |
| 4242 | 0.3657 | 0.6548 | 0.5056 | 0.5670 |
| mean | 0.4812 | 0.6267 | 0.5458 | 0.6080 |
| population SD | 0.0829 | 0.0612 | 0.0289 | 0.0383 |

Generation varies materially across training seeds. Seeded distinct-2 is usually higher than greedy,
but neither it nor final-checkpoint diversity tracks best validation BPC. Samples remain locally
plausible and globally incoherent, so the robustness claim is limited to held-out likelihood.

## Exact Commands

```bash
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed2027.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed2027.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed4242.yaml
uv run --frozen --extra dev python scripts/summarize_runs.py \
  runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop/2026-07-12_22-12-08 \
  runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed2027/2026-07-12_22-37-30 \
  runs/gptiny_bytebpe512_5k_lr1e-3_ctx37_earlystop_seed4242/2026-07-12_22-43-16
uv run --frozen --extra dev make check
```

Generation used `scripts/generate.py` for seeds 2027 and 4242 at `best` and `final`, once with
`--greedy --diagnostics` and once with
`--temperature 0.8 --top-k 10 --seed 1337 --diagnostics`. Seed-1337 diagnostics are the controlled
milestone-022 values.

## Limitations And Next Step

- Three seeds reveal variation but do not estimate a broad seed distribution precisely.
- Every seed shares one chronological validation split and one small English corpus.
- Repeated validation drives early stopping and model selection.
- Distinct-n is not semantic or human evaluation.

The next strong test changes the data axis: evaluate the fixed ByteBPE512 early-stopping protocol on
multiple deterministic corpus splits or an additional public-domain corpus. That would test whether
the tokenizer advantage is distribution-robust rather than merely seed-robust.
