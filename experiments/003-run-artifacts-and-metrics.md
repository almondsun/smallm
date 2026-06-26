# 003 Run Artifacts and Metrics

## Goal

Verify that each training run produces reproducible artifacts: config, metrics, checkpoint, summary, and generated sample.

## Implementation

Training now creates a unique run directory under `runs/<run-name>/<run-id>/`.

Example smoke run:

```text
runs/smoke/2026-06-25_23-32-31/
  checkpoint.pt
  config.yaml
  metrics.jsonl
  sample.txt
  summary.json
```

The checkpoint stores model weights, model config, run path, and embedded tokenizer state so generation does not depend on ignored tokenizer artifacts.

## Metrics Evidence

`metrics.jsonl` records one JSON object per logged row:

```json
{"elapsed_seconds": 0.03023085099994205, "learning_rate": 0.0003, "step": 5, "tokens_per_second": 21170.426198098983, "train_loss": 3.4051673412323, "val_loss": 3.3095778226852417}
```

`summary.json` records final run metadata:

```json
{
  "checkpoint_path": "runs/smoke/2026-06-25_23-32-31/checkpoint.pt",
  "final_train_loss": 3.4051673412323,
  "final_val_loss": 3.3095778226852417,
  "parameter_count": 15482,
  "vocab_size": 26
}
```

Trainer-generated sample:

```text
Oncehl
T Ipeimprwledd
ysww
x,ydrgtcxsdnspwdd.mo
iodoOcf aes mIxf.exinxo i t.e xeyi.od.yxiTTnfoTssIOwwsxr
```

## Validation Results

```bash
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/generate.py --checkpoint runs/smoke/2026-06-25_23-32-31/checkpoint.pt --prompt "Once"
python -m pytest
python -m compileall src scripts
```

- Data preparation passed with tokenizer vocab size 26.
- Smoke training passed and wrote a preserved run under `runs/smoke/2026-06-25_23-32-31/`.
- Manual generation from the preserved checkpoint passed.
- `python -m pytest` passed: 12 tests.
- `python -m compileall src scripts` passed.

## Remaining Limitations

- Run directories are timestamp-based and local-only.
- There is no run index or comparison command yet.
- The sample is generated with the configured prompt and default stochastic sampling.
- Primary checkpoints now live under `runs/`.
