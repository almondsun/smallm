# 002 Training Logger and Tiny Config

## Goal

Separate fast smoke-test execution from real tiny GPT training, and add a professional terminal training progress logger.

## Config Changes

- `configs/smoke.yaml`: 1 layer, 1 head, 32 embedding size, 32 block size, batch size 4, 5 steps.
- `configs/tiny_gpt.yaml`: 2 layers, 2 heads, 64 embedding size, 64 block size, batch size 16, 500 steps.
- `numpy` is now a runtime dependency to avoid Torch startup warnings in this environment.

## Example CLI Output

```text
smaLLM training progress logger
====================================================================================
         project: smaLLM tiny GPT
          device: cpu
      vocab size: 26
      block size: 32
          layers: 1
           heads: 1
   embedding dim: 32
      batch size: 4
       max steps: 5
      parameters: 15,482
      checkpoint: checkpoints/latest.pt
------------------------------------------------------------------------------------
         step train_loss   val_loss         lr  elapsed  tokens/sec      eta
------------------------------------------------------------------------------------
     1/5          3.3685          -   3.00e-04       0s        2634       0s
     5/5          3.4052     3.3096   3.00e-04       0s        8586       0s
------------------------------------------------------------------------------------
training complete in 0s
final train loss: 3.4052
checkpoint saved: checkpoints/latest.pt
```

## Validation Results

```bash
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once"
python -m pytest
python -m compileall src scripts
```

- Data preparation passed with tokenizer vocab size 26.
- Smoke training passed and wrote `checkpoints/latest.pt`.
- Generation passed with a short incoherent character-level sample, as expected.
- `python -m pytest` passed: 9 tests.
- `python -m compileall src scripts` passed.

## Tiny Run Learning Signal

`python scripts/train.py --config configs/tiny_gpt.yaml` completed on CPU in 9 seconds.

- First logged train loss: 3.2168 at step 10.
- First validation loss: 2.4818 at step 50.
- Final train loss: 1.2704 at step 500.
- Final validation loss: 1.2171 at step 500.

The tiny run shows observable learning on the small local corpus. This is not evidence of generalization; it mainly confirms the training loop, validation loop, and logging path are working.

## Remaining Limitations

- The corpus is tiny and local-only.
- The tokenizer is still character-level.
- Checkpoints always write to `checkpoints/latest.pt`.
- The logger is intentionally plain terminal output, not experiment tracking.
