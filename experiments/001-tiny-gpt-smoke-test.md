# 001 Tiny GPT Smoke Test

## Goal

Verify that the full pipeline runs from raw text to generated text.

## Setup

- dataset: `data/raw/input.txt`, four repetitions of a short paragraph about a small model learning from a small story.
- config: `configs/tiny_gpt.yaml`
- command:

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once"
python -m pytest
python -m compileall src scripts
```

## Result

- tokenizer vocab size: 26
- checkpoint path: `checkpoints/latest.pt`
- generated sample:

```text
OncelTwnaw
lc iiyisyg ryamTsms.asocfdyIedet,asOsTx.y.iOwyitxs sacclfmfl
Tdc edtfycIOt ftIxsxnxya gnfrgOd
```

## Notes

- The output is not coherent yet, which is expected for a 5-step character-level smoke test.
- The pipeline ran end-to-end after reducing `configs/tiny_gpt.yaml` to a true smoke-test size.
- The original 1000-step, 4-layer config was too slow for a smoke test and was interrupted manually.
- Generation initially failed because the checkpoint stored a pickled `GPTConfig` object and modern PyTorch loads with `weights_only=True` by default. The checkpoint now stores model config as plain data.
- Torch warns that NumPy is not installed. The warning did not block this pipeline, but adding NumPy would quiet it.
- Validation passed: `python -m pytest` reported 6 passed, and `python -m compileall src scripts` completed successfully.
