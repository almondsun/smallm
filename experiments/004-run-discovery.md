# 004 Run Discovery

## Goal

Make preserved training runs easy to find, inspect, and reuse from the command line.

## Example `list_runs` Output

```text
run_name     run_id                    train        val  duration     params checkpoint
----------------------------------------------------------------------------------------------------
smoke        2026-06-25_23-30-11      3.4052     3.3096        0s     15,482 runs/smoke/2026-06-25_23-30-11/checkpoint.pt
smoke        2026-06-25_23-32-31      3.4052     3.3096        0s     15,482 runs/smoke/2026-06-25_23-32-31/checkpoint.pt
smoke        2026-06-25_23-52-37      3.4052     3.3096        0s     15,482 runs/smoke/2026-06-25_23-52-37/checkpoint.pt
```

## Example `show_run` Output

```text
run: runs/smoke/2026-06-25_23-52-37
config: runs/smoke/2026-06-25_23-52-37/config.yaml
metrics: runs/smoke/2026-06-25_23-52-37/metrics.jsonl
summary: runs/smoke/2026-06-25_23-52-37/summary.json
checkpoint: runs/smoke/2026-06-25_23-52-37/checkpoint.pt
last metric: step=5 train_loss=3.4051673412323 val_loss=3.3095778226852417
sample:
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
python scripts/list_runs.py
python scripts/show_run.py --run runs/smoke/2026-06-25_23-52-37
python scripts/generate.py --run runs/smoke/2026-06-25_23-52-37 --prompt "Once"
python -m pytest
python -m compileall src scripts
```

- Data preparation passed with tokenizer vocab size 26.
- Smoke training passed and wrote `runs/smoke/2026-06-25_23-52-37/`.
- `list_runs.py` found all local smoke runs.
- `show_run.py` printed artifact paths, last metric, and sample text.
- `generate.py --run ...` generated text from a preserved run checkpoint.
- `show_run.py --run latest --run-name smoke` resolved the latest smoke run.
- `generate.py --run latest --run-name smoke --prompt "Once"` generated text from the latest smoke run.
- `python -m pytest` passed: 15 tests.
- `python -m compileall src scripts` passed.

## Notes

- Run discovery is dependency-free and filesystem-based.
- Generated run artifacts remain ignored by git.
- There is no run deletion, comparison, or sorting by metric yet.
