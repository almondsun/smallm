# 014 Optimizer and Sampling Diagnostics

## Goal

Test whether GPTiny's remaining quality gap is partly due to optimizer-related settings rather than architecture or tokenization, while keeping the corpus, character tokenizer, architecture, prompt, seed, and 5k-step budget fixed.

## Setup

- Source: Project Gutenberg ebook #11, _Alice's Adventures in Wonderland_ by Lewis Carroll.
- Fetch URL: `https://www.gutenberg.org/cache/epub/11/pg11.txt`.
- Extraction: body text between the START/END Project Gutenberg ebook markers, with boilerplate removed before normal corpus preparation.
- Raw characters after body extraction: 144,600.
- Prepared characters: 144,530.
- Vocab size: 75 tokenizer-derived characters.
- Train/validation split: 90/10.
- Train characters: 130,077.
- Validation characters: 14,453.
- Prepared SHA-256: `a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`.

Baseline results:

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        4.3175        75.00 equal probability for every character
unigram        3.1699        23.81 train-set character frequencies
bigram         2.4340        11.40 add-1 smoothed character transitions
```

## Implementation Changes

- Added `weight_decay: float = 0.0` to `TrainConfig`.
- Passed `config.train.weight_decay` into `torch.optim.AdamW`.
- Added `best_val_loss` and `best_val_step` to `summary.json`.
- Tracked best validation across interval evaluations and the final validation pass.
- Updated `show_run.py` to display best validation when available.
- Added tests for config loading, optimizer construction, best-validation tracking, and summary fields.

## Diagnostic Configs

All runs use the same larger Alice corpus, character tokenizer, architecture, batch size, block size, generation prompt, seed, sampling controls, and 5k-step budget.

| Config | Run name | Learning rate | Weight decay | Dropout | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| `configs/gptiny_5k.yaml` | `gptiny_5k` | 0.0003 | 0.0 | 0.1 | Control from experiment 013. |
| `configs/gptiny_5k_lr1e-4.yaml` | `gptiny_5k_lr1e-4` | 0.0001 | 0.0 | 0.1 | Lower learning rate. |
| `configs/gptiny_5k_lr1e-3.yaml` | `gptiny_5k_lr1e-3` | 0.001 | 0.0 | 0.1 | Higher learning rate. |
| `configs/gptiny_5k_wd0.01.yaml` | `gptiny_5k_wd0.01` | 0.0003 | 0.01 | 0.1 | Weight decay at control learning rate. |
| `configs/gptiny_5k_dropout0.yaml` | `gptiny_5k_dropout0` | 0.0003 | 0.0 | 0.0 | Dropout disabled. |

## Results

Margin is `bigram validation loss - GPTiny validation loss`, so positive means GPTiny beat bigram.

| Config | Run name | Run path | LR | WD | Dropout | Final train | Final val | Best val | Best step | Final ppl | Best ppl | Margin | Duration | Tokens/sec | Greedy collapse mode | Seeded top-k note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gptiny_5k.yaml` | `gptiny_5k` | `runs/gptiny_5k/2026-06-26_14-18-30` | 0.0003 | 0.0 | 0.1 | 1.9475 | 1.8604 | 1.8604 | 5000 | 6.43 | 6.43 | +0.5736 | 106.47s | 48,138 | Repeats `she`/`was`. | Word-like fragments, still incoherent. |
| `gptiny_5k_lr1e-4.yaml` | `gptiny_5k_lr1e-4` | `runs/gptiny_5k_lr1e-4/2026-06-26_14-20-30` | 0.0001 | 0.0 | 0.1 | 2.3694 | 2.3092 | 2.3092 | 5000 | 10.07 | 10.07 | +0.1248 | 99.80s | 51,363 | Repeats `the`. | Much weaker, mostly character-level fragments. |
| `gptiny_5k_lr1e-3.yaml` | `gptiny_5k_lr1e-3` | `runs/gptiny_5k_lr1e-3/2026-06-26_14-22-22` | 0.001 | 0.0 | 0.1 | 1.6413 | 1.6792 | 1.6501 | 4750 | 5.36 | 5.21 | +0.7548 | 100.13s | 51,192 | Repeats `the`/`she`/`was`/`said`. | Best texture; more recognizable Alice-like phrases, still incoherent. |
| `gptiny_5k_wd0.01.yaml` | `gptiny_5k_wd0.01` | `runs/gptiny_5k_wd0.01/2026-06-26_14-24-15` | 0.0003 | 0.01 | 0.1 | 1.9488 | 1.8601 | 1.8601 | 5000 | 6.42 | 6.42 | +0.5739 | 115.57s | 44,342 | Repeats `she`. | Nearly identical to control. |
| `gptiny_5k_dropout0.yaml` | `gptiny_5k_dropout0` | `runs/gptiny_5k_dropout0/2026-06-26_14-26-26` | 0.0003 | 0.0 | 0.0 | 1.7108 | 1.7992 | 1.7897 | 4750 | 6.04 | 5.99 | +0.6348 | 73.82s | 69,454 | Repeats `the said`/`the was`. | Better than control, but less coherent than high LR. |

## Generation Comparison

### Control

Greedy:

```text
Once the she she she she she she she was the was the she she was the she was the she she was the she was the she she was the she was the she she was the she was the she she was the she was the she she was
```

Seeded top-k:

```text
Once, by dow, “and and seet teat to be
grie sto firse the reand, the
re tarterypell a mes hades muck, the cand such, the stame
the they
pet ancre it on the prosser
she the what he cas a sur as the ond the
```

### Lower Learning Rate

Greedy:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

Seeded top-k:

```text
Oncere and, aite statthe at teanco therer thit of aly th itheand the the art indelind me the sheng
the thang whed re the merat is of oow shind.  “I hon the thesser
“I sthon ha she as dou the she as, sthe,
```

### Higher Learning Rate

Greedy:

```text
Once the she was the was and the said the she was the was the said the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was t
```

Seeded top-k:

```text
Once, but was to shattes at that it at
lowell was all the Dormoused a little offelt a me the so seemal that all
carried the Mock Turtled it a cried how the off she
shright was she caself the see was to do
```

### Weight Decay

Greedy:

```text
Once the she she she she she she she was the was the she she was the she she she she she she she she she she she she she she she she she she she she she she she she she she she she she she she she she she
```

Seeded top-k:

```text
Once, by dow, “and and seet teat to be
grie sto firse the reand, the
re tarterypell a mes cand the
sall the bouse she pucher
this of of a that. “You knove she surtent to was a she cas a sur a seearrout as
```

### Dropout Off

Greedy:

```text
Once the said the was to the said the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was the was th
```

Seeded top-k:

```text
Onced by: what angeardes at tear.
“Do all this after this it
and the her and it, and imes had she wall the
gaing in the collaid the Queen, it a much on the Queener
of to wener she caste sure, it as should
```

## Answers

- Did any optimizer setting beat the 5k control? Yes. `lr=0.001` clearly beat the control: final validation `1.6792` versus `1.8604`, and best validation `1.6501` versus `1.8604`.
- Did any setting reach similar loss faster or more stably? Yes. `lr=0.001` reached validation `1.9828` by step 1000 and `1.8074` by step 1750. Dropout-off also reached the control's final-loss neighborhood by step 4000 and finished faster wall-clock due to disabled dropout.
- Did best validation differ meaningfully from final validation? For the control, lower LR, and weight decay runs, no. For `lr=0.001`, best was `1.6501` at step 4750 versus final `1.6792`; for dropout-off, best was `1.7897` at step 4750 versus final `1.7992`. The difference is real but not dramatic.
- Did generation texture improve? Mixed. Higher LR and dropout-off produced more word-like seeded top-k samples and more recognizable Alice-specific names, but samples are still not coherent prose.
- Did greedy collapse persist? Yes. Every run collapsed under greedy decoding, though the repeated pattern changed by setting.

## Validation

```bash
curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/raw/input.txt
python - <<'PY'
from pathlib import Path

raw_path = Path("data/raw/input.txt")
text = raw_path.read_text(encoding="utf-8")
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
start = text.index(start_marker) + len(start_marker)
end = text.index(end_marker)
raw_path.write_text(text[start:end].strip() + "\n", encoding="utf-8")
PY
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "Larger public-domain prose corpus" --source-note "Project Gutenberg ebook #11, Alice's Adventures in Wonderland by Lewis Carroll, body text between START/END markers; boilerplate removed after fetch from https://www.gutenberg.org/cache/epub/11/pg11.txt"
python scripts/prepare_data.py --config configs/gptiny.yaml
python scripts/evaluate_baselines.py --config configs/gptiny.yaml
python scripts/train.py --config configs/gptiny_5k.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-4.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-3.yaml
python scripts/train.py --config configs/gptiny_5k_wd0.01.yaml
python scripts/train.py --config configs/gptiny_5k_dropout0.yaml
python scripts/show_run.py --run latest --run-name gptiny_5k
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-4
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3
python scripts/show_run.py --run latest --run-name gptiny_5k_wd0.01
python scripts/show_run.py --run latest --run-name gptiny_5k_dropout0
python scripts/generate.py --run latest --run-name gptiny_5k --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-4 --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-4 --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3 --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3 --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_wd0.01 --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_wd0.01 --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_dropout0 --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k_dropout0 --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python -m pytest
python -m compileall src scripts
```

Observed results:

- Public-domain corpus fetch passed.
- Body-only raw corpus extraction wrote 144,600 characters.
- Corpus preparation passed with 144,530 prepared characters and 75 unique characters.
- Tokenizer preparation passed with vocab size 75.
- Baseline evaluation passed and reproduced the larger-corpus bigram loss `2.4340`.
- Training passed for the control, lower LR, higher LR, weight decay, and dropout-off runs.
- Run inspection passed for all five run names and displayed best validation fields.
- Greedy and seeded top-k generation passed for all five run names.
- `python -m pytest` passed: 35 tests.
- `python -m compileall src scripts` passed.

## Limitations

- The architecture stayed fixed; this does not test model capacity.
- The tokenizer stayed character-level; this does not test BPE or subword tokenization.
- The corpus is one public-domain prose work.
- The optimizer grid is small and does not test schedules, warmup, betas, batch size, or gradient clipping.
- Runs are short local experiments, not full convergence studies.
- Generation quality remains qualitative, and greedy collapse persisted in every run.
