# 015 GPTiny Capacity and Generation Diagnostics

## Goal

Test whether GPTiny's remaining generation weakness is mainly due to limited model capacity, while keeping the larger Alice corpus, character tokenizer, optimizer setting, 5k training budget, and generation prompts controlled. Add lightweight generation diagnostics so greedy collapse can be measured instead of only described qualitatively.

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
- Control config: `configs/gptiny_5k_lr1e-3.yaml`, the best optimizer setting from experiment 014.

## Baseline Results

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        4.3175        75.00 equal probability for every character
unigram        3.1699        23.81 train-set character frequencies
bigram         2.4340        11.40 add-1 smoothed character transitions
```

## Implementation Changes

- Added `src/smallm/generation/diagnostics.py`.
- Added `scripts/generate.py --diagnostics`.
- Kept default generation output unchanged unless `--diagnostics` is passed.
- Added diagnostics tests in `tests/test_generation_diagnostics.py`.

## Diagnostics Definitions

- `repetition_rate`: fraction of adjacent character pairs where `text[i] == text[i - 1]`.
- `distinct_1`: unique character unigrams divided by total character unigrams.
- `distinct_2`: unique character bigrams divided by total character bigrams.
- `longest_repeated_token_run`: longest run of the same whitespace-delimited token.

## Capacity Configs

All runs use the same corpus, tokenizer, split, block size, batch size, 5k steps, `lr=0.001`, `weight_decay=0.0`, dropout `0.1`, prompt, seed, and generation controls.

| Config | Run name | Layers | Heads | Embedding | Parameters | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `configs/gptiny_5k_lr1e-3.yaml` | `gptiny_5k_lr1e-3` | 2 | 2 | 64 | 113,867 | High-LR control from experiment 014. |
| `configs/gptiny_5k_lr1e-3_wide.yaml` | `gptiny_5k_lr1e-3_wide` | 2 | 4 | 128 | 424,267 | Wider model, same depth. |
| `configs/gptiny_5k_lr1e-3_deep.yaml` | `gptiny_5k_lr1e-3_deep` | 4 | 4 | 128 | 820,811 | Wider and deeper model. |
| `configs/gptiny_5k_lr1e-3_compact.yaml` | `gptiny_5k_lr1e-3_compact` | 1 | 2 | 64 | 63,883 | Optional lower-capacity contrast. |

## Results

Margin is `bigram validation loss - GPTiny validation loss`, so positive means GPTiny beat bigram.

| Config | Run name | Run path | Final train | Final val | Best val | Best step | Final ppl | Best ppl | Final margin | Best margin | Duration | Tokens/sec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gptiny_5k_lr1e-3.yaml` | `gptiny_5k_lr1e-3` | `runs/gptiny_5k_lr1e-3/2026-06-26_14-55-33` | 1.6413 | 1.6792 | 1.6501 | 4750 | 5.36 | 5.21 | +0.7548 | +0.7839 | 105.77s | 48,454 |
| `gptiny_5k_lr1e-3_wide.yaml` | `gptiny_5k_lr1e-3_wide` | `runs/gptiny_5k_lr1e-3_wide/2026-06-26_14-57-42` | 1.4613 | 1.5775 | 1.5265 | 4750 | 4.84 | 4.60 | +0.8565 | +0.9075 | 221.81s | 23,097 |
| `gptiny_5k_lr1e-3_deep.yaml` | `gptiny_5k_lr1e-3_deep` | `runs/gptiny_5k_lr1e-3_deep/2026-06-26_15-01-40` | 1.2805 | 1.6447 | 1.4950 | 2500 | 5.18 | 4.46 | +0.7893 | +0.9390 | 442.20s | 11,590 |
| `gptiny_5k_lr1e-3_compact.yaml` | `gptiny_5k_lr1e-3_compact` | `runs/gptiny_5k_lr1e-3_compact/2026-06-26_15-09-17` | 1.8400 | 1.7861 | 1.7786 | 4500 | 5.97 | 5.92 | +0.6479 | +0.6554 | 61.03s | 84,004 |

Capacity improved validation. The wide model improved both final and best validation over the control. The deep model reached the best validation point overall, but its final validation rose substantially after step 2500 while training loss kept falling.

## Generation Diagnostics

### Greedy

| Run name | Repetition rate | Distinct-1 | Distinct-2 | Longest repeated token run | Qualitative note |
| --- | ---: | ---: | ---: | ---: | --- |
| `gptiny_5k_lr1e-3` | 0.0000 | 0.0588 | 0.0985 | 1 | Phrase loop around `the` / `she` / `was` / `said`; severe low diversity. |
| `gptiny_5k_lr1e-3_wide` | 0.0591 | 0.0882 | 0.2118 | 1 | Less degenerate than control, but repeats phrases like `the soon`. |
| `gptiny_5k_lr1e-3_deep` | 0.0099 | 0.1324 | 0.4039 | 2 | Best greedy diversity; still repetitive, but closer to sentence-like text. |
| `gptiny_5k_lr1e-3_compact` | 0.0000 | 0.0490 | 0.0640 | 1 | Worst greedy diversity; loops around `the was` / `she was`. |

Greedy excerpts:

```text
control: Once the she was the was and the said the she was the was the said the was the was...
wide:    Once the same the rest of the soon the way of the trees and the rest of the soon...
deep:    Once of the same to the whiting the same to say things and the reason of the sea...
compact: Once the was she was the was the was the she was she was she was the was...
```

### Seeded Top-K

| Run name | Repetition rate | Distinct-1 | Distinct-2 | Longest repeated token run | Qualitative note |
| --- | ---: | ---: | ---: | ---: | --- |
| `gptiny_5k_lr1e-3` | 0.0493 | 0.1275 | 0.4680 | 1 | Word-like but incoherent; includes Alice-like names. |
| `gptiny_5k_lr1e-3_wide` | 0.0296 | 0.1471 | 0.5714 | 1 | Better local texture and punctuation; still semantically loose. |
| `gptiny_5k_lr1e-3_deep` | 0.0345 | 0.1863 | 0.5862 | 1 | Best sampled texture; repeated `said Alice`, but most prose-like. |
| `gptiny_5k_lr1e-3_compact` | 0.0197 | 0.1324 | 0.4631 | 1 | Better than compact greedy, but still choppy and less coherent. |

Seeded top-k excerpts:

```text
control:
Once, but was to shattes at that it at
lowell was all the Dormoused a little offelt a me the so seemal that all

wide:
Once of the right the was good and manage.

At make of things the reason as all, and impatiently right that all

deep:
Once, by the Dormouse.

“Yes!” said Alice.

“I don’t thi little,” said Alice.

compact:
Once, by and the seattere the she be ther the was all you know there had and you like went!”
```

## Answers

- Did wider/deeper GPTiny improve validation loss? Yes. Wide improved final validation from `1.6792` to `1.5775` and best validation from `1.6501` to `1.5265`. Deep reached the best validation loss overall at `1.4950`.
- Did capacity improve generation diagnostics? Yes, especially for distinct-2. Greedy distinct-2 improved from `0.0985` control to `0.2118` wide and `0.4039` deep. Seeded top-k distinct-2 improved from `0.4680` control to `0.5714` wide and `0.5862` deep.
- Did greedy collapse persist quantitatively? Partly. The compact and control models still had very low greedy diversity. Wide and deep reduced the measured collapse, but greedy text still showed phrase-level loops. `longest_repeated_token_run` did not capture these phrase loops well, which is a limitation of that metric.
- Did larger capacity overfit or remain stable? Wide remained reasonably stable but final validation was worse than best. Deep showed stronger train/validation separation: best validation occurred at step 2500, then final validation rose while train loss kept falling. That is an overfit/noisy-validation signal on this single-book chronological split.
- Does the next bottleneck look more like capacity or tokenization? Capacity still helps, but it does not fully solve generation. The deep model produced the best validation and best sampled texture, yet text is still not coherent prose and greedy still loops. The next controlled milestone should be tokenization, likely character-level versus simple BPE.

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
python scripts/train.py --config configs/gptiny_5k_lr1e-3.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-3_wide.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-3_deep.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-3_compact.yaml
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3_wide
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3_deep
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3_compact
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3 --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3 --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_wide --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_wide --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_compact --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_compact --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python -m pytest
python -m compileall src scripts
```

Observed results:

- Public-domain corpus fetch passed.
- Body-only raw corpus extraction wrote 144,600 characters.
- Corpus preparation passed with 144,530 prepared characters and 75 unique characters.
- Tokenizer preparation passed with vocab size 75.
- Baseline evaluation passed and reproduced the larger-corpus bigram loss `2.4340`.
- Training passed for control, wide, deep, and compact capacity configs.
- Run inspection passed for all four run names.
- Greedy and seeded top-k generation with diagnostics passed for all four run names.
- `python -m pytest` passed: 41 tests.
- `python -m compileall src scripts` passed.

## Limitations

- Tokenization stayed character-level.
- The corpus is one public-domain prose work.
- The split is chronological 90/10, not a broad held-out prose distribution.
- Runs are short local 5k-step experiments, not full convergence studies.
- The capacity grid is limited and does not tune each model size separately.
- Diagnostics are simple proxies. They catch diversity and repeated-token runs, but not phrase-level repetition, syntax, semantics, or long-range coherence.
