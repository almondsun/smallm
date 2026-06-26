# 013 GPTiny Training Budget and Optimization

## Goal

Make `gptiny` the canonical model/config/run name for the small GPT family, then test whether the character-level GPT was mainly undertrained on the larger public-domain Alice corpus.

## Naming Migration

`gptiny` supersedes the earlier `tiny_gpt` run and config name. Current configs, command examples, defaults, and run names now use `gptiny`; prose uses GPTiny as the display name.

The old `tiny_gpt` literal remains only in historical experiment reports and their factual run paths, for example `runs/tiny_gpt/...` from experiments 001-011. Those paths were not rewritten because they describe already-created artifacts.

## Setup

- Source: Project Gutenberg ebook #11, _Alice's Adventures in Wonderland_ by Lewis Carroll.
- Fetch URL: `https://www.gutenberg.org/cache/epub/11/pg11.txt`.
- Extraction: body text between the START/END Project Gutenberg ebook markers, with boilerplate removed before normal corpus preparation.
- Prepared characters: 144,530.
- Vocab size: 75 tokenizer-derived characters.
- Train/validation split: 90/10.
- Train characters: 130,077.
- Validation characters: 14,453.
- Prepared SHA-256: `a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`.
- Architecture: unchanged from the prior 500-step config: block size 64, 2 layers, 2 heads, embedding size 64, dropout 0.1.
- Learning rate: unchanged at `0.0003`.

`model.vocab_size` remains a placeholder default in YAML configs. Training replaces it with the tokenizer-derived vocabulary size and stores the actual value in the checkpoint and summary.

## Baseline Results

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        4.3175        75.00 equal probability for every character
unigram        3.1699        23.81 train-set character frequencies
bigram         2.4340        11.40 add-1 smoothed character transitions
```

## Budget Sweep

Margin is `bigram validation loss - GPTiny validation loss`, so positive means GPTiny beat bigram.

| Config | Run name | Run path | Max steps | Final train loss | Final val loss | Final val ppl | Margin vs bigram | Duration | Tokens/sec | Greedy collapsed? | Seeded top-k note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `configs/gptiny.yaml` | `gptiny` | `runs/gptiny/2026-06-26_13-39-29` | 500 | 2.5815 | 2.5914 | 13.35 | -0.1574 | 10.87s | 47,551 | Yes, repeated `the`. | Fragmentary character flow, similar to experiment 011. |
| `configs/gptiny_1k.yaml` | `gptiny_1k` | `runs/gptiny_1k/2026-06-26_13-40-11` | 1,000 | 2.4671 | 2.4677 | 11.79 | -0.0337 | 33.40s | 30,795 | Yes, repeated `the`. | Slightly more word-like fragments, still mostly broken. |
| `configs/gptiny_2k.yaml` | `gptiny_2k` | `runs/gptiny_2k/2026-06-26_13-41-04` | 2,000 | 2.3670 | 2.2187 | 9.20 | +0.2153 | 41.81s | 49,098 | Yes, repeated `the`. | More frequent word-like chunks, still incoherent. |
| `configs/gptiny_5k.yaml` | `gptiny_5k` | `runs/gptiny_5k/2026-06-26_13-42-16` | 5,000 | 1.9488 | 1.8601 | 6.42 | +0.5739 | 124.85s | 41,043 | Yes, shifted to repeated `she`/`was`. | Best local texture, still not coherent prose. |

## Generation Comparison

### 500 Steps

Greedy:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

Seeded top-k:

```text
Oncere than,” tous atthe at tean tome alang sthe anoute t theand thear thart sathous d as saneseng tlle the alisungheend sthathishoulid aerice s hithond heathes st
“I towonora sed ased sout aseeando soud,
```

### 1k Steps

Greedy:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

Seeded top-k:

```text
Oncere and, witong atthe at teandoug t
wang sthof aly t t it and thear thart indelind me thedesthe alle the alis the the merat is oulid ast ancr hithond heaicessereng towon ha sed ased s thed ind to sthe,
```

### 2k Steps

Greedy:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

Seeded top-k:

```text
Oncere thand ite,” and se theeancoug ther thit ad aby th itt and the the art ind ous ime thad shard
the the alis the the merat the the the the warit on the the she
“I stho wha she as doust ousee as, sould
```

### 5k Steps

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

## Answers

- Did longer training close the gap to bigram? Yes. 1k nearly closed it, 2k beat it, and 5k improved substantially.
- Did GPTiny beat bigram? Yes, at 2k and 5k steps.
- Did validation loss improve monotonically across completed budget runs? Yes: `2.5914 -> 2.4677 -> 2.2187 -> 1.8601`.
- Did samples improve? Seeded top-k samples improved in local character and word texture, especially at 5k, but they are still not coherent prose.
- Did greedy collapse persist? Yes. The collapse shifted from repeated `the` to repeated `she`/`was` by 5k, but greedy decoding still collapsed.

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
python scripts/train.py --config configs/gptiny.yaml
python scripts/train.py --config configs/gptiny_1k.yaml
python scripts/train.py --config configs/gptiny_2k.yaml
python scripts/train.py --config configs/gptiny_5k.yaml
python scripts/show_run.py --run latest --run-name gptiny
python scripts/show_run.py --run latest --run-name gptiny_1k
python scripts/show_run.py --run latest --run-name gptiny_2k
python scripts/show_run.py --run latest --run-name gptiny_5k
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_1k --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_1k --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_2k --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_2k --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name gptiny_5k --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python -m pytest
python -m compileall src scripts
```

Observed results:

- Public-domain corpus fetch passed.
- Body-only raw corpus extraction wrote 144,600 characters.
- Corpus preparation passed with 144,530 prepared characters and 75 unique characters.
- Tokenizer preparation passed with vocab size 75.
- Baseline evaluation passed and reproduced the larger-corpus bigram loss `2.4340`.
- Training passed for 500, 1k, 2k, and 5k budgets.
- Run inspection passed for all four run names.
- Greedy and seeded top-k generation passed for all four run names.
- `python -m pytest` passed: 32 tests.
- `python -m compileall src scripts` passed.

## Limitations

- The architecture stayed fixed; this does not test model capacity.
- The tokenizer stayed character-level; this does not test BPE or subword tokenization.
- The optimizer settings stayed fixed except for training budget.
- The corpus is one public-domain prose work.
- The split is chronological 90/10, so validation may not represent broader prose robustly.
- The 5k result is useful but still a short local experiment, not a full convergence study.
