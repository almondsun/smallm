# 011 Larger Public-Domain Corpus Experiment

## Goal

Train the same tiny character-level GPT on a larger public-domain prose corpus and compare baselines, validation loss, perplexity, and controlled generation quality against the previous 4.8k-character run.

## Corpus

- Source: Project Gutenberg ebook #11, _Alice's Adventures in Wonderland_ by Lewis Carroll.
- Fetch URL: `https://www.gutenberg.org/cache/epub/11/pg11.txt`.
- Local raw path: ignored `data/raw/input.txt`.
- Prep note: the downloaded Gutenberg file was reduced to body text between the START/END ebook markers before running the normal corpus preparation pipeline, so the final validation split is prose rather than license boilerplate.
- Raw characters after body extraction: 144,600.
- Prepared characters: 144,530.
- Previous prepared characters: 4,838.
- Vocab size: 75.
- Previous vocab size: 53.
- Train/validation split: 90/10.
- Train characters: 130,077.
- Validation characters: 14,453.
- Prepared SHA-256: `a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`.

## Baselines

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        4.3175        75.00 equal probability for every character
unigram        3.1699        23.81 train-set character frequencies
bigram         2.4340        11.40 add-1 smoothed character transitions
```

Previous 4.8k-character corpus baselines from experiment 006:

```text
baseline     val_loss   perplexity notes
--------------------------------------------------------------------------------
uniform        3.9703        53.00 equal probability for every character
unigram           inf          inf train-set character frequencies
bigram         2.5562        12.89 add-1 smoothed character transitions
```

The larger body-only corpus made the unigram baseline finite and improved the bigram baseline from 2.5562 to 2.4340.

## Tiny GPT

- Run path: `runs/tiny_gpt/2026-06-26_12-17-32`.
- Final train loss: 2.5815.
- Final validation loss: 2.5914.
- Final validation perplexity: 13.35.
- Parameters: 113,867.

Previous 4.8k-character run from experiment 010:

- Run path: `runs/tiny_gpt/2026-06-26_11-52-05`.
- Final train loss: 2.3241.
- Final validation loss: 2.4505.
- Final validation perplexity: 11.59.
- Bigram baseline loss: 2.5562.

## Comparison

| Metric | 4.8k corpus | 144.5k corpus |
| --- | ---: | ---: |
| Prepared characters | 4,838 | 144,530 |
| Vocab size | 53 | 75 |
| Bigram validation loss | 2.5562 | 2.4340 |
| Bigram perplexity | 12.89 | 11.40 |
| Tiny GPT validation loss | 2.4505 | 2.5914 |
| Tiny GPT perplexity | 11.59 | 13.35 |
| Tiny GPT margin over bigram | +0.1057 | -0.1574 |

Answers:

- Did validation loss improve? No. It worsened from 2.4505 to 2.5914.
- Did perplexity improve? No. It worsened from 11.59 to 13.35.
- Did tiny GPT beat bigram by a wider margin? No. It beat bigram by 0.1057 on the small corpus but trailed bigram by 0.1574 on the larger corpus.
- Did generated samples become less broken? Slightly for seeded top-k local character flow, but not enough to count as a quality breakthrough.
- Did greedy still collapse? Yes. Greedy still collapses into repeated `the`.

## Example Greedy Output

Command:

```bash
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --greedy --max-new-tokens 200
```

Output:

```text
Once the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

## Example Seeded Top-K Output

Command:

```bash
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
```

Output:

```text
Oncere than,” tous atthe at tean tome alang sthe anoute t theand thear thart sathous d as saneseng tlle the alisungheend sthathishoulid aerice s hithond heathes st
“I towonora sed ased sout aseeando soud,
```

## Validation

```bash
curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/raw/input.txt
python - <<'PY'
from pathlib import Path
raw_path = Path('data/raw/input.txt')
text = raw_path.read_text(encoding='utf-8')
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
start = text.index(start_marker) + len(start_marker)
end = text.index(end_marker)
raw_path.write_text(text[start:end].strip() + "\n", encoding='utf-8')
PY
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "Larger public-domain prose corpus" --source-note "Project Gutenberg ebook #11, Alice's Adventures in Wonderland by Lewis Carroll, body text between START/END markers; boilerplate removed after fetch from https://www.gutenberg.org/cache/epub/11/pg11.txt on 2026-06-26"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/evaluate_baselines.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/show_run.py --run latest --run-name tiny_gpt
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --greedy --max-new-tokens 200
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200
python -m pytest
python -m compileall src scripts
```

- Public-domain corpus fetch passed.
- Body-only raw corpus extraction wrote 144,600 characters.
- Corpus preparation passed with 144,530 prepared characters and 75 unique characters.
- Tokenizer preparation passed with vocab size 75.
- Baseline evaluation passed.
- Training passed and wrote `runs/tiny_gpt/2026-06-26_12-17-32`.
- Run inspection printed the dataset section and latest metric.
- Greedy generation passed.
- Seeded top-k generation passed.
- `python -m pytest` passed: 31 tests.
- `python -m compileall src scripts` passed.

## Limitations

- This deliberately kept the architecture, tokenizer, and `tiny_gpt.yaml` training scale fixed.
- The larger corpus helped the bigram baseline more than the tiny GPT under the current 500-step budget.
- Absolute loss is not a clean apples-to-apples comparison because the vocab changed from 53 to 75 characters.
- The chronological 90/10 split is still simple and may not represent held-out prose robustly.
- Generated samples remain mostly character-level fragments rather than coherent words or sentences.
