# 024 — Cross-Corpus Robustness

## Goal

Test whether the fixed ByteBPE512 early-stopping protocol still beats a character control on a
different public-domain book. Peter Pan was chosen before training; the corpus, character budget,
seed, architecture, optimizer, validation schedule, and decoding controls were fixed before either
model result was observed.

## Corpus And Provenance

The source is *Peter Pan* by J. M. Barrie, Project Gutenberg ebook #16, fetched from
`https://www.gutenberg.org/cache/epub/16/pg16.txt`. The downloaded UTF-8 file has SHA-256
`6b08714281fe38266a756741e4c62915cda7536c2f78cca501f7fd53f3f445ae`.

`scripts/extract_gutenberg.py` selects the text between the unique ordered Gutenberg START/END
markers, removes leading blank lines, and retains the first 144,530 body characters. That raw body
has SHA-256 `3b8bb7fc929423926bbefe595cd70a8c58e10e6227a90b0916515750e08aa97d`.
Standard normalization produces 144,489 characters (130,040 train; 14,449 validation), 79 distinct
characters, and SHA-256
`16e4f26e7e5287dccced8520bd67965666c51c69afe0ed752f3b7d92f4693612`.
The budget is matched before normalization to the Alice study; the prepared corpora are therefore
near-matched, not byte-identical in size.

## Setup

Both runs use seed 1337, a chronological 90/10 split, full validation every 250 steps, patience 3,
a 5,000-step ceiling, 4 layers, 4 heads, width 128, dropout 0.1, AdamW at `1e-3`, and no weight
decay. The character model uses 64 tokens and batch 16. ByteBPE512 is fitted on training text only
and uses 37 tokens and batch 27. Its training compression is 1.7271 characters/token, so its
37-token window covers about 63.90 characters versus 64 for the character model.

Vocabulary-dependent embeddings make the parameter counts unequal: 822,096 for character and
929,664 for ByteBPE512. This is the same architecture family, not a parameter-matched comparison.

## Baselines

| tokenizer | uniform loss | unigram loss | add-one bigram loss |
| --- | ---: | ---: | ---: |
| character (80 tokens) | 4.3820 | 3.1119 | 2.4078 |
| ByteBPE512 | 6.2383 | 4.3080 | 3.6856 |

Token losses are not comparable across tokenizers; these rows are within-tokenizer references.

## Validation Results

| tokenizer | actual steps | best step | best token loss | best BPC | final BPC | stop reason | duration |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| character | 5,000 | 4,750 | 1.505551 | 2.172051 | 2.178503 | max steps | 496.5s |
| ByteBPE512 | 2,750 | 2,000 | 2.523778 | **2.153930** | 2.179118 | early stopping | 259.7s |

ByteBPE512 wins by 0.018121 BPC, about 0.83% relative to the character result. The direction matches
Alice, but this margin is comparable to the seed variation measured for ByteBPE512 in milestone
023. One seed on one additional book therefore supports cross-corpus replication of the direction,
not a stable effect-size estimate.

Early stopping again removes unnecessary training: ByteBPE512 stops after 55% of the ceiling. The
character model continues improving until step 4,750 and never triggers patience, showing that a
single stopping schedule need not imply similar learning dynamics across tokenizers or corpora.

## Controlled Generation

Prompt `Once`, 100 new tokens. Seeded decoding uses temperature 0.8, top-k 10, seed 1337.

| tokenizer/checkpoint | greedy distinct-2 | seeded distinct-2 |
| --- | ---: | ---: |
| character best | 0.2524 | 0.6893 |
| character final | 0.2136 | 0.6117 |
| ByteBPE512 best | 0.2662 | 0.5238 |
| ByteBPE512 final | 0.5138 | 0.6011 |

Best character seeded output begins `Once. He was the same out them to first her.` Best ByteBPE512
seeded output begins `Once you to be a little house.` Both have local book-like texture but weak
syntax and no sustained coherence. Greedy outputs repeat phrases. Validation BPC and distinct-2 do
not identify the same checkpoint, so generation remains a separate diagnostic rather than proof of
language quality.

## Exact Commands

```bash
curl -fL https://www.gutenberg.org/cache/epub/16/pg16.txt -o data/raw/peter_pan_gutenberg.txt
sha256sum data/raw/peter_pan_gutenberg.txt
uv run --frozen --extra dev python scripts/extract_gutenberg.py \
  --input data/raw/peter_pan_gutenberg.txt --output data/raw/peter_pan_body.txt \
  --max-characters 144530
uv run --frozen --extra dev python scripts/prepare_corpus.py \
  --input data/raw/peter_pan_body.txt --output data/processed/peter_pan_corpus.txt \
  --stats data/processed/peter_pan_corpus_stats.json \
  --manifest data/processed/peter_pan_corpus_manifest.json \
  --source-name "Peter Pan by J. M. Barrie" \
  --source-note "Project Gutenberg ebook #16; body between START/END markers; first 144530 body characters"
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_peterpan_char_5k_lr1e-3_earlystop.yaml
uv run --frozen --extra dev python scripts/evaluate_baselines.py --config configs/gptiny_peterpan_char_5k_lr1e-3_earlystop.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_peterpan_char_5k_lr1e-3_earlystop.yaml
uv run --frozen --extra dev python scripts/prepare_data.py --config configs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml
uv run --frozen --extra dev python scripts/evaluate_baselines.py --config configs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml
uv run --frozen --extra dev python scripts/train.py --config configs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml
uv run --frozen --extra dev make check
```

Run inspection used `scripts/show_run.py`. Generation used `scripts/generate.py` for `best` and
`final`, once with `--greedy --diagnostics` and once with
`--temperature 0.8 --top-k 10 --seed 1337 --diagnostics`.

## Limitations And Next Step

- Peter Pan is a new book but remains English literary prose from Project Gutenberg.
- One shared seed cannot separate corpus effects from seed-by-corpus interactions.
- The chronological tail may differ in difficulty between books.
- Parameter counts differ because vocabularies differ.
- Repeated validation is used for checkpoint selection; there is no sealed test set.
- Distinct-n does not measure factuality, grammar, or long-range coherence.

The next decision-grade experiment is a preregistered corpus-by-seed matrix: the same three seeds on
Alice and Peter Pan, ideally with a sealed terminal test segment. That would estimate tokenizer,
corpus, seed, and interaction effects instead of extrapolating from one run per new distribution.
