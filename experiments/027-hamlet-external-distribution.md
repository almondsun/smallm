# 027 — Hamlet External-Distribution Replication

## Preregistration

This section was committed before downloading, inspecting, preparing, tokenizing, training on, or
evaluating the Hamlet corpus. Later sections will preserve this declaration and separately report
observations.

### Question And Hypothesis

Does the frozen ByteBPE512 decision from milestones 025–026 retain its direction on a dramatic
play rather than narrative prose?

The directional hypothesis is that ByteBPE512 will have lower sealed-test bits per character (BPC)
than the character tokenizer. The null outcome includes a tie or a character advantage. No minimum
effect size is required, and the observed margin will be reported without rounding before the
comparison is calculated.

### Corpus Contract

- Source: *Hamlet* by William Shakespeare, Project Gutenberg ebook #1524.
- URL: `https://www.gutenberg.org/cache/epub/1524/pg1524.txt`.
- Extraction: the complete text between the unique ordered Gutenberg START/END markers, with no
  character-budget truncation.
- Normalization: the existing `prepare_corpus.py` rules only.
- Split: chronological 80% train, 10% validation, 10% sealed test after normalization.
- The raw download, extracted body, prepared corpus, tokenizers, checkpoints, and run artifacts stay
  ignored; hashes and exact counts will be recorded here.

Hamlet was selected because dialogue, verse, speaker labels, and stage directions differ
structurally from the Alice and Peter Pan prose distributions. No alternative corpus will be
substituted based on model results. Substitution is permitted only if the declared source cannot be
fetched or fails the existing deterministic Gutenberg marker contract, and any substitution must be
recorded before training.

### Frozen Comparison

Both models use seed 1337, 4 layers, 4 heads, width 128, dropout 0.1, AdamW at `1e-3`, zero weight
decay, full validation every 250 steps, patience 3, and a 5,000-step ceiling. Character uses context
64 and batch 16. ByteBPE512 uses the existing boundary-aware lossless tokenizer, context 37, batch
27, vocabulary target 512, and minimum merge frequency 2. These settings match milestone 026 and
will not change after corpus access.

The committed configurations are:

- `configs/gptiny_hamlet_char_sealed.yaml`
- `configs/gptiny_hamlet_bytebpe512_sealed.yaml`

### Decision And Access Rule

Tokenizer fitting uses training text only. Gradient updates use training tokens only. Validation
selects each run's `best_checkpoint.pt` and drives early stopping. Both training runs must finish
before either test segment is tokenized or scored. Each best checkpoint is then evaluated exactly
once with full target coverage using `scripts/evaluate_test.py`.

The primary comparison is

\[
\Delta_{\text{Hamlet}} =
\operatorname{BPC}_{\text{ByteBPE512,test}}-
\operatorname{BPC}_{\text{character,test}}.
\]

The hypothesis is supported when \(\Delta_{\text{Hamlet}}<0\). Validation results, stopping steps,
token counts, baselines, test loss, test BPC, checkpoint hashes, and validation-to-test gaps will all
be reported. Test results will not trigger configuration changes or additional Hamlet runs.

## Observations

The preregistration above was committed as `6f130d5b3c9c7b9deaeac09d8e4bb2bd8d325a33` at
`2026-07-13T01:50:48-05:00`, before the first network request. The declared protocol was executed
without substitution or post-access configuration changes.

### Corpus And Provenance

The Project Gutenberg download is 202 KiB with SHA-256
`31584c19795779431b5933499a45b9cecb03e8c246b6ea7e245a2b6d65e8e784`. Deterministic marker
extraction produced 177,962 characters with SHA-256
`ebce1da4e6c704d3969efe6c56bd09c9392f1291c59f857348a4b9802696691f`. Standard normalization
produced 177,932 characters, 69 distinct source characters, and SHA-256
`cf18ea4afacfe22a86e74ae5f524a2017cd6f1079bc5b72d884cea5498567c1e`.

| train characters | validation characters | sealed-test characters |
| ---: | ---: | ---: |
| 142,345 | 17,793 | 17,794 |

The character tokenizer has 70 tokens including its unknown token. ByteBPE512 reached its declared
512-token vocabulary and encoded training text in 86,403 tokens, or 1.6475 source characters per
token. Vocabulary-dependent embeddings yield 819,526 character-model parameters and 929,664
ByteBPE512 parameters; this remains an architecture-family comparison rather than a
parameter-matched comparison.

### Validation Baselines

| tokenizer | uniform loss | unigram loss | add-one bigram loss |
| --- | ---: | ---: | ---: |
| character | 4.2485 | 3.3269 | 2.5210 |
| ByteBPE512 | 6.2383 | 4.4516 | 3.7830 |

These are validation token losses and are comparable only within a tokenizer.

### Training And Frozen Selection

Both runs completed before either test evaluation. Their summaries contained the 17,794-character
sealed count but no test tokens, loss, or BPC.

| tokenizer | actual steps | best step | best validation loss | best validation BPC | final validation BPC | duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| character | 3,500 | 2,750 | 1.756849 | 2.534598 | 2.547963 | 349.4s |
| ByteBPE512 | 3,000 | 2,250 | 2.802604 | **2.525928** | 2.578391 | 263.5s |

ByteBPE512's validation advantage is 0.008670 BPC. Both tokenizers stopped early after three exact
non-improvements, and both final checkpoints are worse than their selected best checkpoints.

Run paths are:

- `runs/gptiny_hamlet_char_sealed/2026-07-13_01-52-34`
- `runs/gptiny_hamlet_bytebpe512_sealed/2026-07-13_01-59-06`

### One-Shot Sealed Test

The two best checkpoints were evaluated once, after both training jobs completed.

| tokenizer | test tokens | target tokens | target characters | test loss | test BPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| character | 17,794 | 17,793 | 17,793 | 1.609405 | 2.321881 |
| ByteBPE512 | 10,813 | 10,812 | 17,792 | 2.571678 | **2.254615** |

Both artifacts report full target-token coverage. ByteBPE512's first autoregressive input token
spans two source characters, while the character model's spans one; therefore their scored target
character counts differ by one. Each BPC denominator matches the characters completed by its scored
target tokens.

The preregistered contrast is

\[
\Delta_{\text{Hamlet}}=2.2546152062724256-2.321880862208522
=\mathbf{-0.06726565593609646}\ \text{BPC}.
\]

The directional hypothesis is supported. ByteBPE512 now beats character on sealed test BPC for
Alice, Peter Pan, and Hamlet. The effect size is not stable: Hamlet's validation margin is only
0.008670 BPC, while its test margin is 0.067266 BPC.

Unlike milestone 026, Hamlet's terminal segment is easier than its middle validation segment. The
test-minus-best-validation gap is -0.212717 BPC for character and -0.271313 for ByteBPE512. This
does not prove improved generalization; chronological regions can differ in speaker mix, scene
structure, verse density, and intrinsic entropy. It does show that the prior harder-tail pattern is
not distribution-invariant.

Checkpoint SHA-256 values are:

- character: `cc75d4fdb311f7b61da2ba1f0b2d674e3f4f7f7c42a24eafbc9246320ffff35d`
- ByteBPE512: `36022b1879f8c664c2a0c89bfc426bf94721b7805822b48f08fd4b31580fe224`

### Exact Commands

```bash
curl -fL https://www.gutenberg.org/cache/epub/1524/pg1524.txt \
  -o data/raw/hamlet_gutenberg.txt
sha256sum data/raw/hamlet_gutenberg.txt
.venv/bin/python scripts/extract_gutenberg.py \
  --input data/raw/hamlet_gutenberg.txt --output data/raw/hamlet_body.txt
.venv/bin/python scripts/prepare_corpus.py \
  --input data/raw/hamlet_body.txt --output data/processed/hamlet_corpus.txt \
  --stats data/processed/hamlet_corpus_stats.json \
  --manifest data/processed/hamlet_corpus_sealed_manifest.json \
  --source-name "Hamlet by William Shakespeare" \
  --source-note "Project Gutenberg ebook #1524; complete body between unique START/END markers; fetched from https://www.gutenberg.org/cache/epub/1524/pg1524.txt" \
  --train-split 0.8 --validation-split 0.1
.venv/bin/python scripts/prepare_data.py --config configs/gptiny_hamlet_char_sealed.yaml
.venv/bin/python scripts/evaluate_baselines.py --config configs/gptiny_hamlet_char_sealed.yaml
.venv/bin/python scripts/prepare_data.py --config configs/gptiny_hamlet_bytebpe512_sealed.yaml
.venv/bin/python scripts/evaluate_baselines.py --config configs/gptiny_hamlet_bytebpe512_sealed.yaml
.venv/bin/python scripts/train.py --config configs/gptiny_hamlet_char_sealed.yaml
.venv/bin/python scripts/train.py --config configs/gptiny_hamlet_bytebpe512_sealed.yaml
.venv/bin/python scripts/evaluate_test.py \
  --run runs/gptiny_hamlet_char_sealed/2026-07-13_01-52-34 --checkpoint-kind best
.venv/bin/python scripts/evaluate_test.py \
  --run runs/gptiny_hamlet_bytebpe512_sealed/2026-07-13_01-59-06 --checkpoint-kind best
make PYTHON=.venv/bin/python check
make PYTHON=.venv/bin/python audit
```

### Limitations And Decision

- This is one seed on one play, not a variance estimate for drama or Shakespeare.
- Alice, Peter Pan, and Hamlet remain canonical English literary texts despite the structural shift.
- Vocabulary-dependent parameter counts remain unequal.
- The one-token autoregressive prefix causes a one-character difference in scored BPC support.
- A chronological split measures forward textual position, not exchangeable sampling.
- The Hamlet test segment is now consumed and cannot guide further tuning or reruns.

The frozen ByteBPE512 decision survives the preregistered external-distribution test. The honest
claim is directional robustness across three texts, not a universal margin or universal
generalization gap. A stronger next milestone would preregister a small corpus-family panel that
includes nonfiction or speeches and multiple seeds, with new sealed segments defined before any
model access.
