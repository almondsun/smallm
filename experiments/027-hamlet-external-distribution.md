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

Pending execution of the preregistered protocol.
