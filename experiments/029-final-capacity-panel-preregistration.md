# 029 — Final Capacity-Controlled Panel Preregistration

## Status

This protocol is frozen before any of the three declared sources are fetched or inspected. The
study is the final planned modeling work for smaLLM. Its result will be reported without changing
the design and will lead directly to the permanent `1.0.0` completion release.

## Question and Hypotheses

Milestones 026–028 found lower sealed-test bits per character (BPC) for ByteBPE512 than for a
character tokenizer, but ByteBPE512 had roughly 103k–106k more parameters because of its larger
vocabulary. Does the advantage survive a near-matched total-parameter character control on three
fresh English genres?

For corpus \(c\), seed \(s\), and arm \(a\), let \(B_{c,s,a}\) be sealed-test BPC. The primary
contrast is

\[
\Delta^{\mathrm{matched}}_{c,s}=B_{c,s,\mathrm{ByteBPE512}}-B_{c,s,\mathrm{char136}}.
\]

The primary directional hypothesis requires a negative mean paired contrast on each corpus. The
strong outcome requires all nine paired contrasts to be negative. Secondary contrasts are
ByteBPE512 minus character-128 (continuity with prior evidence) and character-136 minus
character-128 (the effect of added character-model capacity).

All cells, ties, reversals, source problems, parameter differences, and validation-to-test gaps
will be reported. Means and population standard deviations describe only the complete fixed panel;
they are not confidence intervals or population-level hypothesis tests.

## Frozen Corpus Contract

1. **Novel:** Mary Shelley's *Frankenstein; Or, The Modern Prometheus*, Project Gutenberg ebook
   #84, `https://www.gutenberg.org/cache/epub/84/pg84.txt`.
2. **Autobiography:** Frederick Douglass's *Narrative of the Life of Frederick Douglass, an
   American Slave*, Project Gutenberg ebook #23,
   `https://www.gutenberg.org/cache/epub/23/pg23.txt`.
3. **Scientific nonfiction:** Charles Darwin's *On the Origin of Species*, Project Gutenberg
   ebook #1228, `https://www.gutenberg.org/cache/epub/1228/pg1228.txt`.

For every source, the complete body between the unique ordered Gutenberg START/END markers will be
extracted without a character budget, then normalized only by `prepare_corpus.py`. Each prepared
corpus receives a chronological 80% train, 10% validation, and 10% sealed-test split. Downloads,
extracted bodies, prepared data, tokenizers, checkpoints, and runs remain ignored; hashes and
counts will be recorded in the result report.

A source may be replaced only if its URL cannot be fetched, its marker contract fails, or its
marker title contradicts the declared work. Any replacement and rationale must be committed before
tokenizer fitting or training. Corpus size, style, vocabulary, or preliminary metrics are not
grounds for substitution.

## Frozen 3 × 3 × 3 Design

The panel crosses corpora `frankenstein`, `douglass`, and `origin`; arms `char128`, `char136`, and
`bytebpe512`; and seeds 1337, 2027, and 4242, for 27 runs.

| Arm | Tokenizer | Width | Context | Batch | Vocabulary target |
| --- | --- | ---: | ---: | ---: | ---: |
| `char128` | character | 128 | 64 | 16 | train characters + `<unk>` |
| `char136` | character | 136 | 64 | 16 | train characters + `<unk>` |
| `bytebpe512` | boundary-aware byte BPE | 128 | 37 | 27 | 512 |

Character-136 is fixed before source access. Based on the observed English character vocabularies
in milestone 028, it should be within 1.5% of ByteBPE512's total parameter count. The actual
difference is part of the result. If the tolerance is exceeded, the analyzer will reject the
capacity-matched claim and the mismatch will be reported; width will not be adjusted.

All arms use four layers, four heads, dropout 0.1, AdamW at `1e-3`, zero weight decay, exact full
validation every 250 steps, patience 3, and a 5,000-step ceiling. ByteBPE uses minimum merge
frequency 2. The context and batch settings preserve the established tokenization controls.

The committed configurations are:

```text
configs/gptiny_{frankenstein,douglass,origin}_{char128,char136,bytebpe512}_final_seed{1337,2027,4242}.yaml
```

## Access, Selection, and Analysis Rules

Tokenizer fitting uses training text only. Gradients use training tokens only. Validation selects
`best_checkpoint.pt` independently in each run. All 27 runs must be complete and must report
`test_status: sealed_unread` before any test region is tokenized or scored. Each best checkpoint is
then evaluated once with complete target-token coverage.

The capacity-panel analyzer is frozen with this protocol. It requires exactly three corpora, the
three declared arms, and the three declared seeds; verifies unique runs, complete cells, corpus and
checkpoint identity, full coverage, best-checkpoint selection, configurations, fingerprints, and
the 1.5% capacity tolerance; and computes only the three declared paired contrasts.

No test result may trigger a rerun, configuration change, replacement seed, additional corpus, or
additional test access. The project closes after reporting this panel whether the primary
hypothesis succeeds, fails, or is rejected because the parameter tolerance is not met.

## Planned Commands and Completion Gate

For each corpus, the source will be fetched and extracted with the existing Gutenberg extractor,
then prepared with `--train-split 0.8 --validation-split 0.1`. For every committed config:

```bash
python scripts/prepare_data.py --config configs/<final-config>.yaml
python scripts/evaluate_baselines.py --config configs/<final-config>.yaml
python scripts/train.py --config configs/<final-config>.yaml
```

Only after a global 27-run sealed audit:

```bash
python scripts/evaluate_test.py --run runs/<name>/<id> --checkpoint-kind best
python scripts/summarize_capacity_panel.py --cell <corpus> <arm> <run-dir> ...
```

The final report will include source and prepared hashes, counts, vocabulary and compression,
baselines, parameters, stopping and best steps, validation and test BPC, support, checkpoint hashes,
all paired contrasts, descriptive summaries, exact commands, failures, limitations, and the final
project conclusion.
