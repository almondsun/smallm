# 028 — Preregistered External Corpus Panel

## Preregistration

This section and all 12 referenced configurations are committed before downloading, inspecting,
preparing, tokenizing, training on, or evaluating either new corpus. Later observations will be
reported separately without changing this declaration.

### Question And Hypotheses

Does the frozen ByteBPE512 decision retain its direction across multiple seeds on two new corpus
families: a translated military treatise and a collection of political speeches and letters?

For corpus \(c\) and seed \(s\), the paired contrast is

\[
\Delta_{c,s} =
\operatorname{BPC}_{\text{ByteBPE512,test},c,s}-
\operatorname{BPC}_{\text{character,test},c,s}.
\]

The primary directional hypothesis is that the mean paired contrast is negative for each corpus.
A stronger robustness outcome requires all six paired contrasts to be negative. Ties, positive
means, positive individual pairs, and corpus interactions will be reported rather than excluded.
Population standard deviation describes only the complete three-seed panel; it is not a confidence
interval or population-level hypothesis test.

### Corpus Contract

1. **Nonfiction:** *The Art of War* by Sun Tzu, translated by Lionel Giles, Project Gutenberg
   ebook #132, fetched from
   `https://www.gutenberg.org/cache/epub/132/pg132.txt`.
2. **Speeches and letters:** *Lincoln's Inaugurals, Addresses and Letters (Selections)*, edited by
   Daniel Kilham Dodge, Project Gutenberg ebook #14721, fetched from
   `https://www.gutenberg.org/cache/epub/14721/pg14721.txt`.

For each source, the complete content between the unique ordered Gutenberg START/END markers will
be extracted without character-budget truncation, then normalized only by the existing
`prepare_corpus.py` rules. Each prepared corpus receives a chronological 80% train, 10%
validation, and 10% sealed-test split. Downloads, extracted bodies, prepared text, tokenizers,
checkpoints, metrics, and runs remain ignored; exact hashes and counts will be reported.

A source may be replaced only if its declared URL cannot be fetched, its marker contract fails, or
the marker title contradicts the declared work. Any replacement must be committed before training.
Corpus size or content style observed after a valid fetch is not grounds for substitution.

### Frozen Factorial Design

The panel crosses:

- corpora: `artofwar`, `lincoln`;
- tokenizers: character, boundary-aware lossless ByteBPE512;
- seeds: 1337, 2027, 4242.

All 12 models use 4 layers, 4 heads, width 128, dropout 0.1, AdamW at `1e-3`, zero
weight decay, full validation every 250 steps, patience 3, and a 5,000-step ceiling. Character uses
context 64 and batch 16. ByteBPE512 uses context 37, batch 27, vocabulary target 512, and minimum
merge frequency 2. No setting will change after corpus access.

The committed configurations follow:

```text
configs/gptiny_{artofwar,lincoln}_{char,bytebpe512}_sealed_seed{1337,2027,4242}.yaml
```

### Access, Selection, And Analysis Rules

Tokenizer fitting uses train text only. Gradients use train tokens only. Validation selects
`best_checkpoint.pt` independently within each run and drives the fixed early-stopping rule.
Every one of the 12 training jobs must complete before either sealed test region is tokenized or
scored. Then each best checkpoint is evaluated exactly once with full target-token coverage.

The report will include corpus hashes and counts, vocabularies, compression, baselines, parameter
counts, all validation and test results, stopping steps, checkpoint hashes, test support, paired
same-seed contrasts, per-corpus means and population standard deviations, corpus interaction by
seed, and validation-to-test gaps. No test result may trigger configuration changes, replacement
runs, seed selection, or additional use of these test segments.

## Observations

Before training, source verification found that ebook #14721's marker and front matter identify
*Speeches & Letters of Abraham Lincoln, 1832–1865*, edited by Merwin Roe, rather than the Lincoln
collection title initially declared above. This provenance correction was committed before
tokenizer fitting, baselines, training, or test access. The URL, ebook identifier, complete-body
extraction, factorial design, hypotheses, and analysis rules are unchanged.

Further observations are pending execution of the preregistered protocol.
