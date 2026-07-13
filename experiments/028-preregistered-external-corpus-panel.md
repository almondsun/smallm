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

### Audit Trail

The preregistration and 12 configs were committed as `5ee2274` at
`2026-07-13T02:15:26-05:00`, before either URL was accessed. The Lincoln provenance amendment was
committed as `f1bd534` at `2026-07-13T02:16:56-05:00`, before tokenizer fitting, baselines, or
training. The sealed-matrix loader and analysis were committed as `fb5d59c` before test access.

All 12 training jobs completed before either test region was tokenized. A global pre-access audit
found exactly 12 balanced schema-v2 runs, all with `test_status: sealed_unread`, no test metric
fields, and no existing `test_evaluation_best.json`. Each best checkpoint was then evaluated once;
the evaluator's no-overwrite rule now prevents accidental reuse of these consumed segments.

After test access, the matrix loader correctly rejected the first aggregation attempt because its
training-comparison fingerprint included `sample_seed`. The preregistered configs intentionally
pair that generation-only seed with the training seed. Commit `9d937ee` removed all sample-only
controls from the likelihood-comparison fingerprint and added a regression test. It changed no
config, model, checkpoint, evaluation artifact, or numerical result.

### Corpus And Tokenizer Provenance

| Corpus | Download bytes / SHA-256 | Extracted characters / SHA-256 | Prepared characters / SHA-256 | Train / validation / test characters |
| --- | --- | --- | --- | --- |
| Art of War | 342,105 / `c478f7e05a07bf190585fcea0bc4658ce22e47949ac18444eca53fb274878ffe` | 312,842 / `14b1ca4cf78693eb4f3ba24dad7143337bb76f6d041d6e57f2569908f12fe654` | 312,378 / `560195e8b717e83c4bd3a53949b71bac6962731c9ef940ed38301804c3545262` | 249,902 / 31,238 / 31,238 |
| Lincoln | 536,309 / `7a3ef6688c53c55fbc4c37fb7f283f17c48d21c3f3e5de7b9034fd28573923f2` | 507,454 / `1fef9ffbffe7f701caa1aeef2b6e689f0a119a96cd6f522c0ef13d231566a8a0` | 507,076 / `3b76831dfd33c8a738d4f68064c209e7278302541733bbfb6354fa998710e1f5` | 405,660 / 50,708 / 50,708 |

Character vocabularies contain the train vocabulary plus `<unk>`: 97 entries for Art of War and
86 for Lincoln. ByteBPE512 reaches its declared 512 entries on both corpora. On train text it
compresses Art of War from 249,902 characters to 149,379 tokens (`1.6730` characters/token) and
Lincoln from 405,660 to 234,775 (`1.7279` characters/token). The character models have 826,465 and
823,638 parameters respectively; both ByteBPE512 models have 929,664. This vocabulary-dependent
parameter difference is part of the frozen design, not evidence of a parameter-matched comparison.

Validation token-loss baselines were:

| Corpus | Tokenizer | Uniform | Unigram | Add-one bigram |
| --- | --- | ---: | ---: | ---: |
| Art of War | character | 4.5747 | 3.1695 | 2.5016 |
| Art of War | ByteBPE512 | 6.2383 | 4.4276 | 3.5525 |
| Lincoln | character | 4.4543 | 3.0733 | 2.4529 |
| Lincoln | ByteBPE512 | 6.2383 | 4.3502 | 3.4119 |

These token losses are within-tokenizer references only; their units differ across tokenizers.

### Validation Selection And Sealed Tests

| Corpus | Tokenizer | Seed | Actual / best step | Best validation BPC | Test BPC | Test minus validation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Art of War | character | 1337 | 5000 / 5000 | 2.154108 | 2.181844 | +0.027735 |
| Art of War | character | 2027 | 5000 / 5000 | 2.158726 | 2.177676 | +0.018950 |
| Art of War | character | 4242 | 5000 / 5000 | 2.164572 | 2.181226 | +0.016654 |
| Art of War | ByteBPE512 | 1337 | 5000 / 4250 | 2.018544 | 2.074171 | +0.055627 |
| Art of War | ByteBPE512 | 2027 | 5000 / 4500 | 2.018501 | 2.061350 | +0.042849 |
| Art of War | ByteBPE512 | 4242 | 4750 / 4000 | 2.020842 | 2.064947 | +0.044105 |
| Lincoln | character | 1337 | 5000 / 5000 | 2.139197 | 2.254961 | +0.115764 |
| Lincoln | character | 2027 | 5000 / 5000 | 2.110965 | 2.220802 | +0.109837 |
| Lincoln | character | 4242 | 5000 / 5000 | 2.114325 | 2.231029 | +0.116703 |
| Lincoln | ByteBPE512 | 1337 | 5000 / 5000 | 1.959578 | 2.082262 | +0.122684 |
| Lincoln | ByteBPE512 | 2027 | 5000 / 5000 | 1.971808 | 2.075354 | +0.103546 |
| Lincoln | ByteBPE512 | 4242 | 5000 / 4500 | 1.969978 | 2.086232 | +0.116254 |

Every test gap is positive. This is consistent with harder chronological tails, but does not by
itself distinguish distribution shift from checkpoint-selection optimism. Mean gaps are `+0.0211`
character and `+0.0475` ByteBPE512 on Art of War, and approximately `+0.1141` for both tokenizers
on Lincoln.

Full-coverage test support is 31,238 character tokens versus 18,644 ByteBPE tokens for Art of War,
and 50,708 versus 30,032 for Lincoln. The normalized likelihood denominators are 31,237 and 50,706
target characters for ByteBPE512; the one-character differences from the character denominators
come from the first token's span, and are handled by the exact coverage contract.

### Paired Confirmatory Result

| Corpus | Seed 1337 | Seed 2027 | Seed 4242 | Mean ± population SD | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Art of War | -0.107673 | -0.116326 | -0.116279 | -0.113426 ± 0.004068 | -0.116326–-0.107673 |
| Lincoln | -0.172699 | -0.145448 | -0.144797 | -0.154315 ± 0.013002 | -0.172699–-0.144797 |

The primary hypothesis is supported for both corpora, and the stronger criterion is also met: all
six preregistered same-seed contrasts are negative. Mean cell results are `2.066823 ± 0.005399`
ByteBPE512 versus `2.180249 ± 0.001837` character on Art of War, and
`2.081283 ± 0.004494` versus `2.235597 ± 0.014314` on Lincoln.

The Lincoln-minus-Art-of-War interaction is negative at every seed (`-0.065025`, `-0.029122`, and
`-0.028518`), with mean `-0.040889 ± 0.017069` BPC. Direction therefore transfers, but magnitude
does not: ByteBPE512's advantage is larger on this speeches-and-letters collection than on the
translated treatise. This is descriptive heterogeneity, not a population-level interaction test.

### Checkpoint Identity

The evaluated best-checkpoint SHA-256 prefixes, in seed order 1337 / 2027 / 4242, are:

- Art of War character: `c16dbc44365a`, `299836416066`, `fb90f8765d80`.
- Art of War ByteBPE512: `962bbf242de2`, `2ce7024cff0f`, `dded2e5d8c09`.
- Lincoln character: `68fcd2a41f9e`, `ee8e958fdb3a`, `50444690c309`.
- Lincoln ByteBPE512: `354c7c134442`, `cb37e53e0dc1`, `c693f18fc458`.

The complete hashes remain in each ignored `test_evaluation_best.json`; the report uses prefixes
for readability while corpus/checkpoint identity checks used all 64 hexadecimal characters.

### Commands And Validation

The protocol used the repository-native commands below for each corpus/config/run. Angle-bracketed
arguments vary only by the committed config or the hashed local artifact described above:

```bash
python scripts/prepare_corpus.py --input <extracted-body> --output <prepared-corpus> \
  --stats <stats-json> --manifest <sealed-manifest> --train-split 0.8 \
  --validation-split 0.1 --source-name <verified-title> --source-note <provenance>
python scripts/prepare_data.py --config configs/<panel-config>.yaml
python scripts/evaluate_baselines.py --config configs/<panel-config>.yaml
python scripts/train.py --config configs/<panel-config>.yaml
python scripts/evaluate_test.py --run runs/<name>/<id> --checkpoint-kind best
```

The exact final matrix command was:

```bash
python scripts/summarize_test_matrix.py \
  --reference-tokenizer char --candidate-tokenizer byte_bpe \
  --cell artofwar char runs/gptiny_artofwar_char_sealed_seed1337/2026-07-13_02-25-10 \
  --cell artofwar char runs/gptiny_artofwar_char_sealed_seed2027/2026-07-13_02-33-27 \
  --cell artofwar char runs/gptiny_artofwar_char_sealed_seed4242/2026-07-13_02-41-51 \
  --cell artofwar byte_bpe runs/gptiny_artofwar_bytebpe512_sealed_seed1337/2026-07-13_02-51-59 \
  --cell artofwar byte_bpe runs/gptiny_artofwar_bytebpe512_sealed_seed2027/2026-07-13_03-00-35 \
  --cell artofwar byte_bpe runs/gptiny_artofwar_bytebpe512_sealed_seed4242/2026-07-13_03-09-17 \
  --cell lincoln char runs/gptiny_lincoln_char_sealed_seed1337/2026-07-13_03-17-06 \
  --cell lincoln char runs/gptiny_lincoln_char_sealed_seed2027/2026-07-13_03-27-18 \
  --cell lincoln char runs/gptiny_lincoln_char_sealed_seed4242/2026-07-13_03-36-41 \
  --cell lincoln byte_bpe runs/gptiny_lincoln_bytebpe512_sealed_seed1337/2026-07-13_03-47-12 \
  --cell lincoln byte_bpe runs/gptiny_lincoln_bytebpe512_sealed_seed2027/2026-07-13_03-55-25 \
  --cell lincoln byte_bpe runs/gptiny_lincoln_bytebpe512_sealed_seed4242/2026-07-13_04-04-20
```

The aggregator verifies the complete declared matrix, cell comparability, corpus and checkpoint
identity, best-checkpoint selection, full coverage, balanced seeds, and unique run paths before
reporting any contrast. Focused Ruff, mypy, and 24 evaluation tests passed after the fingerprint
repair. The final frozen quality gate passed Ruff formatting/lint, mypy over 44 source files,
166 tests at 90.74% coverage, compileall, and Markdown link validation. `pip-audit` found no known
vulnerabilities; the local `smallm` package is correctly skipped because it is not a PyPI release.

### Conclusion And Limits

Across two preregistered external corpora and three preregistered seeds, ByteBPE512 robustly beats
the character control on one-shot sealed-test BPC. Together with milestones 025–027, this makes the
direction a repeated empirical result in this small lab rather than a one-corpus or one-seed
accident. It does not establish universality, a causal mechanism, or a production tokenizer claim.

The panel has only two new English public-domain sources; both are single chronological documents,
their sizes differ, Gutenberg bodies include editorial material, the Lincoln metadata required a
transparent correction, and vocabulary-dependent embeddings give ByteBPE512 roughly 103k–106k
more parameters. Corpora are fixed rather than sampled, population standard deviations are purely
descriptive, and both new terminal regions are now consumed. The cleanest next modeling question
is a preregistered parameter-matched architecture ablation on new sealed data, not further tuning
against any test segment reported here.
