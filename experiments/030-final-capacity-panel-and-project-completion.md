# 030 — Final Capacity-Controlled Panel and Project Completion

## Status and Decision Contract

This report completes the protocol preregistered in [milestone 029](029-final-capacity-panel-preregistration.md)
at commit `9de60ae`, before any declared source was fetched or inspected. No corpus, arm, seed,
training rule, capacity tolerance, contrast, or completion condition changed after access. This is
the final planned modeling study and closes smaLLM's research roadmap regardless of outcome.

The primary question is whether ByteBPE512's previously observed sealed-test BPC advantage
survives comparison with a near-parameter-matched width-136 character model. Lower BPC is better;
the preregistered paired contrast is ByteBPE512 minus char136.

## Corpus Provenance

All three declared Project Gutenberg URLs were available. Each contained unique, ordered START
and END markers matching the declared title, so no substitution rule was used. The complete body
between those markers was normalized by the existing corpus preparer and split chronologically
80/10/10.

| Corpus | Ebook | Download SHA-256 | Extracted-body SHA-256 | Prepared SHA-256 |
| --- | ---: | --- | --- | --- |
| Frankenstein | 84 | `7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3` | `3581eea97498d1523c79cc9834b1933923c9022ee0fddb8aa5b6492997ae1665` | `3fef5c99657c90b1ba1ada7453d8800c5a7d2ebe9899b9ff01f0022dab682bd2` |
| Douglass | 23 | `e39d8d5e516947d578555d8e549206e44e07f44697143b6d808eec8d503729bf` | `a451fa67af4bb67a168fe647b4736b1285655042b8ab2318457389d80132a066` | `9bcbdbf5e503142493181d37e9834d7120d2641f4b1a9b15803e68244af264b8` |
| Origin | 1228 | `ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4` | `6e91d485042baba0158cee14fa2f14aff2fc45726e085228e50b929636dbe2ce` | `159b097edde209c0f67c91ebe8634e966b0c08e17dc65283c4ea79b3d9ba0ef8` |

| Corpus | Extracted characters | Prepared characters | Train | Validation | Sealed test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frankenstein | 419,341 | 419,189 | 335,351 | 41,919 | 41,919 |
| Douglass | 224,000 | 223,901 | 179,120 | 22,390 | 22,391 |
| Origin | 932,994 | 932,880 | 746,304 | 93,288 | 93,288 |

Runtime downloads, extracted text, prepared corpora, tokenizers, checkpoints, and run directories
remain ignored and are not release assets.

## Frozen Arms and Capacity Check

Every arm used four layers, four heads, dropout 0.1, AdamW at `1e-3`, zero weight decay, full
validation every 250 steps, patience 3, and a 5,000-step ceiling. Seeds were 1337, 2027, and 4242.

| Corpus | Character vocabulary | char128 parameters | char136 parameters | ByteBPE512 parameters | char136 vs ByteBPE512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frankenstein | 83 | 822,867 | 926,515 | 929,664 | -0.3387% |
| Douglass | 81 | 822,353 | 925,969 | 929,664 | -0.3975% |
| Origin | 86 | 823,638 | 927,334 | 929,664 | -0.2506% |

All matched-control gaps are inside the preregistered 1.5% tolerance. ByteBPE512 vocabulary is 512
for every corpus. Character vocabulary differs because tokenizers are fit on training text only.

## Pre-Access Baselines

| Corpus | Tokenizer | Uniform BPC | Unigram BPC | Bigram BPC |
| --- | --- | ---: | ---: | ---: |
| Frankenstein | character | 4.4188 | infinity | 2.4045 |
| Frankenstein | ByteBPE512 | 6.2383 | infinity | 3.3863 |
| Douglass | character | 4.3944 | 3.0626 | 2.4330 |
| Douglass | ByteBPE512 | 6.2383 | 4.2869 | 3.5817 |
| Origin | character | 4.4543 | 3.0129 | 2.4054 |
| Origin | ByteBPE512 | 6.2383 | 4.3278 | 3.1558 |

Frankenstein's unigram infinity is retained rather than hidden: at least one held-out symbol had
zero train-only unigram probability. The add-one bigram remains finite.

## Global Sealed-Test Audit

Before any terminal segment was tokenized or scored, a global audit found exactly 27 expected and
27 actual cells. All 27 had complete schema-2 summaries with `test_status: sealed_unread`; none
contained a test loss, BPC, coverage, support, or checkpoint score field; and no run contained a
`test_evaluation*.json` artifact. The audit initially failed closed because an overbroad diagnostic
also classified the pre-existing `test_characters` split-size field as a score. No evaluation ran
during that failed diagnostic. The corrected schema-aware audit passed with zero errors.

Each best checkpoint was then evaluated exactly once. All 27 artifacts report full coverage,
best-checkpoint identity, matching prepared-corpus checksums, and equal evaluated/total target-token
support. The structured record, including every checkpoint SHA-256 and support count, is
[`results/final_capacity_panel.json`](../results/final_capacity_panel.json).

## Results

### Cell Means

| Corpus | Arm | Mean best validation BPC | Mean sealed-test BPC | Mean test minus validation |
| --- | --- | ---: | ---: | ---: |
| Frankenstein | char128 | 2.0452 | 2.0887 | +0.0435 |
| Frankenstein | char136 | 2.0277 | 2.0783 | +0.0506 |
| Frankenstein | ByteBPE512 | **1.8703** | **1.9336** | +0.0633 |
| Douglass | char128 | **2.1234** | 2.4225 | +0.2991 |
| Douglass | char136 | 2.1266 | 2.4148 | +0.2882 |
| Douglass | ByteBPE512 | 2.0335 | **2.3960** | +0.3625 |
| Origin | char128 | 1.8588 | 2.1771 | +0.3183 |
| Origin | char136 | 1.8364 | 2.1489 | +0.3125 |
| Origin | ByteBPE512 | **1.6325** | **1.9772** | +0.3447 |

Bold marks the lowest value within each corpus and column. Every mean test-minus-validation gap is
positive: the three terminal regions are harder than their validation regions under every arm.

### Primary Contrast: ByteBPE512 Minus char136

| Corpus | Seed 1337 | Seed 2027 | Seed 4242 | Mean ± population SD | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frankenstein | -0.1394 | -0.1455 | -0.1493 | **-0.1447 ± 0.0041** | -0.1493 to -0.1394 |
| Douglass | -0.0270 | -0.0311 | +0.0018 | **-0.0187 ± 0.0147** | -0.0311 to +0.0018 |
| Origin | -0.1567 | -0.1846 | -0.1737 | **-0.1717 ± 0.0115** | -0.1846 to -0.1567 |

The preregistered directional hypothesis succeeds: the mean paired contrast is negative on every
corpus. The stronger outcome does not: ByteBPE512 wins eight of nine pairs, with a narrow reversal
on Douglass seed 4242 where char136 is better by 0.0018 BPC.

### Secondary Contrasts

| Contrast | Corpus | Mean ± population SD | Seed-pair wins for candidate |
| --- | --- | ---: | ---: |
| ByteBPE512 minus char128 | Frankenstein | -0.1551 ± 0.0094 | 3/3 |
| ByteBPE512 minus char128 | Douglass | -0.0265 ± 0.0062 | 3/3 |
| ByteBPE512 minus char128 | Origin | -0.1999 ± 0.0004 | 3/3 |
| char136 minus char128 | Frankenstein | -0.0104 ± 0.0103 | 2/3 |
| char136 minus char128 | Douglass | -0.0077 ± 0.0095 | 2/3 |
| char136 minus char128 | Origin | -0.0282 ± 0.0118 | 3/3 |

ByteBPE512 preserves all nine wins against the historical char128 arm. Added character-model
capacity helps on average in all three corpora and in seven of nine pairs, but it does not account
for the ByteBPE result. Douglass also shows why the result must remain corpus-qualified: its
capacity-controlled effect is small and includes the only primary reversal.

Douglass ByteBPE512 stopped early at 4,000, 3,500, and 3,750 actual steps with best steps 3,250,
2,750, and 3,000. Every other cell ran to 5,000 steps; Frankenstein ByteBPE seed 2027 and three
character cells selected step 4,750, while the remaining best checkpoints were at step 5,000.

## Interpretation and Project Conclusion

The final panel supports the narrow project claim: in these fixed tiny-model, single-document,
chronological-split experiments, boundary-aware ByteBPE512 generally reduces sealed-test bits per
character beyond what can be explained by its larger embedding/output parameter count. The result
is not universal. One of nine matched pairs reverses, corpus means vary from -0.0187 to -0.1717
BPC, and the large validation-to-test gaps show strong chronological distribution shift.

This fixed panel does not estimate a population effect, establish production tokenizer quality,
or justify choosing models from these consumed terminal regions. No result triggered a rerun,
replacement, configuration change, or additional corpus. Version `1.0.0` permanently completes the
declared project scope; see [Project Completion](../docs/project-completion.md).

## Reproduction Record

The frozen configs are `configs/gptiny_{frankenstein,douglass,origin}_{char128,char136,bytebpe512}_final_seed{1337,2027,4242}.yaml`.
The workflow used the repository scripts in this order:

```bash
python scripts/extract_gutenberg.py --input data/raw/<download>.txt --output data/raw/<body>.txt
python scripts/prepare_corpus.py --input data/raw/<body>.txt --output data/processed/<corpus>.txt \
  --train-split 0.8 --validation-split 0.1
python scripts/prepare_data.py --config configs/<final-config>.yaml
python scripts/evaluate_baselines.py --config configs/<final-config>.yaml
python scripts/train.py --config configs/<final-config>.yaml
```

After the global sealed audit only:

```bash
python scripts/evaluate_test.py --run runs/<name>/<id> --checkpoint-kind best
python scripts/summarize_capacity_panel.py --cell <corpus> <arm> <run-dir> ...
```

## Release Validation

- `uv run make check`: passed Ruff format/lint, strict mypy over 47 source files, all 182 tests,
  90.49% branch coverage, compileall, and Markdown link validation. The first run exposed one stale
  five-corpus chart assertion; after updating it to the published eight-corpus contract, the full
  command passed.
- `uv run make audit`: no known vulnerabilities; the local `smallm==1.0.0` package is correctly
  skipped because it is not published on PyPI.
- `uv run make demo`: passed corpus preparation, tokenizer fitting, baselines, five-step training,
  run inspection, and deterministic greedy generation on the committed synthetic corpus.
- `uv build`: built `smallm-1.0.0-py3-none-any.whl` and `smallm-1.0.0.tar.gz`; wheel metadata reports
  version 1.0.0.
- `python scripts/render_results_chart.py --check`, JSON parsing, `git diff --check`, and
  `sha256sum --check RELEASE_ASSETS.sha256`: passed.

The final aggregate data, chart, citation metadata, wheel, and source distribution checksums are
recorded in [`RELEASE_ASSETS.sha256`](../RELEASE_ASSETS.sha256).
