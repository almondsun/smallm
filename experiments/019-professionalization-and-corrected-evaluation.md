# 019 Professionalization and Corrected Evaluation

## Goal

Harden smaLLM's scientific, security, artifact, and project contracts; add an advanced theory
handbook; then rerun the deep character and BPE128 comparison without validation leakage or
coverage-mismatched metrics.

## Contract corrections

- Split characters before fitting either tokenizer; fit on training text only.
- Verify corpus checksum, counts, and split before creating a run.
- Evaluate deterministic non-overlapping target blocks with target-weighted NLL.
- Use full validation in official configs and compute BPC from exactly evaluated characters.
- Load checkpoints in weights-only mode and validate their schema.
- Write final artifacts atomically and exclude incomplete runs from `latest`.

These changes create a new metric contract. Experiments 016–017 are retained with errata rather
than silently rewritten.

## Professionalization

- Frozen `uv` environment; Ruff, strict mypy, pytest coverage, compile, and link gates.
- Python-version CI matrix, CodeQL, dependency review, and dependency automation.
- Contribution, security, conduct, changelog, issue, and pull-request guidance.
- A theory-dense [`notes/`](../notes/) handbook covering the complete system.

## Validation commands

```bash
uv sync --frozen --extra dev
make check
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/evaluate_baselines.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/show_run.py --run latest --run-name smoke
python scripts/generate.py --run latest --run-name smoke --prompt "Once" --greedy --max-new-tokens 20
```

Corrected modeling commands:

```bash
python scripts/train.py --config configs/gptiny_5k_lr1e-3_deep.yaml
python scripts/train.py --config configs/gptiny_bpe128_5k_lr1e-3_deep.yaml
```

## Results

Both corrected runs used corpus SHA-256
`a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`, full non-overlapping
validation, seed `1337`, and the same 5,000-step deep architecture and optimizer controls.

| tokenizer | train tokens | validation targets | final val loss | best val loss / step | final BPC | best BPC | duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| character | 130,077 | 14,452 | 1.4390 | 1.4390 / 5,000 | 2.0760 | 2.0760 | 412.65s |
| BPE128 | 86,368 | 9,542 | 2.2907 | 2.2020 / 2,500 | 2.1820 | 2.0976 | 408.60s |

The corrected BPE128 best checkpoint remains worse than the corrected character control by about
`0.0216` BPC, but the gap is far smaller than the superseded estimate (`2.4453` versus `2.1569`).
The earlier qualitative conclusion survives only narrowly: this BPE setup shortens sequences by
roughly one third without winning the character-normalized metric.

### Controlled generation

Greedy character final/best outputs are identical because the best checkpoint is the final step;
their distinct-2 is `0.5049`. BPE final greedy is locally coherent and more diverse
(`distinct-2=0.6577`), while BPE best greedy repeats “I wish” and “were” patterns
(`distinct-2=0.4333`, longest repeated run `4`).

With temperature `0.8`, top-k `10`, and seed `1337`, distinct-2 is `0.6408` for the character
final checkpoint, `0.6824` for BPE final, and `0.6687` for BPE best. These surface measures favor
BPE diversity in this probe, but all samples remain locally plausible and globally incoherent.

### Validation evidence

- `90 passed`; total coverage `90.53%`.
- Ruff formatting and linting passed.
- Strict mypy passed for 34 source files.
- `pip-audit` found no known vulnerabilities in third-party dependencies.
- The complete prepare-data, baselines, smoke-train, show-run, and generation path passed.
- Character corrected run: `runs/gptiny_5k_lr1e-3_deep/2026-07-12_19-45-00`.
- BPE128 corrected run: `runs/gptiny_bpe128_5k_lr1e-3_deep/2026-07-12_19-52-06`.

## Limitations

- The educational BPE is not a production tokenizer.
- One public-domain corpus and two configurations do not establish general scaling behavior.
- Seeded and greedy samples are controlled probes, not human-evaluation studies.
