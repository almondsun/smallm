# 020 BPE Context and Learning-Rate Study

## Goal

Test whether BPE128's narrow gap to the character control comes from comparing unequal
character-context lengths or from using a learning rate that reaches its optimum too early.

## Design

The corrected experiment-019 character control uses a 64-character block, batch 16, 5,000 steps,
and learning rate `1e-3`. BPE128 averages `130077 / 86368 = 1.506` training characters per token,
so a 42-token block covers about 63.3 training characters. Batch 24 gives `42 * 24 = 1008` token
positions per step, close to the control's `64 * 16 = 1024`.

| run | block | batch | positions/step | approximate character context | learning rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| character control (019) | 64 chars | 16 | 1,024 | 64 | `1e-3` |
| BPE control (019) | 64 tokens | 16 | 1,024 | 96.4 | `1e-3` |
| BPE ctx42 high LR | 42 tokens | 24 | 1,008 | 63.3 | `1e-3` |
| BPE ctx42 low LR | 42 tokens | 24 | 1,008 | 63.3 | `5e-4` |
| BPE ctx64 low LR | 64 tokens | 16 | 1,024 | 96.4 | `5e-4` |

All runs use the same training-only BPE128 tokenizer, corpus checksum
`a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`, architecture depth and
width, seed, full non-overlapping validation, and 5,000-step budget.

## Commands

```bash
python scripts/train.py --config configs/gptiny_bpe128_5k_lr1e-3_ctx42.yaml
python scripts/train.py --config configs/gptiny_bpe128_5k_lr5e-4_ctx42.yaml
python scripts/train.py --config configs/gptiny_bpe128_5k_lr5e-4_ctx64.yaml
```

Final and best checkpoints were evaluated with greedy decoding and with temperature `0.8`, top-k
`10`, seed `1337`, prompt `Once`, and 100 new tokens.

## Validation results

| configuration | best loss | best BPC | best step | final loss | final BPC | duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| character ctx64, `1e-3` (019) | 1.4390 | 2.0760 | 5,000 | 1.4390 | 2.0760 | 412.65s |
| BPE ctx64, `1e-3` (019) | **2.2020** | **2.0976** | 2,500 | 2.2907 | 2.1820 | 408.60s |
| BPE ctx42, `1e-3` | 2.2153 | 2.1101 | 3,000 | 2.3078 | 2.1983 | 435.89s |
| BPE ctx42, `5e-4` | 2.2094 | 2.1045 | 3,250 | 2.2194 | 2.1141 | 448.07s |
| BPE ctx64, `5e-4` | 2.2044 | 2.0998 | 4,750 | 2.2225 | 2.1170 | 622.15s |

The lower learning rate delayed the optimum and reduced final overfitting, but did not beat the
original BPE control. Matching character context also did not close the gap: it weakened best BPC
by `0.0126` at `1e-3` and by `0.0069` at `5e-4` relative to the BPE control.

## Controlled generation

| configuration/checkpoint | greedy distinct-2 | seeded distinct-2 | qualitative result |
| --- | ---: | ---: | --- |
| ctx42 `1e-3` final | 0.6115 | 0.6948 | Locally plausible dialogue; incoherent transitions. |
| ctx42 `1e-3` best | 0.5828 | **0.7467** | Highest seeded diversity; still semantically unstable. |
| ctx42 `5e-4` final | 0.6333 | 0.6531 | Repeated dialogue framing without long token runs. |
| ctx42 `5e-4` best | 0.6276 | 0.5890 | More phrase reuse; no qualitative gain from best loss. |
| ctx64 `5e-4` final | 0.4150 | 0.5769 | Greedy `you can` loops; lower diversity. |
| ctx64 `5e-4` best | 0.4564 | 0.7095 | Better seeded diversity than final, not better validation than control. |

Distinct-n remains a surface statistic: high values can reward incoherent novelty. No sample is
globally coherent enough to reverse the validation conclusion.

## Conclusion

The experiment does not identify a BPE128 setting that beats the corrected character control.
Longer BPE context is slightly better for held-out compression, while the matched context can
improve seeded diversity. Lowering learning rate mostly moves the best checkpoint later and makes
the final checkpoint less degraded. The remaining `0.0216` BPC character advantage is unlikely to
be explained by either context mismatch or this two-point learning-rate choice alone.

A useful next modeling study should change the tokenizer itself—vocabulary size, boundary policy,
or byte fallback—rather than continue fine-grained tuning around this BPE128 implementation.

## Run evidence

- `runs/gptiny_bpe128_5k_lr1e-3_ctx42/2026-07-12_20-48-14`
- `runs/gptiny_bpe128_5k_lr5e-4_ctx42/2026-07-12_20-55-36`
- `runs/gptiny_bpe128_5k_lr5e-4_ctx64/2026-07-12_21-03-09`
- Every run recorded `validation_coverage=1.0`, final/best checkpoints, schema-v2 artifacts, and the
  copied verified dataset manifest.

## Limitations

- One seed and one corpus cannot establish general tokenizer rankings.
- Positions per step approximate dominant compute but do not equalize every attention/MLP cost.
- The educational BPE lacks production word-boundary and byte-fallback behavior.
- Generation diagnostics are not a blinded human evaluation.
