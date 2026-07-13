# 017 Best-Checkpoint Evaluation and Early-Overfit Diagnostics

> **Erratum (milestone 019):** checkpoint mechanics and controlled generation observations remain
> useful, but tokenizer fitting included validation text and checkpoint selection used an
> overlapping prefix estimator. BPC used mismatched coverage. Corrected milestone-019 runs
> supersede these headline validation and cross-tokenizer comparisons.

## Goal

Save the model at its best validation point and compare controlled generation
from the final and best-validation checkpoints. The experiment asks whether the
early validation peak observed for BPE128 in experiment 016 also marks better
generated text.

## Motivation

Experiment 016 found that BPE128 reached validation loss `2.5727` at step 1250,
then ended at `2.8109` while train loss continued down to `1.4121`. Generation
was evaluated only from the final checkpoint because training did not preserve
the earlier model. That made the validation diagnosis sound but left the sample
quality comparison incomplete.

## Implementation Changes

- Training now writes `best_checkpoint.pt` whenever validation reaches a new
  minimum, while preserving the existing final `checkpoint.pt` behavior.
- Best and final checkpoints contain the model state, model config, tokenizer
  state and path, run path, and checkpoint step.
- `summary.json` now records `best_checkpoint_path` and
  `best_checkpoint_exists` alongside the existing validation fields.
- `scripts/generate.py` accepts `--checkpoint-kind final|best` for run-based
  generation. It defaults to `final`, rejects use with an explicit
  `--checkpoint`, and gives an actionable error when an older run has no best
  checkpoint.
- `scripts/show_run.py` displays the best checkpoint when present, plus best
  validation loss and step and final validation loss.
- The run utility resolver validates checkpoint kinds and preserves support for
  existing final-only runs.

## Setup

- Corpus: Project Gutenberg ebook #11, *Alice's Adventures in Wonderland* by
  Lewis Carroll, with Gutenberg boilerplate removed.
- Prepared corpus: 144,530 characters; SHA-256
  `a4c81ef23eb99f8b14be2474be0410b708cc99293ce6d88cf6799335926639b9`.
- Split: 130,077 train characters and 14,453 validation characters.
- Prompt: `Once`; 200 new tokens.
- Decoding: greedy, and seeded top-k with temperature `0.8`, top-k `10`, and
  seed `1337`.
- Rerun configs: `configs/gptiny_bpe128_5k_lr1e-3_deep.yaml` and
  `configs/gptiny_5k_lr1e-3_deep.yaml`.
- Both BPE128 and the character deep control were rerun through the new
  checkpoint path. No BPE256 training or BPE tuning was performed.

## Training Results

| tokenizer | run name | run path | final checkpoint | best checkpoint | final val loss | best val loss | best step | final bits/char | best bits/char | final train loss | duration | tokens/sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE128 | `gptiny_bpe128_5k_lr1e-3_deep` | `runs/gptiny_bpe128_5k_lr1e-3_deep/2026-06-27_22-06-14` | `runs/gptiny_bpe128_5k_lr1e-3_deep/2026-06-27_22-06-14/checkpoint.pt` | `runs/gptiny_bpe128_5k_lr1e-3_deep/2026-06-27_22-06-14/best_checkpoint.pt` | 2.8109 | 2.5727 | 1250 | 2.6717 | 2.4453 | 1.4121 | 445.36s | 11,502 |
| char | `gptiny_5k_lr1e-3_deep` | `runs/gptiny_5k_lr1e-3_deep/2026-06-27_22-14-37` | `runs/gptiny_5k_lr1e-3_deep/2026-06-27_22-14-37/checkpoint.pt` | `runs/gptiny_5k_lr1e-3_deep/2026-06-27_22-14-37/best_checkpoint.pt` | 1.6447 | 1.4950 | 2500 | 2.3728 | 2.1569 | 1.2805 | 438.36s | 11,687 |

The reruns reproduced the losses and best steps reported in experiment 016.
Loading each `best_checkpoint.pt` confirmed that its embedded step equals the
corresponding `best_val_step`.

## Final-vs-Best Generation Diagnostics

| tokenizer | checkpoint | decoding | repetition rate | distinct-1 | distinct-2 | longest repeated token run | short excerpt | qualitative note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| BPE128 | final | greedy | 0.0131 | 0.1176 | 0.4918 | 2 | `It didn’t know what are _you_ myself` | Word-like locally, then repeats `had never`. |
| BPE128 | best | greedy | 0.0293 | 0.0906 | 0.3079 | 1 | `“Then you know,” said Alice` | Strong repeated dialogue templates and `I could you know` loops. |
| BPE128 | final | seeded top-k | 0.0171 | 0.1224 | 0.5222 | 1 | `Why, I don’t have done when I heard` | Incoherent, but more varied than the best-checkpoint sample. |
| BPE128 | best | seeded top-k | 0.0225 | 0.1186 | 0.4695 | 1 | `I’ve beginning_ you know mind you know` | More `you know` phrase reuse; no quality gain. |
| char | final | greedy | 0.0099 | 0.1324 | 0.4039 | 2 | `the whiting the same to say things` | Repetitive but retains more character n-gram diversity. |
| char | best | greedy | 0.0148 | 0.1127 | 0.2611 | 1 | `a long of the same to say to size` | Clear phrase loop; worse diversity than final. |
| char | final | seeded top-k | 0.0345 | 0.1863 | 0.5862 | 1 | `Once, by the Dormouse.` | Best diversity scores in this comparison, though still incoherent. |
| char | best | seeded top-k | 0.0099 | 0.1471 | 0.5123 | 1 | `but what a starts. But I can’t talking` | Lower adjacent repetition but less diverse and not more coherent. |

`longest_repeated_token_run` did not expose the main failure mode: the models
repeat multi-token phrases rather than identical whitespace-delimited tokens.
The excerpts and distinct-n scores are therefore necessary context.

## Answers

### Did the best checkpoint improve generation?

No. For BPE128, the best checkpoint lowered greedy distinct-2 from `0.4918` to
`0.3079` and seeded distinct-2 from `0.5222` to `0.4695`; both samples showed
more obvious dialogue or `you know` phrase reuse. The character best checkpoint
also lost diversity in both decoding modes. Better validation loss did not
select better samples under these two controlled probes.

### Did the best checkpoint reduce phrase-level repetition?

No. Phrase loops remained and were qualitatively stronger in both greedy
best-checkpoint samples. The identical-token run metric sometimes improved,
but it does not measure repeated phrases and should not be interpreted as a
phrase-repetition improvement.

### Was the final-vs-best difference larger for BPE128?

Not consistently. BPE128 had the larger greedy distinct-2 drop (`0.1839`
versus `0.1428`), while the character model had the larger seeded distinct-2
drop (`0.0739` versus `0.0527`). Two samples per checkpoint are insufficient
to claim a tokenizer-specific checkpoint effect.

### Did BPE128 still underperform the character control?

Yes. At their best validation points, estimated bits per character remained
`2.4453` for BPE128 versus `2.1569` for the character control. Character final
and best samples also had higher distinct-1 and distinct-2 in both decoding
modes. BPE128 retained some locally word-like chunks but did not become more
coherent.

### Did this change the experiment 016 conclusion?

No. BPE128 still shortens sequences but underperforms the character control on
estimated bits per character and does not solve phrase-level repetition.
Evaluating the best checkpoint removes an evaluation ambiguity without
reversing the result.

### What is the next technical direction?

The checkpoint tooling is sufficient for the next controlled study. The next
technical milestone should tune BPE training and tokenization together, using
both final and best checkpoints, rather than add more general evaluation
infrastructure. This milestone intentionally did not begin that sweep.

## Validation

Commands run:

```bash
python -m pytest
python -m compileall src scripts
python scripts/train.py --config configs/smoke.yaml
python scripts/show_run.py --run latest --run-name smoke
python scripts/generate.py --run latest --run-name smoke --checkpoint-kind final --prompt Once --greedy --max-new-tokens 5 --diagnostics
python scripts/generate.py --run latest --run-name smoke --checkpoint-kind best --prompt Once --greedy --max-new-tokens 5 --diagnostics
python scripts/train.py --config configs/gptiny_bpe128_5k_lr1e-3_deep.yaml
python scripts/train.py --config configs/gptiny_5k_lr1e-3_deep.yaml
python scripts/show_run.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep
python scripts/show_run.py --run latest --run-name gptiny_5k_lr1e-3_deep
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --checkpoint-kind final --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --checkpoint-kind best --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --checkpoint-kind final --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_bpe128_5k_lr1e-3_deep --checkpoint-kind best --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --checkpoint-kind final --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --checkpoint-kind best --prompt "Once" --greedy --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --checkpoint-kind final --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
python scripts/generate.py --run latest --run-name gptiny_5k_lr1e-3_deep --checkpoint-kind best --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 200 --diagnostics
```

Observed results:

- Focused checkpoint, run, and generation tests: 16 passed.
- Full test suite: 57 passed.
- `python -m compileall src scripts`: passed.
- Smoke training created final and best checkpoints; both generation paths
  loaded successfully.
- Both 5k reruns completed and reproduced the prior losses and best steps.
- The Markdown link/path check from `docs/codex/build-and-test.md`: passed.

## Limitations

- The corpus is one small public-domain book with a chronological split.
- Training is short and local; conclusions may not transfer to larger runs.
- Best validation loss may not align with best perceived sample quality.
- Generation diagnostics are simple proxies and do not directly detect phrase
  loops or semantic coherence.
- The comparison uses one prompt and one seed per stochastic checkpoint sample.
- Bits per character remains an estimate derived from token loss and
  token/character counts.
