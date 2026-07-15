# 11 — End-to-End Code Walkthrough: Trace One Batch Through smaLLM

This chapter is the bridge from understanding components to reading the repository as one system.
Keep the source files open and follow the numbered path. The thin scripts parse arguments; reusable
behavior lives under `src/smallm/`.

## 1. Configuration declares the experiment

Start with [`configs/demo.yaml`](../configs/demo.yaml). It declares paths and split policy, tokenizer
type and block size, model dimensions, optimizer settings, validation cadence, decoding controls,
and random seeds.

Two `block_size` fields exist because data and model contracts meet at that boundary. Configuration
validation requires them to match. The configured model vocabulary is a placeholder; training uses
the tokenizer-derived vocabulary size so embeddings and output logits agree with actual IDs.

## 2. The script delegates immediately

[`scripts/train.py`](../scripts/train.py) parses `--config`, calls `load_config`, and passes the
result to `smallm.training.train`. This thin edge keeps CLI concerns out of the training logic.

## 3. Training verifies data identity

[`train`](../src/smallm/training/trainer.py) seeds random sources, selects a device, loads normalized
text, reads its manifest, and verifies the file checksum and declared split. A filename alone would
not prove which bytes are being trained.

It then splits Unicode text into train, validation, and optional sealed test regions. The tokenizer
is fitted only on `train_text`. Training and validation are encoded; test text is left unencoded.

At this point:

```text
text: Python str
train_token_ids: list[int]
train_tokens: torch.long tensor shaped (N_train)
val_tokens: torch.long tensor shaped (N_val)
```

## 4. Dataset and DataLoader create batches

`TokenBlockDataset(train_tokens,T)` exposes stride-one shifted windows. `DataLoader` shuffles their
indices and groups examples. A full emitted pair has:

```text
x: (B,T), integer token IDs
y: (B,T), the same windows shifted one token left
```

Shuffling changes order, not alignment inside an example. Separate batch rows do not attend to one
another.

## 5. Model construction turns dimensions into parameters

Training creates `GPTConfig` with tokenizer vocabulary `V`, context `T`, layers `L`, heads `H`,
width `C`, and dropout. `GPT(config)` constructs embeddings, independent Transformer blocks, final
LayerNorm, and the language head. `.to(device)` moves parameters and registered buffers—including
the causal mask—to CPU or accelerator.

AdamW receives `model.parameters()`. The optimizer does not know about text, tokens, or attention;
it sees parameter tensors and their eventual gradients.

## 6. One forward pass

For a batch `(x,y)`:

```text
x (B,T)
  -> token embedding (B,T,C)
  +  position embedding (T,C), broadcast over B
  -> dropout (B,T,C)
  -> L Transformer blocks, each preserving (B,T,C)
  -> final LayerNorm (B,T,C)
  -> language head (B,T,V)
  -> flatten to (B×T,V)
  -> cross-entropy against y flattened to (B×T)
  -> scalar loss
```

Inside each attention branch:

```text
(B,T,C) -> Q,K,V (B,H,T,D)
-> masked scores (B,H,T,T)
-> weighted values (B,H,T,D)
-> concatenated/projected output (B,T,C)
```

Every token target contributes to the scalar mean, but the causal mask restricts its hidden state
to the legal prefix.

## 7. One backward pass and update

The loop reads the scalar for logging, clears old gradients with `set_to_none=True`, calls
`loss.backward()`, then `optimizer.step()`. Backpropagation reaches the language head, final norm,
all residual branches, attention/MLP projections, and the embedding rows used by the batch.

Weights unused by a particular path may receive no gradient; shared parameters accumulate
contributions from all batch positions that used them.

## 8. Validation changes observation, not parameters

At configured intervals, `evaluate_tokens` enters evaluation mode and disables gradient tracking.
It traverses deterministic non-overlapping blocks, multiplies each mean loss by its target count,
and returns total NLL, mean loss, exact token/character support, coverage, and BPC.

If this is the lowest observed validation loss, training saves `best_checkpoint.pt`. The evaluator
restores the model's prior training mode before updates resume.

## 9. Artifacts make the run inspectable

The run directory receives a config snapshot, copied dataset manifest, JSONL metrics, best/final
checkpoints, generated sample, and a summary written only after successful completion. Checkpoints
contain model configuration, tokenizer state, model weights, and step—not the raw corpus.

The summary distinguishes training loss, last validation loss, best validation loss/step, coverage,
parameter count, environment, stopping reason, generation controls, and dataset identity.

## 10. Generation closes the autoregressive loop

[`scripts/generate.py`](../scripts/generate.py) validates and loads a checkpoint, reconstructs the
tokenizer/model, loads weights, encodes the prompt, and calls [`generate`](../src/smallm/generation/sample.py).

Training predicts all aligned positions in parallel with true prefixes. Generation repeatedly uses
only final-position logits, selects one ID, appends it, and uses that enlarged model-produced prefix
on the next iteration.

## Debugging map

| Symptom | First boundary to inspect |
| --- | --- |
| token ID exceeds vocabulary | tokenizer/checkpoint identity |
| sequence length error | data/model `block_size` contract |
| `n_embd` not divisible by heads | `(C,H,D)` attention reshape |
| future token changes earlier output | causal mask |
| validation changes later dropout behavior | mode restoration |
| plausible samples but poor held-out loss | decoding is not likelihood evidence |
| incomparable tokenizer perplexities | use exact character-normalized BPC |
| run cannot be reproduced | config, manifest, environment, or seed provenance |

## Capstone exercises

1. Run `make demo`, then locate every displayed value in `config.yaml` or `summary.json`.
2. Put a temporary breakpoint in `GPT.forward` and print shapes for one demo batch.
3. Calculate the demo parameter count by hand and compare it with the logger.
4. Set attention dropout to zero in a local experiment and explain which behavior changes in train
   versus evaluation mode. Do not interpret five-step sample quality as evidence.
5. Explain why generation uses `logits[:, -1, :]` while training uses every `(B,T,V)` row.
6. Starting from one validation block, derive its contribution to total NLL and BPC.

## Graduation checkpoint

You understand smaLLM's implemented Transformer depth when you can reconstruct this walkthrough
without the page, derive each tensor shape, explain each information boundary, and name a focused
test that would detect a violation.

Next: [Reference](12-reference.md), or return to the [handbook index](README.md).
