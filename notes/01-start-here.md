# 01 — Start Here: What You Are About to Build in Your Head

This chapter assumes you have read the [minimum glossary](00-glossary.md), especially the entries
for corpus, token, vocabulary, context, parameter, loss, and checkpoint.

## The one-sentence model

smaLLM repeatedly answers one question:

> Given the tokens seen so far, which token is likely to come next?

Everything else exists to make that question computable, learnable, or scientifically honest.
Tokenization turns text into finite choices. A Transformer turns a prefix into scores. Softmax
turns scores into probabilities. Cross-entropy says how wrong those probabilities were.
Backpropagation assigns responsibility to weights. AdamW changes them. Evaluation asks whether the
improvement survives on text not used for fitting. Generation feeds a chosen prediction back into
the next prefix.

## What “learning” means here

The source code does not contain English grammar rules. At initialization, its parameter tensors
are mostly random numbers. Training presents many next-token questions and slightly changes those
numbers in directions that reduce average error. Patterns such as spelling, whitespace, quotation
structure, and recurring phrases can become useful because they help prediction.

A **parameter** is simply a number the optimizer may change. A model with 900,000 parameters has
900,000 adjustable numbers, not 900,000 hand-written facts. A **hyperparameter**—for example the
learning rate or number of layers—is chosen by the experimenter rather than fitted by gradients.

## Transformer, GPT, and LLM are not synonyms

- A **Transformer** is a neural-network architecture organized around attention, residual paths,
  normalization, and position-wise feed-forward layers.
- **GPT** means a Generative Pre-trained Transformer family: a causal, decoder-only Transformer
  trained to predict the next token. smaLLM implements this architectural core, although its
  training is better described as small-corpus pretraining.
- A **large language model** is a language model with substantial scale. smaLLM is intentionally
  tiny. It is a language-model laboratory, not “large” by modern standards.

Smallness is useful for learning: every tensor fits in one process, the architecture is visible,
and experiments can run on a CPU.

## The three kinds of state

Keep these separate from the beginning:

1. **Data state:** text, tokenizer vocabulary, token IDs, and input/target batches.
2. **Model state:** learned weights such as embeddings and linear projections.
3. **Optimizer state:** moving averages AdamW uses to decide updates.

A checkpoint stores model state and enough configuration/tokenizer identity to interpret it. It is
not the training corpus, and it is not a complete resumable training state in this repository.

## Run the smallest complete path

From the repository root:

```bash
make demo
```

The demo prepares a committed synthetic corpus, fits a character tokenizer, evaluates count-based
baselines, trains five optimizer steps, inspects the run, and generates text. Five steps validate
the machinery; they are not expected to learn good prose.

As you read its output, identify:

- vocabulary size `V`
- context length `T` (`block_size`)
- embedding width `C` (`n_embd`)
- number of layers and heads
- batch size `B`
- parameter count
- training and validation loss

These names will stop looking like configuration trivia once later chapters trace their effects.

## A useful debugging hierarchy

When something feels mysterious, descend in this order:

```text
word-level story
  -> tiny numerical example
  -> tensor values
  -> tensor shapes
  -> source-code operation
  -> test that would fail if the story were wrong
```

Do not use “the model understands” as a substitute for a mechanism. Ask which values changed,
which positions could communicate, what objective rewarded the behavior, and what evidence supports
the claim.

## Chapter checkpoint

Before continuing, explain in your own words:

1. Why is next-token prediction a finite classification problem only after tokenization?
2. What is the difference between a parameter and a hyperparameter?
3. Why can a tiny GPT teach the Transformer mechanism without being an LLM in the scale sense?
4. Which three kinds of state participate in training?

Next: [Computation and tensors](02-computation-and-tensors.md).
