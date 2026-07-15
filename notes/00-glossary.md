# 00 — Before You Begin: The Minimum Vocabulary

Read this page before chapter 01. These definitions are intentionally short: they give each word a
stable meaning so later chapters can refine the mathematics without introducing an undefined term.

## Text and data

- **Corpus:** the complete body of text chosen as data for an experiment. In smaLLM a corpus is an
  ordered Unicode string plus source metadata and checksums. It is not a bag of unrelated sentences.
- **Raw corpus:** source text before smaLLM's deterministic normalization.
- **Prepared corpus:** normalized text that is actually split, tokenized, and modeled.
- **Dataset:** an indexed collection of model examples derived from data. Here, the training dataset
  consists of overlapping fixed-length input/target token blocks derived from the training corpus.
- **Example:** one input sequence paired with its next-token target sequence.
- **Train split:** corpus region allowed to fit tokenizer rules and model parameters.
- **Validation split:** held-out region allowed to select checkpoints and configurations, but not to
  update parameters or fit tokenizer rules.
- **Test split:** terminal held-out region reserved for one-shot evaluation after the decision rule
  and checkpoint are frozen.
- **Leakage:** use of information outside its declared role—for example fitting BPE merges on
  validation text.

## Tokens and sequences

- **Token:** one discrete symbol predicted by the model. A token may be a character, byte, whitespace
  run, or multi-character subword; it is not necessarily a word.
- **Tokenizer:** a fitted mapping between text and token IDs, together with a decoding rule.
- **Token ID:** an integer name for one vocabulary entry. Its numeric magnitude has no semantic
  ordering.
- **Vocabulary:** the finite set of token types available to the model; its size is denoted by `V`.
- **Sequence:** an ordered list of tokens or token IDs.
- **Position:** an index within a sequence.
- **Prefix:** all tokens observed before or through a specified position.
- **Context:** the prefix made available to a prediction. smaLLM limits it to at most `block_size`
  tokens.
- **Block:** a fixed-length contiguous token subsequence used as a model input.
- **Batch:** several independent blocks processed together using the same model parameters.

## Model and learning

- **Model:** a parameterized mathematical function mapping token-ID sequences to next-token logits.
- **Parameter:** a scalar component of a model tensor that the optimizer may update.
- **Hyperparameter:** an experimenter-selected setting, such as width, layer count, learning rate,
  or step budget.
- **Embedding:** a learned lookup that maps a token or position ID to a vector of continuous features.
- **Activation:** an intermediate tensor produced during a forward computation.
- **Logit:** an unrestricted real-valued score for one possible next token, before softmax.
- **Probability distribution:** non-negative weights over mutually exclusive outcomes that sum to one.
- **Loss:** a scalar function measuring predictive error; smaLLM trains with mean next-token
  cross-entropy.
- **Gradient:** the vector of partial derivatives of loss with respect to model parameters.
- **Backpropagation:** efficient reverse-mode application of the chain rule to compute gradients.
- **Optimizer:** an update rule that uses gradients and optimizer state to change parameters.
- **Training step:** one forward pass, backward pass, and parameter update for one minibatch.
- **Checkpoint:** serialized model configuration, tokenizer state, learned weights, and step identity.

## Transformer-specific terms

- **Attention:** a differentiable weighted retrieval operation over token representations.
- **Query:** representation of what one destination position is looking for.
- **Key:** representation used to match an available source position against a query.
- **Value:** representation mixed into the output according to query-key weights.
- **Head:** one independently projected attention subspace; several heads operate in parallel.
- **Causal mask:** rule assigning zero attention probability to future source positions.
- **Residual connection:** addition of a sublayer's output to its input, preserving an identity path.
- **Layer normalization:** per-token feature normalization followed by learned scale and shift.
- **MLP:** the position-wise two-linear-layer nonlinear network inside each Transformer block.

## Evidence and outputs

- **Metric:** a declared numerical summary, such as validation loss or bits per character.
- **Baseline:** a simpler reference method used to determine whether the neural model adds value.
- **Generation:** repeated next-token selection in which each selected token becomes later context.
- **Provenance:** evidence identifying data, transformations, configuration, environment, and output.
- **Reproducibility:** ability to reconstruct and audit a result from recorded inputs and procedures;
  it does not guarantee bit-identical floating-point results on every platform.

When a later chapter needs a stricter mathematical definition, it will explicitly refine one of
these meanings. The complete lookup remains in [chapter 12](12-reference.md).

Next: [Start here](01-start-here.md).
