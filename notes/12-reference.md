# 12 — Reference: Notation, Glossary, Derivations, and Sources

## Reading the notation

- `x∈R^C` means `x` is a vector of `C` real numbers.
- `W∈R^{m×n}` means `W` is a matrix with `m` rows and `n` columns.
- $x_0,\ldots,x_{T-1}$ denotes a sequence of length $T$ under the zero-based convention.
- $x_{a:b}$ mirrors a half-open Python slice: positions $a,\ldots,b-1$.
- `x_{<t}` means positions strictly before `t`.
- `Σ_i` means sum the following expression over index `i`.
- `∂L/∂w` is the partial derivative of loss with respect to parameter `w`.
- `θ` conventionally denotes the collection of all learned parameters.
- `≈` means approximately equal; `=` states equality under the definitions.
- $[n]=\{0,\ldots,n-1\}$ is the zero-based finite index set used throughout.
- $\mathbf{1}[P]$ is `1` when proposition `P` is true and `0` otherwise.
- $\Delta^{V-1}$ is the probability simplex in $\mathbb R^V$.
- $\mathbb E[X]$ denotes expectation of random variable $X$.
- $\arg\min_x f(x)$ is the set of inputs attaining the minimum of `f`; likewise for `argmax`.

## Shape table

| Symbol | Meaning |
| --- | --- |
| $B$ | batch size |
| $T$ | current sequence length |
| $V$ | vocabulary size |
| $C$ | embedding width |
| $H$ | attention heads |
| $D=C/H$ | per-head width |
| $L$ | number of Transformer blocks |
| $P$ | total scalar model parameters |
| $N$ | token or example count, defined locally |
| $T_{\max}$ | maximum configured context length |
| $X\in\mathbb R^{B\times T\times C}$ | hidden states |
| $Q,K,V_a\in\mathbb R^{B\times H\times T\times D}$ | attention tensors |
| $Z\in\mathbb R^{B\times T\times V}$ | output logits |

## Worked weighted-loss example

Suppose validation blocks contain `128`, `128`, and `17` targets with mean NLL `1.5`, `1.7`, and
`2.0`. Total NLL is `128(1.5)+128(1.7)+17(2.0)=443.6`; mean NLL is `443.6/273=1.6249`, not the
unweighted `1.7333`. If those targets represent `400` source characters, BPC is
`443.6/(400 ln 2)=1.6000`.

## Glossary

- **Activation:** intermediate tensor produced by a model operation; also used for a nonlinear
  function such as GELU.
- **Autograd:** PyTorch's automatic differentiation system.
- **Backpropagation:** reverse application of the chain rule to compute parameter gradients.
- **Batch:** independent examples evaluated together with shared parameters.
- **Causal:** position `t` cannot depend on tokens after `t`.
- **Checkpoint:** serialized model configuration, weights, tokenizer state, and run metadata.
- **Context:** prefix supplied to a prediction, bounded by the configured context length.
- **Corpus:** ordered text selected as experiment data, together with source identity and provenance.
- **Cross-entropy:** expected negative log probability assigned to the target.
- **Dataset:** indexed collection of input/target examples derived from data.
- **Embedding:** learned lookup from categorical IDs to continuous feature vectors.
- **Gradient:** partial derivative of loss with respect to each adjustable parameter.
- **Held-out:** excluded from fitting, including learned preprocessing.
- **Hyperparameter:** experimenter-chosen setting such as width, learning rate, or step budget.
- **Leakage:** information from evaluation data influences fitting or selection improperly.
- **Logit:** unnormalized log-score before softmax.
- **Loss:** scalar objective whose local reduction guides parameter updates.
- **Nat:** information unit from a natural logarithm; one bit is `ln(2)` nats.
- **Parameter:** model number adjusted by the optimizer.
- **Perplexity:** exponential of mean token NLL.
- **Provenance:** evidence identifying inputs, transformations, environment, and outputs.
- **Residual connection:** identity path plus a learned transformation.
- **Softmax:** conversion from a vector of logits to positive probabilities summing to one.
- **Tensor:** multidimensional numeric array with a shape and data type.
- **Token:** vocabulary unit predicted by the model; not necessarily a character or word.
- **Tokenizer:** fitted encode/decode rules mapping between text and vocabulary IDs.
- **Vocabulary:** finite mapping between tokens and integer IDs.

## Primary references

- Goodfellow, Bengio, and Courville, [Deep Learning](https://www.deeplearningbook.org/), especially
  chapters 2–8 for linear algebra, probability, optimization, and neural networks.
- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/),
  especially the language-modeling and Transformer chapters.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018.
- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), 2016.
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.
- Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016.
- Hendrycks and Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), 2016.
- Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html), 2014.
- Dodge et al., [Show Your Work: Improved Reporting of Experimental Results](https://aclanthology.org/D19-1224/), 2019.

## Codebases consulted

These external implementations were studied or used for comparison. The commit column records the
local revision consulted during smaLLM's development; full clones remain local-only and ignored.

| Project | Repository | Consulted commit |
| --- | --- | --- |
| LLMs-from-scratch | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | `ff0b3d9` |
| llm.c | [karpathy/llm.c](https://github.com/karpathy/llm.c) | `f1e2ace` |
| minGPT | [karpathy/minGPT](https://github.com/karpathy/minGPT) | `37baab7` |
| nanoGPT | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | `3adf61e` |
| nn-zero-to-hero | [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | `73c3fcc` |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | `c96378c413` |

## Further navigation

Return to the [handbook index](README.md), inspect the
[architecture](../docs/architecture.md), or follow the chronological
[experiment index](../docs/experiments.md).
