# 03 — Probability and Language Modeling

## From uncertainty to numbers

A probability describes uncertainty using a number from 0 to 1. For mutually exclusive outcomes,
the probabilities sum to 1. If a tokenizer vocabulary is `['a', 'b', ' ']`, a next-token
distribution might be:

```text
P('a') = 0.60, P('b') = 0.10, P(' ') = 0.30
```

This does not say what will happen; it says how strongly the model distributes belief before the
answer is revealed. A categorical distribution is appropriate because the next token must be one
member of a finite vocabulary.

Conditional probability adds context. `P('u' | 'q')` means the probability of `u` given that the
observed prefix ends in `q`. A language model estimates a new distribution for every prefix.

### Formal probability model

Let $\mathcal V=\{0,\ldots,V-1\}$ be the finite token vocabulary. A token sequence of length $T$
is an element of the Cartesian product $\mathcal V^T$. On an underlying probability space
$(\Omega,\mathcal F,\mathbb P)$, define discrete random variables
$X_t:\Omega\to\mathcal V$. For token values $x_0,\ldots,x_t$, conditional probability is

$$
\mathbb P(X_t=x_t\mid X_{<t}=x_{<t})
=\frac{\mathbb P(X_{\le t}=x_{\le t})}
{\mathbb P(X_{<t}=x_{<t})},
$$

provided the denominator is nonzero. The unknown data-generating distribution is not available to
the program; the model supplies a parameterized approximation
$p_\theta(\cdot\mid x_{<t})\in\Delta^{V-1}$, where

$$
\Delta^{V-1}=\left\{p\in\mathbb R^V:p_i\ge0,\ \sum_{i=0}^{V-1}p_i=1\right\}
$$

is the probability simplex.

## One sequence becomes many prediction problems

For tokens $x_0,x_1,\ldots,x_{T-1}$, the probability chain rule says

$$
p(x_0,\ldots,x_{T-1})=p(x_0)\prod_{t=1}^{T-1}p(x_t\mid x_0,\ldots,x_{t-1}).
$$

The notation `x_<t` means every token before position `t`. This identity turns sequence modeling
into repeated next-token classification. Given `c a t`, training questions can include:

| supplied prefix | target |
| --- | --- |
| `c` | `a` |
| `c a` | `t` |

smaLLM evaluates all aligned positions of a block in parallel. Causality ensures the representation
at each position sees only its permitted prefix even though the GPU/CPU processes the full block at
once.

The chain rule follows by repeatedly rearranging the conditional-probability definition:

$$
p(x_0,\ldots,x_{T-1})
=p(x_0)\prod_{t=1}^{T-1}
\frac{p(x_0,\ldots,x_t)}{p(x_0,\ldots,x_{t-1})}.
$$

All intermediate joint probabilities telescope, leaving the joint probability on the left.

Given training sequences $x^{(1)},\ldots,x^{(N)}$, maximum-likelihood estimation chooses

$$
\hat\theta\in\arg\max_\theta
\sum_{n=1}^N\sum_t
\log p_\theta\!\left(x_t^{(n)}\mid x_{<t}^{(n)}\right).
$$

Equivalently, smaLLM minimizes the empirical negative log-likelihood

$$
\widehat{\mathcal L}(\theta)
=-\frac1M\sum_{n,t}
\log p_\theta\!\left(x_t^{(n)}\mid x_{<t}^{(n)}\right),
$$

where $M$ is the number of target positions included in the minibatch or evaluation region.

## Why multiply probabilities—and why logs appear

The chain rule multiplies conditional probabilities. Long sequences therefore produce extremely
small numbers. Logarithms turn multiplication into addition:

$$
\log(ab)=\log a+\log b.
$$

Because probabilities are at most 1, their logs are zero or negative. **Negative log-likelihood**
(NLL) flips the sign so better predictions approach zero:

$$
S=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

If the model gives the true token probability `0.8`, its NLL is `-log(0.8)≈0.223`. Probability
`0.1` costs `2.303`. Being confidently wrong is punished strongly.

The training loss is mean NLL across the batch and positions. **Teacher forcing** supplies the true
prefix at every training position. During generation, prefixes contain model choices, so an early
mistake changes later conditions. This difference is often called exposure mismatch.

## Logits: scores before probabilities

The network outputs one unrestricted real number per vocabulary item. These are **logits**, not
probabilities. For a batch of `B` sequences and `T` positions, logits have shape `(B,T,V)`.

Softmax converts a vector of logits `z` into probabilities:

$$
p(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Exponentiation makes every value positive; division normalizes the sum to one. Adding the same
constant to every logit changes no probability, because the common exponential factor cancels.
Stable implementations exploit this by subtracting the maximum before exponentiating.

Worked example for logits `[2, 1, 0]`:

```text
exp(logits) ≈ [7.389, 2.718, 1.000]
sum         ≈ 11.107
softmax     ≈ [0.665, 0.245, 0.090]
```

Softmax preserves ordering but expresses relative gaps. Logits are convenient because the model
can emit any real values and because cross-entropy has a useful gradient.

Softmax is invariant to a common shift $c\in\mathbb R$:

$$
\operatorname{softmax}(z+c\mathbf1)_i
=\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
=\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}
=\operatorname{softmax}(z)_i.
$$

Subtracting $\max_i z_i$ therefore preserves the distribution while ensuring the largest
exponent is $e^0=1$, reducing overflow risk.

## Cross-entropy: the training signal

For one true class `y`, categorical cross-entropy is

$$
\ell=-\log p(y).
$$

PyTorch's `F.cross_entropy` combines stable log-softmax and target lookup; smaLLM should not apply
softmax before calling it. The gradient with respect to logit `i` is

$$
\frac{\partial\ell}{\partial z_i}=p(i)-\mathbf 1[i=y].
$$

For every wrong class, the gradient is its predicted probability, pushing that logit down under
gradient descent. For the correct class it is `p(y)-1`, pushing the logit up. Confident wrong
classes receive larger corrections.

To derive the gradient, write

$$
\ell(z,y)=-z_y+\log\sum_j e^{z_j}.
$$

Then

$$
\frac{\partial\ell}{\partial z_i}
=-\mathbf1[i=y]+\frac{e^{z_i}}{\sum_j e^{z_j}}
=p(i)-\mathbf1[i=y].
$$

The gradient components sum to zero, consistent with softmax's invariance to common logit shifts.

In [`GPT.forward`](../src/smallm/model/gpt.py), logits `(B,T,V)` are flattened to `(B×T,V)` and
targets `(B,T)` to `(B×T)`. This does not mix examples; it presents `B×T` classification rows to
one vectorized loss function.

## What cross-entropy does and does not mean

If the unknown true distribution is `q` and the model distribution is `p`, expected cross-entropy
decomposes as

$$
H(q,p)=H(q)+D_{KL}(q\Vert p).
$$

`H(q)` is uncertainty intrinsic to the data distribution. KL divergence is non-negative mismatch
between model and truth. Fitting can reduce mismatch but cannot remove inherent ambiguity.

For discrete distributions on the same support,

$$
H(q)=-\sum_i q_i\log q_i,
\qquad
D_{KL}(q\Vert p)=\sum_i q_i\log\frac{q_i}{p_i}.
$$

Substitution gives $H(q,p)=-\sum_i q_i\log p_i=H(q)+D_{KL}(q\Vert p)$. Gibbs' inequality
implies $D_{KL}\ge0$, with equality exactly when $p=q$ on the support of $q$.

Lower held-out cross-entropy means the model assigned better probability to that sample under the
declared coverage. It does not establish factuality, reasoning, coherence, fairness, or usefulness.

## Perplexity and bits per character

Per-token perplexity is

$$
\operatorname{PPL}=e^{\mathcal L}.
$$

Loss `2.0` nats gives perplexity `7.39`, interpretable as a geometric-mean effective branching
factor. But tokenizers define different units: predicting one character and predicting one
multi-character BPE token are not the same event. Their token losses and perplexities cannot be
compared directly.

For total NLL `S` over `C_s` represented source characters,

$$
\operatorname{BPC}=\frac{S}{C_s\ln 2}.
$$

This converts natural-log information into bits and normalizes by the same source unit. `693.147`
nats over `1,000` represented characters is `1.0` bit per character. Numerator and denominator
must cover identical targets; a prefix loss multiplied by a whole-corpus count is not exact BPC.

## Tiny manual check

If the correct-token probabilities at three positions are `[0.5, 0.25, 0.5]`:

```text
total NLL = -log(0.5) - log(0.25) - log(0.5)
          = 0.693 + 1.386 + 0.693
          = 2.772 nats
mean loss = 2.772 / 3 = 0.924 nats/token
perplexity = exp(0.924) ≈ 2.52
```

If those targets represent four source characters, BPC is `2.772 / (4 ln 2) = 1.0`.

## Common wrong turns

- Treating logits as probabilities.
- Applying softmax before `F.cross_entropy`.
- Averaging batch means equally when target counts differ.
- Comparing perplexity across different tokenizers.
- Calling lower likelihood “understanding” without specifying behavior.
- Selecting a checkpoint on test data and then reporting that test score as untouched.

## Chapter checkpoint

1. Turn a four-token sequence into its aligned next-token questions.
2. Why is NLL additive across positions while sequence probability multiplies?
3. What are the shape and meaning of `(B,T,V)` logits?
4. Explain the cross-entropy logit gradient without using the phrase “backprop does it.”
5. Why is BPC sometimes comparable when perplexity is not?

Next: [From text to training examples](04-data-tokenization-datasets.md).
