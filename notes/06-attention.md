# 06 — Attention From Scratch: Learnable Retrieval Over a Prefix

## The problem attention solves

After embedding, each position has a vector describing its current features. To predict the next
token, a position may need earlier information: an opening quote, a name, a newline, or a local
spelling pattern. Fixed-size convolutions only see a predetermined neighborhood. Self-attention
lets every position build a weighted mixture of all **allowed** positions in its sequence.

Attention can be read as differentiable retrieval:

1. A **query** describes what the current position is looking for.
2. Each **key** describes what an available position offers for matching.
3. Query-key compatibility becomes a weight.
4. A weighted sum retrieves **values**, the information actually carried forward.

Queries, keys, and values are learned projections of the same hidden states. They are roles played
by vectors, not separate input data structures.

## Formal definition of causal self-attention

For one batch item and one head, let \(X\in\mathbb R^{T\times C}\) and parameters

\[
W_Q,W_K,W_V\in\mathbb R^{D\times C}.
\]

For position \(i\in[T]\), define

\[
q_i=W_Qx_i,\qquad k_i=W_Kx_i,\qquad v_i=W_Vx_i,
\]

where each result lies in \(\mathbb R^D\). Define the causal additive mask

\[
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
\]

Scores, normalized weights, and outputs are

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt D}+M_{ij},
\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{r=0}^{T-1}e^{s_{ir}}},
\qquad
y_i=\sum_{j=0}^{T-1}\alpha_{ij}v_j.
\]

With the convention \(e^{-\infty}=0\), this reduces to

\[
y_i=\sum_{j=0}^{i}\alpha_{ij}v_j,
\qquad
\alpha_{ij}\ge0,
\qquad
\sum_{j=0}^{i}\alpha_{ij}=1.
\]

Therefore \(y_i\) is a convex combination of the permitted value vectors. It lies in their convex
hull before the learned output projection.

This is the deterministic attention operator used at inference. During training, attention
dropout is applied to the normalized weights afterward; an individual dropped-out realization is
therefore not itself constrained to be a convex combination.

## One head, one sequence

Let hidden states be `X` with shape `(T,C)`. Learned matrices produce

\[
Q=XW_Q^\top,\qquad K=XW_K^\top,\qquad V=XW_V^\top.
\]

For one attention head of width `D`, all three have shape `(T,D)`. Scores are

\[
S=QK^\top,
\]

with shape `(T,T)`. Row `i`, column `j` is the dot product between query `i` and key `j`: how much
position `i` matches position `j`. Rows are destinations asking questions; columns are sources
being inspected.

This orientation is worth memorizing by meaning, not convention. After row-wise softmax, row `i`
must sum to one because it contains the mixture weights used to update position `i`.

## Tiny numerical retrieval

Suppose one query has scores `[2, 1, 0]` against three keys. Softmax gives approximately
`[0.665, 0.245, 0.090]`. If scalar values are `[10, 0, -5]`, the retrieved result is

```text
0.665×10 + 0.245×0 + 0.090×(-5) = 6.20
```

The output is not a selected token. It is a weighted combination of value features. Softmax makes
retrieval smooth, allowing gradients to tell the query/key projections how changing compatibility
would have changed the loss.

## Why divide by the square root of head width

If query and key components are roughly independent with variance one, their dot product sums `D`
products and has variance proportional to `D`. As `D` grows, logits become large, softmax becomes
nearly one-hot, and its gradients shrink for most entries.

Scaled dot-product attention uses

\[
S=\frac{QK^\top}{\sqrt D}.
\]

Dividing by `sqrt(D)` keeps score variance on a more stable scale across head widths. This is a
variance argument, not a normalization guarantee; learned distributions need not remain unit
variance.

Under the simplifying assumptions that \(q_r,k_r\) are independent across `r`, have mean zero,
variance one, and are independent of each other,

\[
\operatorname{Var}(q^\top k)
=\operatorname{Var}\!\left(\sum_{r=0}^{D-1}q_rk_r\right)
=\sum_{r=0}^{D-1}\operatorname{Var}(q_rk_r)=D.
\]

Thus \(\operatorname{Var}(q^\top k/\sqrt D)=1\). Dividing by `D` would instead shrink variance as
`1/D`; omitting scaling lets it grow as `D`.

## Causality is an information boundary

During parallel training, the tensor contains the whole block, including future targets. A
decoder-only language model must prevent position `i` from reading any key `j>i`. smaLLM creates a
lower-triangular Boolean mask:

```text
query\key  0  1  2  3
0          ✓  ✗  ✗  ✗
1          ✓  ✓  ✗  ✗
2          ✓  ✓  ✓  ✗
3          ✓  ✓  ✓  ✓
```

Forbidden scores become negative infinity before softmax. Their exponentials are zero, so their
weights are exactly zero. Masking after softmax would be wrong unless rows were renormalized, and
even then it is less direct.

Position `i` may attend to itself. Its hidden vector represents token `x_i`, and the aligned target
is `x_{i+1}`. Self-access therefore does not reveal the target.

**Causality proposition.** For fixed parameters and dropout disabled, output \(y_i\) is a function
only of \(x_0,\ldots,x_i\).

**Proof.** The query \(q_i\) depends only on \(x_i\). For every \(j>i\), the mask makes
\(\alpha_{ij}=0\). Every remaining key/value pair depends only on \(x_j\) for \(j\le i\). The
weighted sum therefore contains no function of \(x_j\) for \(j>i\). ∎

Because LayerNorm and the MLP act independently at each position and residual addition is
position-aligned, induction over Transformer blocks preserves this causal dependency property.

## Batch and multi-head shapes

smaLLM begins with hidden states:

```text
X: (B, T, C)
```

One linear layer computes all three projections:

```text
qkv(X): (B, T, 3C)
split:  Q, K, V each (B, T, C)
```

With `H` heads and `D=C/H`, each tensor is reshaped and transposed:

```text
(B, T, C)
-> view (B, T, H, D)
-> transpose (B, H, T, D)
```

Now batched matrix multiplication gives:

```text
Q @ K.transpose(-2, -1)
(B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
```

Softmax runs over the last axis—the source/key positions. Multiplying by values gives:

```text
attention_weights @ V
(B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
```

The code transposes back, calls `contiguous()`, and views heads together as `(B,T,C)`.

## Why multiple heads?

A single head produces one retrieval distribution per position. Multiple heads have separate slices
of projected features and can learn different compatibility patterns simultaneously: perhaps a
nearby-character pattern in one head and a longer structural relation in another.

This is capacity, not a promise of human-readable specialization. Heads are not assigned linguistic
jobs, and interpreting one attention map as a complete explanation of a prediction is unsafe.

After concatenation, `self.proj` mixes head outputs in the shared `C`-dimensional space. Without
that learned output projection, each head's feature slice would remain more isolated.

Formally, for heads \(h\in[H]\), let
\(Y^{(h)}\in\mathbb R^{T\times D}\) be each attention output and define feature-axis concatenation

\[
Y_{cat}=\operatorname{Concat}(Y^{(0)},\ldots,Y^{(H-1)})\in\mathbb R^{T\times HD}.
\]

Since smaLLM requires \(HD=C\), the output projection
\(W_O\in\mathbb R^{C\times C}\) gives

\[
\operatorname{MHA}(X)=Y_{cat}W_O^\top+b_O\in\mathbb R^{T\times C}.
\]

This codomain equality is what permits residual addition with the original \(X\).

## The exact smaLLM implementation

Read [`CausalSelfAttention.forward`](../src/smallm/model/attention.py) in this order:

| Source operation | Shape | Meaning |
| --- | --- | --- |
| `batch_size, seq_len, n_embd = x.shape` | `(B,T,C)` | Establish contracts |
| `self.qkv(x).split(n_embd, dim=2)` | three `(B,T,C)` | Learned Q/K/V projections |
| `view(...).transpose(1,2)` | `(B,H,T,D)` | Separate heads |
| `q @ k.transpose(-2,-1)` | `(B,H,T,T)` | All query-key scores |
| multiply by `head_size**-0.5` | unchanged | Scale scores |
| `masked_fill(..., -inf)` | unchanged | Remove future access |
| `softmax(..., dim=-1)` | unchanged | Normalize each query row |
| `att @ v` | `(B,H,T,D)` | Retrieve value mixtures |
| transpose + view | `(B,T,C)` | Concatenate heads |
| `self.proj` | `(B,T,C)` | Mix head outputs |

The registered mask has shape `(1,1,block_size,block_size)`. Its leading singleton axes broadcast
across batches and heads. Slicing to `:seq_len` supports any sequence no longer than the configured
context.

Attention dropout randomly removes some normalized links during training, while residual dropout
acts after output projection. Both are disabled in evaluation mode.

## Complexity and the context limit

The score tensor contains `B×H×T×T` entries. Time is approximately `O(BT²C)` and score memory
`O(BHT²)`. Doubling `T` roughly quadruples score work and memory. Token compression therefore
changes more than vocabulary: a fixed number of tokens may cover more source characters.

smaLLM recomputes the full cropped context at each generated token. It does not implement a KV
cache. That omission keeps the mechanism inspectable but makes generation less efficient than
production decoders.

## What attention is not

- It is not automatically long-term memory beyond `block_size`.
- It does not choose one source position unless softmax happens to become nearly one-hot.
- Its weights alone are not causal explanations of model behavior.
- It does not know token order without position information and causal structure.
- It does not mix separate batch examples.

## Prove it to yourself

The model-shape tests verify output shape and the causal mask; configuration tests separately
reject widths that are not divisible by the head count:

```bash
python -m pytest tests/test_model_shapes.py -q
```

Then inspect the mask directly:

```bash
python - <<'PY'
from smallm.model.attention import CausalSelfAttention

layer = CausalSelfAttention(n_embd=8, n_head=2, block_size=4, dropout=0.0)
print(layer.causal_mask[0, 0].int())
PY
```

## Chapter checkpoint

1. In retrieval language, distinguish query, key, and value.
2. Why does `QKᵀ` have one `T` axis for destinations and another for sources?
3. Why is the mask applied before softmax?
4. Derive every shape from `(B,T,C)` through scores and back.
5. Why scale by `sqrt(D)` rather than `D`?
6. What does multi-head attention permit, and what does it not guarantee?

Next: [The complete smaLLM Transformer](07-transformer.md).
