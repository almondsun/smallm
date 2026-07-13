# Datasets, Causal Attention, and GPT

## Shifted token blocks

For tokens \([x_0,\ldots,x_T]\), an input block is \([x_0,\ldots,x_{T-1}]\) and targets are
\([x_1,\ldots,x_T]\). Every target is the next token for its aligned prefix. Training uses many
stride-one blocks to expose varied contexts; validation uses non-overlapping blocks so duplicated
prefixes do not dominate the estimate.

## Embeddings and positions

Token embedding matrix \(E\in\mathbb R^{V\times C}\) and learned positional matrix
\(P\in\mathbb R^{T_{max}\times C}\) produce

\[
X_{b,t}=E[x_{b,t}]+P[t].
\]

Positions are learned parameters rather than an intrinsic notion of order. Inputs longer than
`block_size` have no defined position embedding and are rejected; generation crops its conditioning
window to the most recent block.

## Scaled dot-product attention

One projection produces queries, keys, and values:

\[
[Q,K,V]=XW_{qkv},\qquad W_{qkv}\in\mathbb R^{C\times3C}.
\]

After reshaping `(B,T,C)` to `(B,H,T,D)`, each head computes

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right),\qquad Y=AV.
\]

The causal mask has \(M_{ij}=0\) when \(j\le i\), otherwise \(-\infty\). Thus row `i` cannot
use a future key. Scaling by \(\sqrt D\) prevents dot-product variance from growing with head
width and pushing softmax into saturated regions. Concatenated heads return to `(B,T,C)` and a
learned output projection mixes their subspaces.

Attention time is \(O(BHT^2D)=O(BT^2C)\); the score tensor costs \(O(BHT^2)\) memory. This is why
token compression and context length materially alter compute.

## Pre-norm Transformer block

smaLLM uses

\[
X'=X+\operatorname{Attention}(\operatorname{LN}(X)),
\quad
Y=X'+\operatorname{MLP}(\operatorname{LN}(X')).
\]

Layer normalization standardizes each token's feature vector and learns affine scale/bias. The MLP
expands `C -> 4C`, applies GELU, projects `4C -> C`, and drops activations. Residual paths preserve
an identity route for information and gradients. Pre-norm generally stabilizes deep optimization
relative to placing normalization only after residual addition.

After repeated blocks, final layer normalization and a linear head map `(B,T,C)` to `(B,T,V)`.
The current implementation does not tie input embeddings to output weights.

## Parameter accounting

Ignoring biases and layer norms, embeddings contribute \(VC+T_{max}C\). Per block, attention
contributes approximately \(4C^2\), while the MLP contributes \(8C^2\); the MLP therefore owns
most block parameters. The language head adds \(CV\).

Implementation: [`attention.py`](../src/smallm/model/attention.py),
[`blocks.py`](../src/smallm/model/blocks.py), and [`gpt.py`](../src/smallm/model/gpt.py).

Checks: prove the triangular mask forbids all `j>i`; trace every reshape; explain why `C` must be
divisible by `H`.
