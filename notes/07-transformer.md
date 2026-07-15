# 07 — The Complete smaLLM Transformer

Attention is one component, not the whole model. GPT combines token and position embeddings,
repeated pre-normalized residual blocks, a final normalization, and a vocabulary projection. This
chapter traces that composition exactly as [`GPT`](../src/smallm/model/gpt.py) implements it.

## Decoder-only means causal generation

The original Transformer separated an encoder that read a source sequence from a decoder that
generated a target sequence. GPT uses only the decoder-style causal stack: every position may use
its prefix, and the output predicts the next token in the same sequence.

smaLLM does not implement cross-attention, an encoder, bidirectional attention, or masked-token
reconstruction. “Decoder-only” identifies this information-flow design even though the code does
not define a class named `Decoder`.

## Formal specification of the model

Fix vocabulary size $V$, maximum context $T_{\max}$, width $C$, heads $H$, head width
$D=C/H$, and layers $L$. For actual length $T\le T_{\max}$, input is
$X^{id}\in[V]^{B\times T}$. Learned token and position tables are

$$
E\in\mathbb R^{V\times C},\qquad P\in\mathbb R^{T_{\max}\times C}.
$$

The initial residual stream is

$$
R^{(0)}_{bti}=E_{X^{id}_{bt},i}+P_{t,i}.
$$

Ignoring dropout notation, block $\ell\in\{0,\ldots,L-1\}$ computes

$$
A^{(\ell)}=R^{(\ell)}+\mathrm{MHA}^{(\ell)}
\!(\mathrm{LN}^{(\ell)}_1(R^{(\ell)})),
$$

$$
R^{(\ell+1)}=A^{(\ell)}+\mathrm{MLP}^{(\ell)}
\!(\mathrm{LN}^{(\ell)}_2(A^{(\ell)})).
$$

For output parameters $W_U\in\mathbb R^{V\times C}$, $b_U\in\mathbb R^V$,

$$
Z_{bt}=W_U\mathrm{LN}_f(R^{(L)}_{bt})+b_U,
\qquad Z\in\mathbb R^{B\times T\times V}.
$$

The induced conditional distribution is

$$
p_\theta(y_{bt}=v\mid X^{id}_{b,0:t})
=\mathrm{softmax}(Z_{bt})_v.
$$

By chapter 05's causality proposition, this distribution is invariant to changes in input tokens
at positions strictly after `t`.

## Step 1: token embeddings

Input token IDs have shape `(B,T)`. The token embedding table has `(V,C)` learned parameters.
Indexing it produces `(B,T,C)`:

$$
X^{token}_{b,t}=E[x_{b,t}].
$$

The model now represents each categorical token using `C` continuous features. The same row is used
wherever that token ID occurs, before context modifies it.

## Step 2: learned position embeddings

Self-attention without position information is permutation-equivariant: reordering keys and values
reorders the computation but does not inherently encode “first” or “third.” smaLLM learns a
position table `(block_size,C)` and adds row `t` to every token at position `t`:

$$
X_{b,t}=E[x_{b,t}]+P[t].
$$

`positions = torch.arange(seq_len)` has shape `(T)`, so its embedding has `(T,C)` and broadcasts
across batch axis `B`. Positions are relative to the current cropped context, not absolute document
offsets. Sequences longer than `block_size` are rejected; generation retains only its newest
`block_size` tokens.

Learned positions do not automatically extrapolate beyond the configured table. That limitation is
one reason context length is a model contract.

## Step 3: embedding dropout

Dropout randomly zeroes features during training and rescales survivors by `1/(1-p)`, preserving
their expectation. This discourages brittle co-adaptation. It changes neither shape nor parameter
count and is disabled during evaluation/generation.

## Layer normalization

For one token vector `x∈R^C`, LayerNorm computes its feature mean and variance:

$$
\mu=\frac1C\sum_i x_i,\qquad
\sigma^2=\frac1C\sum_i(x_i-\mu)^2,
$$

then

$$
\mathrm{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

Each token is normalized independently across its `C` features. It does not average across batch or
time. Learned `γ` and `β` allow the network to restore or adjust scales useful downstream.

LayerNorm helps keep activation scales manageable, but it does not make all training stable or
remove the need for good initialization and learning rates.

## Residual connections

A residual sublayer returns

$$
y=x+F(x).
$$

The addition requires `F(x)` to have the same shape as `x`. An identity path lets information and
gradients bypass the learned transformation. The sublayer can learn a small correction rather than
reconstructing the whole representation.

smaLLM uses **pre-norm** blocks:

$$
X'=X+\mathrm{Attention}(\mathrm{LN}_1(X)),
$$

$$
Y=X'+\mathrm{MLP}(\mathrm{LN}_2(X')).
$$

Normalization occurs before each learned branch; the residual stream itself remains an explicit
running state. Pre-norm commonly improves optimization of deeper stacks relative to only
normalizing after residual addition.

## The position-wise MLP

Attention moves information between token positions. The MLP transforms features independently at
each position using shared parameters:

```text
(B,T,C)
-> Linear(C,4C)
-> GELU
-> Linear(4C,C)
-> Dropout
-> (B,T,C)
```

Expanding to `4C` provides a wider nonlinear workspace. Because the same MLP is applied at every
position, it does not by itself move information through time; attention has already placed
contextual information into each position's feature vector.

For token state $x\in\mathbb R^C$, the exact parameterized map is

$$
\mathrm{MLP}(x)=W_2\mathrm{GELU}(W_1x+b_1)+b_2,
$$

with $W_1\in\mathbb R^{4C\times C}$, $b_1\in\mathbb R^{4C}$,
$W_2\in\mathbb R^{C\times4C}$, and $b_2\in\mathbb R^C$. The same map is applied to every
`(b,t)` pair.

## One Transformer block

[`TransformerBlock.forward`](../src/smallm/model/blocks.py) is intentionally two lines:

```python
x = x + self.attn(self.ln_1(x))
return x + self.mlp(self.ln_2(x))
```

Those lines encode two different operations:

- attention: mix information across allowed positions, then add it to the residual stream;
- MLP: transform features within each position, then add that result.

Every block preserves `(B,T,C)`, allowing `n_layer` blocks to compose in a `ModuleList`. Layers
have identical architecture but independent parameters.

## Final normalization and language-model head

After all blocks, final LayerNorm produces `(B,T,C)`. A linear head maps each token representation
to `V` logits:

```text
(B,T,C) -> Linear(C,V) -> (B,T,V)
```

There is one distribution per batch element and input position. If targets are present, the model
flattens the first two axes and computes mean cross-entropy. If not, loss is `None`; generation
uses the final position's logits.

For targets $Y\in[V]^{B\times T}$, the implemented scalar objective is

$$
\mathcal L(\theta;X^{id},Y)
=-\frac1{BT}\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}
\log\mathrm{softmax}(Z_{bt})_{Y_{bt}}.
$$

Flattening `(B,T,V)` to `(BT,V)` is an index bijection and leaves this double sum unchanged.

The input token embedding and output head weights are **not tied** in smaLLM. Some GPT systems share
them, reducing parameters and imposing a relationship between input/output token geometry. This
repository leaves them independent for explicitness.

## End-to-end shape trace

For `B=4`, `T=32`, `C=128`, `H=4`, and `V=83`:

| Stage | Shape |
| --- | --- |
| token IDs | `(4,32)` |
| token embeddings | `(4,32,128)` |
| position embeddings | `(32,128)` |
| summed hidden states | `(4,32,128)` |
| Q/K/V before heads | three `(4,32,128)` |
| Q/K/V after heads (`D=32`) | three `(4,4,32,32)` |
| attention scores | `(4,4,32,32)` |
| each block output | `(4,32,128)` |
| logits | `(4,32,83)` |
| flattened logits for loss | `(128,83)` |
| flattened targets | `(128)` |
| loss | scalar |

Notice that the number `32` appears as both sequence length and head width in this example. Shapes
alone are insufficient unless each axis's meaning is annotated.

## Exact parameter accounting

Let maximum context be `T_max`, layers `L`, and assume linear biases and LayerNorm affine parameters
as in the implementation.

- token embedding: `VC`
- position embedding: `T_max C`
- attention per block: `4C² + 4C`
- two LayerNorms per block: `4C`
- MLP per block: `8C² + 5C`
- final LayerNorm: `2C`
- language head: `CV + V`

Total:

$$
2VC+T_{\max}C+L(12C^2+13C)+2C+V.
$$

For the demo (`V=35`, `T_max=32`, `C=32`, `L=1`), this gives `16,067`, matching the logged model.
Dropout and GELU add no learned parameters.

The quadratic block term is dominated by MLP (`8C²`) and attention projections (`4C²`). Vocabulary
size affects both input embeddings and output head, which motivated smaLLM's final capacity-matched
tokenizer experiment.

## What this GPT does not implement

- weight tying
- rotary or sinusoidal positions
- attention/key-value caching
- flash/fused attention
- padding masks or variable-length packed batches
- cross-attention
- mixture-of-experts layers
- distributed or mixed-precision execution

These omissions do not make the core causal Transformer incorrect; they define the educational
and experimental boundary.

## Chapter checkpoint

1. Why does attention need position information even with a causal mask?
2. Across which axis does LayerNorm operate here, and why?
3. Assign distinct jobs to attention, the MLP, residual connections, and normalization.
4. Trace `(B,T)` token IDs to scalar loss with every intermediate shape.
5. Derive the demo's `16,067` parameters.
6. What would weight tying change conceptually and numerically?

Next: [Training the model](08-training.md).
