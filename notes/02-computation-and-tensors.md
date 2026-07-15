# 02 — Computation and Tensors: The Math You Need, Built From Zero

## Numbers arranged with purpose

A **scalar** is one number, such as a loss of `2.1`. A **vector** is an ordered list of numbers,
such as `[0.2, -0.7, 1.4]`. A **matrix** is a rectangular table. A **tensor** is the general name
for an array with any number of axes. Scalars, vectors, and matrices are 0-, 1-, and 2-dimensional
tensors.

## Formal objects and conventions

Let $\mathbb N=\{0,1,2,\ldots\}$ and let $\mathbb R$ denote the real numbers. Define the finite
index set $[n]=\{0,1,\ldots,n-1\}$. A real tensor of shape
$(d_1,\ldots,d_k)$ is formally a function

$$
X:[d_1]\times\cdots\times[d_k]\to\mathbb R.
$$

Thus a vector $x\in\mathbb R^n$ assigns one real number $x_i$ to each $i\in[n]$; a matrix
$A\in\mathbb R^{m\times n}$ assigns $A_{ij}$ to every row-column pair. The number of scalar
entries is

$$
\operatorname{numel}(X)=\prod_{r=1}^k d_r.
$$

A reshape is admissible only when this product is unchanged. It changes the index factorization,
not the ordered scalar storage. A transpose composes the indexing function with an axis
permutation. These definitions explain why reshape/transpose do not introduce learned parameters.

This handbook uses zero-based indices to match Python, even when conventional mathematical texts
use indices from one. Vectors are conceptually column vectors unless a transpose is shown.

For $x,y\in\mathbb R^n$, define the Euclidean inner product and norm:

$$
\langle x,y\rangle=x^\top y=\sum_{i=0}^{n-1}x_i y_i,
\qquad
\lVert x\rVert_2=\sqrt{\langle x,x\rangle}.
$$

The inner product combines alignment and magnitude. Cosine similarity divides by both norms;
attention uses the unnormalized inner product and learns the surrounding scales.

“Dimension” is overloaded. A matrix has two **axes**, while an embedding may have 128 **features**.
This handbook says “axis” for the former and uses **shape** to remove ambiguity.

```python
import torch

x = torch.tensor([[10, 11, 12], [20, 21, 22]])
print(x.shape)  # torch.Size([2, 3])
```

The shape `(2, 3)` means two rows and three columns. `x[1, 2]` is `22` because Python indices begin
at zero.

## Axes in a language model

Suppose two sequences each contain four token IDs:

```text
[[5, 2, 9, 1],
 [3, 3, 8, 4]]
```

This has shape `(B=2, T=4)`. After embedding each ID as six learned features, shape becomes
`(B=2, T=4, C=6)`. The three axes answer three different questions:

- Which sequence in the batch?
- Which position in that sequence?
- Which learned feature at that position?

Shapes are contracts. Adding tensors element-by-element requires compatible shapes. Matrix
multiplication requires matching inner dimensions. Most Transformer implementation errors are
violations of a shape contract disguised as complicated math.

## Element-wise operations and broadcasting

Adding `[1, 2, 3] + [10, 20, 30]` gives `[11, 22, 33]`. Multiplying them element by element gives
`[10, 40, 90]`. This is not a dot product.

**Broadcasting** lets PyTorch reuse a smaller tensor across missing or size-one axes. Token
embeddings have shape `(B,T,C)`, while position embeddings have `(T,C)`. Adding them broadcasts
the same position table across all `B` sequences. No batch-specific copy is learned.

Broadcasting is convenient but should be explainable. When two shapes are aligned from the right,
each pair of axes must be equal or one must be `1` (with absent leading axes treated as `1`).

## Dot products: a learned compatibility score

For equal-length vectors,

$$
a\cdot b=\sum_i a_i b_i.
$$

For `a=[1,2]` and `b=[3,4]`, the dot product is `1×3 + 2×4 = 11`. It multiplies aligned features
and sums them into one number. If learned query and key vectors align in useful directions, their
dot product is large. Attention uses exactly this idea as a compatibility score.

The dot product depends on both direction and magnitude. Chapter 05 explains why attention scales
it before softmax.

## Matrix multiplication: many dot products at once

If `A` has shape `(m,n)` and `B` has `(n,p)`, then `A @ B` has `(m,p)`. Entry `(i,j)` is the dot
product of row `i` from `A` and column `j` from `B`.

```text
[[1, 2],       [[5],       [[1×5 + 2×7],       [[19],
 [3, 4]]   @    [7]]   =    [3×5 + 4×7]]   =    [43]]
```

A linear layer applies `y = xWᵀ + b`. If `x` ends in `C_in` features and the layer produces
`C_out`, PyTorch stores a weight matrix shaped `(C_out, C_in)`. All leading axes are preserved.
Thus `nn.Linear(C, 3*C)` maps `(B,T,C)` to `(B,T,3C)` independently at every batch/position pair.

Formally, for $A\in\mathbb R^{m\times n}$ and
$B\in\mathbb R^{n\times p}$, their product $C=AB\in\mathbb R^{m\times p}$ is defined by

$$
C_{ij}=\sum_{r=0}^{n-1}A_{ir}B_{rj}.
$$

The repeated inner index $r$ is contracted; the uncontracted indices $i,j$ remain as output
axes. Batched multiplication keeps all leading batch axes and performs this contraction on the
final two axes. For attention,

$$
S_{bhij}=\sum_{r=0}^{D-1}Q_{bhir}K_{bhjr},
$$

so $Q,K\in\mathbb R^{B\times H\times T\times D}$ produce
$S\in\mathbb R^{B\times H\times T\times T}$. Writing indices makes the two different `T` axes
unambiguous.

An affine map $f(x)=Wx+b$ is not strictly linear when $b\ne0$, because then $f(0)=b$.
Machine-learning code conventionally calls it a “linear layer.” Its parameters are
$W\in\mathbb R^{C_{out}\times C_{in}}$ and $b\in\mathbb R^{C_{out}}$.

## Reshape, transpose, and contiguous memory

These operations reorganize interpretation rather than learn values:

- `view`/`reshape` changes the axis grouping when element count is preserved.
- `transpose(a,b)` swaps two axes.
- `contiguous()` lays values out in memory according to the new logical order so a later `view` is
  valid.

Attention transforms `(B,T,C)` into `(B,H,T,D)` where `C=H×D`. It first views features as
`(H,D)`, then transposes the position and head axes. Heads must precede `T` so batched matrix
multiplication computes a separate `(T,T)` score table per head.

Example: `(2,4,12)` with `H=3` becomes `(2,4,3,4)`, then `(2,3,4,4)`. The two final `4`s mean
different things: four positions and four features per head.

## Functions, slopes, and gradients

A function maps inputs to outputs. `f(w)=w²` maps `3` to `9`. Its **derivative** says how a tiny
change in `w` changes `f`; here `df/dw=2w`, so the slope at `w=3` is `6`.

A model loss depends on many parameters. The **gradient** is the collection of partial derivatives:
one slope for each parameter while conceptually holding the others fixed. If a parameter's gradient
is positive, a small move in the negative direction tends to lower the loss locally:

$$
w_{new}=w-\eta\frac{\partial L}{\partial w}.
$$

The learning rate `η` controls step size. This local statement does not promise the loss surface is
simple or that a large step will help.

The one-variable derivative is the limit

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},
$$

when the limit exists. For $f:\mathbb R^n\to\mathbb R$, the partial derivative with respect to
coordinate $i$ is

$$
\frac{\partial f}{\partial x_i}(x)
=\lim_{h\to0}\frac{f(x+h e_i)-f(x)}{h},
$$

where $e_i$ is the `i`th standard basis vector. The gradient collects these coordinates:

$$
\nabla_x f(x)=
\begin{bmatrix}
\partial f/\partial x_0 & \cdots & \partial f/\partial x_{n-1}
\end{bmatrix}^{\!\top}.
$$

For a vector-valued map $g:\mathbb R^n\to\mathbb R^m$, the Jacobian
$J_g(x)\in\mathbb R^{m\times n}$ has entries
$(J_g)_{ij}=\partial g_i/\partial x_j$. If
$f:\mathbb R^m\to\mathbb R$, the multivariable chain rule is

$$
\nabla_x(f\circ g)(x)=J_g(x)^\top\nabla_y f(g(x)).
$$

Reverse-mode automatic differentiation computes these transposed-Jacobian/vector products without
materializing every full Jacobian. This is efficient when a computation has many parameters but one
scalar loss.

## Reductions and numerical meaning

`sum`, `mean`, and `max` remove or shrink axes. Always ask which axis:

- Softmax sums probabilities along vocabulary or key-position axis.
- Cross-entropy averages over flattened batch/position targets.
- LayerNorm computes statistics across the `C` feature axis for each token separately.

Reducing the wrong axis can produce a valid tensor with entirely wrong meaning—the most dangerous
kind of shape bug.

## Prove it to yourself

Run this from the repository root:

```bash
python - <<'PY'
import torch

B, T, C, H = 2, 4, 12, 3
x = torch.arange(B * T * C).reshape(B, T, C)
heads = x.view(B, T, H, C // H).transpose(1, 2)
print("x", tuple(x.shape))
print("heads", tuple(heads.shape))
print("scores", tuple((heads @ heads.transpose(-2, -1)).shape))
PY
```

Expected shapes are `(2,4,12)`, `(2,3,4,4)`, and `(2,3,4,4)`. In the last tensor the axes mean
batch, head, query position, and key position.

## Chapter checkpoint

1. Why does `(B,T,C) @ (C,3C)` produce `(B,T,3C)` conceptually?
2. What is the difference between element-wise multiplication and a dot product?
3. Trace `(B,T,C) -> (B,H,T,D)` and explain why `C` must be divisible by `H`.
4. In plain language, what does a gradient contain?
5. Why can reducing the wrong axis be worse than receiving a shape error?

Next: [Probability and language modeling](03-language-modeling.md).
