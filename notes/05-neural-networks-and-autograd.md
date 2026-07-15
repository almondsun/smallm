# 05 — Neural Networks and Autograd: How Prediction Becomes Learning

## A model is a parameterized function

A function maps inputs to outputs. A neural network is a function whose behavior depends on many
adjustable parameters `θ`:

$$
\mathrm{logits}=f_\theta(\mathrm{token\ IDs}).
$$

Formally, let the complete parameter vector be $\theta\in\Theta\subseteq\mathbb R^P$, where
$P$ is the number of scalar model parameters. For fixed vocabulary $\mathcal V=[V]$ and
context length $T$, smaLLM implements

$$
f_\theta:\mathcal V^T\to\mathbb R^{T\times V}.
$$

For batches, the same map is applied independently to each row, yielding
$f_\theta^{(B)}:\mathcal V^{B\times T}\to\mathbb R^{B\times T\times V}$. Batch rows share
$\theta$ but do not exchange hidden states.

Training searches for parameter values that give low next-token loss. “Neural” describes the
historical inspiration and layered structure; the implementation is tensor arithmetic.

## Linear layers

The fundamental learned operation is an affine map:

$$
y=xW^\top+b.
$$

`W` contains weights and `b` contains biases. A purely linear stack collapses into one linear map,
so networks insert nonlinear activation functions between layers. smaLLM uses GELU in its MLP.

For `nn.Linear(3,2)`, one input has three features and the output has two. There are `2×3=6`
weights and two biases. Applied to `(B,T,3)`, the same parameters transform every token position,
producing `(B,T,2)`. Parameter sharing across positions is a core source of efficiency.

For $x\in\mathbb R^n$, $W\in\mathbb R^{m\times n}$, and $b\in\mathbb R^m$, define

$$
g_{W,b}(x)=Wx+b.
$$

Its Jacobians are

$$
\frac{\partial g_i}{\partial x_j}=W_{ij},\qquad
\frac{\partial g_i}{\partial W_{kl}}=\mathbf{1}[i=k]x_l,
\qquad
\frac{\partial g_i}{\partial b_k}=\mathbf{1}[i=k].
$$

These local derivatives are what reverse-mode autograd combines with the upstream loss gradient.

## Embeddings are learned lookup tables

An embedding matrix has shape `(V,C)`. Token ID `i` retrieves row `i`, turning a categorical ID
into `C` continuous features. This is equivalent to multiplying a one-hot vector by the embedding
matrix, but lookup avoids constructing mostly-zero vectors.

The initial dimensions do not have human-assigned meanings. Training organizes them jointly with
the rest of the network. Similar tokens may develop useful geometric relations because shared
contexts reward similar downstream behavior, but no single coordinate is guaranteed to mean “noun”
or “plural.”

## Activation functions

Without nonlinearities, composing affine maps remains affine. GELU approximately gates values by
their magnitude:

$$
\mathrm{GELU}(x)=x\Phi(x),
$$

where `Φ(x)` is the standard normal cumulative distribution. Unlike a hard threshold, GELU is
smooth and allows small negative outputs. In smaLLM, the MLP expands each token from `C` to `4C`,
applies GELU, then projects back to `C`. This lets the model compute richer per-position feature
transformations.

## Computational graphs

PyTorch records operations involving tensors that require gradients. The result is a directed
computational graph:

```text
weights -> logits -> cross-entropy -> loss
```

Calling `loss.backward()` walks this graph backward and applies the chain rule. If
`L=f(g(w))`, then

$$
\frac{dL}{dw}=\frac{dL}{dg}\frac{dg}{dw}.
$$

In a deep network, autograd repeatedly multiplies local derivatives and accumulates contributions
when one value affects loss through multiple paths. The resulting gradient is stored in each
parameter's `.grad` field.

For a composition $h=f_L\circ f_{L-1}\circ\cdots\circ f_1$, define activations
$a_0=x$ and $a_l=f_l(a_{l-1};\theta_l)$. If scalar loss is $\ell(a_L)$, reverse mode starts
with $\bar a_L=\nabla_{a_L}\ell$ and recursively computes

$$
\bar a_{l-1}=J_{f_l,a_{l-1}}^\top\bar a_l,
\qquad
\nabla_{\theta_l}\ell=J_{f_l,\theta_l}^\top\bar a_l.
$$

The bar denotes an adjoint or upstream gradient. Residual addition creates two graph paths, so their
gradient contributions add. Parameter reuse across positions likewise sums all contributions into
one shared `.grad` tensor.

## A one-parameter learning example

Let prediction be `ŷ = wx`, target `y=6`, input `x=2`, and squared loss `(ŷ-y)²`. Start with `w=1`:

```text
prediction = 1×2 = 2
loss = (2-6)² = 16
dL/dw = 2(wx-y)x = 2(2-6)2 = -16
```

With learning rate `0.1`, gradient descent gives `w_new = 1 - 0.1(-16) = 2.6`. The prediction
moves toward six. Real language modeling uses cross-entropy, hundreds of thousands of parameters,
and AdamW, but the responsibility chain is the same.

## The training-step lifecycle

A correct optimizer step has four conceptual stages:

```python
optimizer.zero_grad()   # discard gradients from the previous step
logits, loss = model(x, y)
loss.backward()         # populate parameter.grad
optimizer.step()        # update parameters using those gradients
```

PyTorch accumulates gradients by default because some algorithms need multiple backward passes.
For ordinary independent minibatches, forgetting `zero_grad()` unintentionally sums gradients
across steps.

## Why minibatches work

The ideal empirical objective averages over every training target. Computing that full gradient at
every step is expensive. A minibatch samples several blocks and produces a noisy estimate.

- Larger batches reduce gradient noise but cost more memory and computation per step.
- Smaller batches update more frequently but vary more from sample to sample.
- Shuffling changes which examples share batches and therefore the noise sequence.

Batch positions do not communicate with each other in the model; batching is parallel evaluation
of independent sequences using shared weights.

If the finite training dataset has $N$ examples with losses $\ell_i(\theta)$, full empirical
risk is

$$
L(\theta)=\frac1N\sum_{i=0}^{N-1}\ell_i(\theta),
\qquad
\nabla L(\theta)=\frac1N\sum_i\nabla\ell_i(\theta).
$$

For a uniformly sampled minibatch $S$ of size $B$, the estimator

$$
\widehat g_S(\theta)=\frac1B\sum_{i\in S}\nabla\ell_i(\theta)
$$

is unbiased under the stated uniform-sampling scheme:
$\mathbb E_S[\widehat g_S]=\nabla L$. Successive shuffled DataLoader batches are not independent
draws with replacement, but their ordering still supplies a controlled stochastic update sequence.

## Training mode and evaluation mode

`nn.Module` tracks a `.training` flag. `model.train()` enables dropout; `model.eval()` disables it.
This does not turn gradients on or off. `torch.no_grad()` separately prevents gradient graph
construction during evaluation or generation.

smaLLM's validation helper remembers the previous mode, evaluates without gradients, then restores
training mode. Otherwise a validation call could silently disable dropout for later updates.

## Prove it to yourself

```bash
python - <<'PY'
import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
y = torch.tensor(6.0)
loss = (w * x - y) ** 2
loss.backward()
print("loss", loss.item())
print("gradient", w.grad.item())
PY
```

Expect loss `16` and gradient `-16`. Change `w` and predict the gradient before rerunning.

## Chapter checkpoint

1. Why would a network made only of linear layers still be one linear transformation?
2. How is an embedding lookup related to one-hot matrix multiplication?
3. What does `loss.backward()` calculate, and where are its results stored?
4. Why must ordinary training zero gradients each step?
5. Distinguish `model.eval()` from `torch.no_grad()`.

Next: [Attention from scratch](06-attention.md).
