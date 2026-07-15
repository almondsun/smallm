# 08 — Training the Model

Training is repeated measurement and correction. Each optimizer step obtains a minibatch, computes
predictions and scalar loss, clears old gradients, backpropagates responsibility, and updates the
parameters. Validation periodically measures a separate region without updating anything.

Keep the units distinct: an **example** is one shifted token block; a **batch** contains several
examples; a **target token** is one decision inside an example; an **optimizer step** is one update;
and an **epoch** is one dataset pass. smaLLM budgets steps, not epochs. A full `(B,T)` batch contains
`B×T` targets, but the last DataLoader batch can be smaller.

## Gradient-based fitting

Backpropagation applies the chain rule from cross-entropy through the language head, residual
blocks, attention, and embeddings. A minibatch gradient is a noisy estimate of the full-corpus
gradient. Shuffling changes the noise sequence, so the data-loader generator and global seed are
part of run provenance.

## AdamW

Plain gradient descent uses the same learning rate everywhere. Adam tracks a moving mean and
moving squared magnitude for every parameter's gradients. The first behaves like momentum; the
second adapts coordinates that consistently receive large gradients.

For gradient $g_t$, Adam maintains

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

All products, squares, divisions, and square roots in these equations are element-wise. With
$m_0=v_0=0$, the moments are biased toward zero early in training, so Adam uses

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

Bias-corrected moments $\hat m_t,\hat v_t$ update parameters approximately as

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t.
$$

Equivalently,

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The multiplicative shrinkage term is separate from the gradient moments; that separation is the
defining distinction between AdamW and adding $\lambda\theta$ to the gradient before Adam's
adaptive normalization. smaLLM uses PyTorch's default $\beta_1,\beta_2,\epsilon$ and configures
only learning rate and weight decay.

AdamW decouples weight decay from the adaptive gradient. Learning rate $\eta$, decay $\lambda$,
batch size, and training steps jointly determine the optimization regime; changing one while
calling the rest “controlled” still changes effective regularization or noise.

## Dropout and model modes

Dropout multiplies activations by a Bernoulli mask during training and rescales survivors by the
keep probability. Evaluation disables masks. Any validation helper must restore the model's prior
mode; otherwise subsequent training silently runs without dropout or generation remains stochastic.

For dropout probability $p$ and activation $x_i$, let $m_i\sim\operatorname{Bernoulli}(1-p)$.
Training output is

$$
\tilde x_i=\frac{m_i}{1-p}x_i,
\qquad
\mathbb E[\tilde x_i]=x_i.
$$

The equality is coordinate-wise expectation over the dropout mask. It does not imply a nonlinear
network's expected final output equals its evaluation-mode output.

## Underfitting, overfitting, and checkpoint choice

High training and validation loss suggest insufficient capacity, optimization, or budget. Falling
training loss with rising validation loss indicates widening generalization error. The best
validation checkpoint minimizes a declared held-out estimator; the final checkpoint captures the
end of optimization. Neither is guaranteed to maximize a subjective sample judgment.

Evaluation cadence creates interval censoring: the true best step may lie between evaluations.
Using sampled validation adds estimator variance. Official configs therefore use full,
deterministic validation; `eval_batches` remains available for quick experiments and records its
coverage explicitly.

### Validation-based early stopping

Let validation be observed at evaluation index $j$, with loss $L_j$. Given minimum meaningful
improvement $\delta\geq0$, maintain a reference $R$ and stale count $q$:

$$
(R,q)\leftarrow
\begin{cases}
(L_j,0), & L_j < R-\delta,\\
(R,q+1), & \text{otherwise}.
\end{cases}
$$

Training stops when $q\geq P_{stop}$, where $P_{stop}$ is patience measured in validation
events—not gradient steps or epochs. smaLLM still stores the numerically lowest validation
checkpoint independently of the early-stopping reference. This matters when improvements smaller
than $\delta$ are real enough
to preserve but intentionally too small to reset patience.

Early stopping is a sequential model-selection rule, not regularization: it limits exposure to
overfitting but does not change the objective or gradients before the stop. Its latency is bounded by
$P_{stop}\times\texttt{eval_interval}$, and sampled validation can make the stopping time noisy. Official
modeling configs therefore use full deterministic validation. Summaries record the step ceiling,
actual steps, stop reason, patience, minimum delta, and terminal stale count.

## Determinism limits

Setting Python and PyTorch seeds controls many random choices, but bitwise identity can still fail
across PyTorch versions, devices, kernels, thread schedules, and nondeterministic GPU operations.
Reproducibility means recording enough environment and input identity to explain or bound these
differences—not promising universal identical floating-point bits.

Implementation: [`trainer.py`](../src/smallm/training/trainer.py),
[`seed.py`](../src/smallm/utils/seed.py), and configs under [`configs/`](../configs/).

Checks: distinguish examples, tokens, and optimizer steps; explain why the last short batch changes
throughput; state which evidence selects `best_checkpoint.pt`.

## Trace one smaLLM update

Inside [`train`](../src/smallm/training/trainer.py), the exact order is: fetch `(x,y)` shaped
`(B,T)`; move it to the device; run `model(x,y)`; reject non-finite loss; clear old gradients; call
`loss.backward()` and `optimizer.step()`; then record metrics and periodic validation evidence.

The final checkpoint captures the last step. `best_checkpoint.pt` is saved at the lowest observed
validation loss. Early stopping uses a separate reference plus `min_delta`, so a tiny numerical
improvement can be preserved as best without resetting patience.

## Chapter checkpoint

1. How many target decisions are in a full `(B,T)` batch?
2. Why is a minibatch gradient noisy but still useful?
3. What distinct roles do Adam's first and second moments play?
4. Why can best-checkpoint saving and early-stopping updates disagree for a tiny gain?
5. Why does validation never call `optimizer.step()`?

Next: [Evaluation and generation](09-evaluation-generation.md).
