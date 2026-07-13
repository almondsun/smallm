# Optimization and Training Dynamics

## Gradient-based fitting

Backpropagation applies the chain rule from cross-entropy through the language head, residual
blocks, attention, and embeddings. A minibatch gradient is a noisy estimate of the full-corpus
gradient. Shuffling changes the noise sequence, so the data-loader generator and global seed are
part of run provenance.

## AdamW

For gradient \(g_t\), Adam maintains

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Bias-corrected moments \(\hat m_t,\hat v_t\) update parameters approximately as

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t.
\]

AdamW decouples weight decay from the adaptive gradient. Learning rate \(\eta\), decay \(\lambda\),
batch size, and training steps jointly determine the optimization regime; changing one while
calling the rest “controlled” still changes effective regularization or noise.

## Dropout and model modes

Dropout multiplies activations by a Bernoulli mask during training and rescales survivors by the
keep probability. Evaluation disables masks. Any validation helper must restore the model's prior
mode; otherwise subsequent training silently runs without dropout or generation remains stochastic.

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

Let validation be observed at evaluation index (j), with loss (L_j). Given minimum meaningful
improvement (delta\geq0), maintain a reference (R) and stale count (q):

\[
(R,q)\leftarrow
\begin{cases}
(L_j,0), & L_j < R-\delta,\\
(R,q+1), & \text{otherwise}.
\end{cases}
\]

Training stops when (q\geq P), where (P) is patience measured in validation events—not gradient
steps or epochs. smaLLM still stores the numerically lowest validation checkpoint independently of
the early-stopping reference. This matters when improvements smaller than (delta) are real enough
to preserve but intentionally too small to reset patience.

Early stopping is a sequential model-selection rule, not regularization: it limits exposure to
overfitting but does not change the objective or gradients before the stop. Its latency is bounded by
(P\times\texttt{eval_interval}), and sampled validation can make the stopping time noisy. Official
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
