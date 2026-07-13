# Baselines, Evaluation, and Generation

## Count-based baselines

Uniform assigns \(p(x)=1/V\), giving loss \(\log V\). A unigram estimates
\(p(x)=c(x)/N\). A bigram estimates transitions using add-\(\alpha\) smoothing:

\[
p(x_t=j\mid x_{t-1}=i)=\frac{c(i,j)+\alpha}{c(i)+\alpha V}.
\]

Baselines answer whether the neural model beats trivial vocabulary knowledge or local transition
statistics. They must use the same training-fitted tokenizer and held-out text as the neural run.

## Non-overlapping weighted validation

Validation partitions target positions into consecutive blocks. If block `k` has mean loss
\(\ell_k\) over \(n_k\) targets, corpus loss is

\[
\mathcal L=\frac{\sum_k n_k\ell_k}{\sum_k n_k},
\]

not \(K^{-1}\sum_k\ell_k\). When sampling blocks, evenly spaced deterministic indices cover the
sequence more honestly than the first few overlapping windows. Metrics record evaluated and total
target counts so `coverage = evaluated/total` is auditable.

## Autoregressive decoding

Generation repeatedly crops the prefix to the context limit, computes final-position logits, and
chooses one token. Greedy decoding uses `argmax`; it is deterministic but can enter repetition
loops. Temperature \(\tau\) changes logits to \(z/\tau\): low values sharpen, high values flatten.
Top-k sets all but the `k` largest logits to \(-\infty\), renormalizes, then samples.

A seeded `torch.Generator` controls categorical draws on the selected device. The full comparison
contract includes checkpoint kind, exact prompt, maximum new tokens, temperature, top-k, seed, and
greedy flag.

## Surface diagnostics

Distinct-1 and distinct-2 measure unique unigrams/bigrams divided by their totals. Repetition rate
and longest repeated runs detect collapse. These measures can reward random nonsense and penalize
legitimate refrain; they complement, not replace, blinded human judgment or held-out NLL.

Implementation: [`baselines.py`](../src/smallm/evaluation/baselines.py),
[`sample.py`](../src/smallm/generation/sample.py), and
[`diagnostics.py`](../src/smallm/generation/diagnostics.py).

Checks: explain why temperature is irrelevant in greedy mode; why BPE distinct-n differs from
character distinct-n; and why a best-loss checkpoint may generate less diverse text.
