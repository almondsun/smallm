# 09 — Evaluation and Generation

Training loss measures batches the model may learn from. Evaluation asks how much probability a
frozen checkpoint assigns to held-out targets. Generation asks what happens when model choices
become later context. These are related views, not interchangeable definitions of quality.

## Count-based baselines

Uniform assigns \(p(x)=1/V\), giving loss \(\log V\). A unigram estimates
\(p(x)=c(x)/N\). A bigram estimates transitions using add-\(\alpha\) smoothing:

\[
p(x_t=j\mid x_{t-1}=i)=\frac{c(i,j)+\alpha}{c_{prev}(i)+\alpha V}.
\]

Baselines answer whether the neural model beats trivial vocabulary knowledge or local transition
statistics. They must use the same training-fitted tokenizer and held-out text as the neural run.

Formally, for training token sequence \(z_{0:N-1}\), define token counts
\(c(j)=\sum_{t=0}^{N-1}\mathbf1[z_t=j]\) and transition counts
\(c(i,j)=\sum_{t=1}^{N-1}\mathbf1[z_{t-1}=i\land z_t=j]\). Then

\[
p_{uniform}(j)=\frac1V,
\qquad
p_{unigram}(j)=\frac{c(j)}{N},
\]

and, with \(c_{prev}(i)=\sum_j c(i,j)\),

\[
p_{bigram}(j\mid i)=\frac{c(i,j)+\alpha}{c_{prev}(i)+\alpha V}.
\]

The unsmoothed unigram assigns zero probability to a token absent from training, causing infinite
NLL if it occurs in held-out text. Add-\(\alpha\) bigram smoothing with \(\alpha>0\) gives every
transition positive support.

When scoring validation, smaLLM conditions its first token on the final training token; later
validation tokens are conditioned on the preceding validation token. All frequency counts remain
fitted from training transitions only.

## Non-overlapping weighted validation

Validation partitions target positions into consecutive blocks. If block `k` has mean loss
\(\ell_k\) over \(n_k\) targets, corpus loss is

\[
\mathcal L=\frac{\sum_k n_k\ell_k}{\sum_k n_k},
\]

not \(K^{-1}\sum_k\ell_k\). When sampling blocks, evenly spaced deterministic indices cover the
sequence more honestly than the first few overlapping windows. Metrics record evaluated and total
target counts so `coverage = evaluated/total` is auditable.

Let held-out targets be indexed by finite set \(I\), partitioned into disjoint block sets
\(I_1,\ldots,I_K\). If token NLL is \(s_i\), then

\[
\ell_k=\frac1{|I_k|}\sum_{i\in I_k}s_i
\quad\Longrightarrow\quad
\frac{\sum_k|I_k|\ell_k}{\sum_k|I_k|}
=\frac1{|I|}\sum_{i\in I}s_i.
\]

This equality proves the weighted aggregation. The unweighted mean of block means equals the corpus
mean only when every block has equal target count.

## Autoregressive decoding

Generation repeatedly crops the prefix to the context limit, computes final-position logits, and
chooses one token. Greedy decoding uses `argmax`; it is deterministic but can enter repetition
loops. Temperature \(\tau\) changes logits to \(z/\tau\): low values sharpen, high values flatten.
Top-k sets all but the `k` largest logits to \(-\infty\), renormalizes, then samples.

A seeded `torch.Generator` controls categorical draws on the selected device. The full comparison
contract includes checkpoint kind, exact prompt, maximum new tokens, temperature, top-k, seed, and
greedy flag.

One generation iteration crops the prefix to `block_size`, runs the model, keeps only final-position
logits, chooses one ID, appends it, and repeats. Temperature below one sharpens relative gaps; above
one flattens them. Top-k removes all but the largest `k` candidates. Greedy mode uses `argmax`, so
this implementation bypasses temperature and top-k in greedy comparisons.

Given final-position logits \(z\in\mathbb R^V\), greedy decoding selects

\[
\hat x=\arg\max_{j\in[V]}z_j.
\]

Mathematically, `argmax` is set-valued when logits tie. The implementation uses `torch.argmax`,
which returns the first maximizing index, so greedy decoding remains a deterministic function.

Temperature sampling with \(\tau>0\) uses

\[
p_\tau(j)=\frac{e^{z_j/\tau}}{\sum_r e^{z_r/\tau}}.
\]

For top-k set \(K\subseteq[V]\) containing indices of the `k` largest logits, define

\[
p_{\tau,k}(j)=
\begin{cases}
\dfrac{e^{z_j/\tau}}{\sum_{r\in K}e^{z_r/\tau}},&j\in K,\\
0,&j\notin K.
\end{cases}
\]

As \(\tau\to0^+\), mass concentrates on maximizing logits (with tie behavior depending on the
limit); as \(\tau\to\infty\), the untruncated distribution approaches uniform over the vocabulary.

## Surface diagnostics

Distinct-1 and distinct-2 measure unique unigrams/bigrams divided by their totals. Repetition rate
and longest repeated runs detect collapse. These measures can reward random nonsense and penalize
legitimate refrain; they complement, not replace, blinded human judgment or held-out NLL.

Implementation: [`baselines.py`](../src/smallm/evaluation/baselines.py),
[`sample.py`](../src/smallm/generation/sample.py), and
[`diagnostics.py`](../src/smallm/generation/diagnostics.py).

Checks: explain why temperature is irrelevant in greedy mode; why BPE distinct-n differs from
character distinct-n; and why a best-loss checkpoint may generate less diverse text.

## Chapter checkpoint

1. Why must validation block means be weighted by target count?
2. What does full coverage mean, and where is it recorded?
3. Trace one generation iteration from token IDs to an appended ID.
4. Contrast greedy decoding, temperature sampling, and top-k sampling.
5. Why can a lower-loss checkpoint still produce subjectively worse text?

Next: [Reproducible experiments and honest evidence](10-reproducibility.md).
