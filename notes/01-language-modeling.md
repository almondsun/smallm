# Language Modeling Foundations

## Autoregressive probability

For tokens \(x_{1:T}\), the chain rule gives

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

A causal model shares parameters \(\theta\) across these conditionals and minimizes empirical
negative log-likelihood (NLL):

\[
\mathcal L(\theta)=-\frac{1}{N}\sum_{n,t}\log
p_\theta(x_t^{(n)}\mid x_{<t}^{(n)}).
\]

Teacher forcing supplies true prefixes during training. Generation supplies model-produced
prefixes, so local errors alter later conditioning—a source of exposure mismatch.

## Logits, softmax, and cross-entropy

At position \(t\), logits \(z_t\in\mathbb R^V\) induce

\[
p_t(i)=\frac{e^{z_{t,i}}}{\sum_j e^{z_{t,j}}},\qquad
\ell_t=-\log p_t(y_t).
\]

The useful gradient identity is

\[
\frac{\partial \ell_t}{\partial z_{t,i}}=p_t(i)-\mathbf 1[i=y_t].
\]

In [`GPT.forward`](../src/smallm/model/gpt.py), `(B,T,V)` logits and `(B,T)` targets become
`B*T` categorical decisions. PyTorch uses a stable log-sum-exp calculation.

For true distribution \(q\) and model \(p\), cross-entropy decomposes as
\(H(q,p)=H(q)+D_{KL}(q\Vert p)\). Lower held-out loss means better predictive probability under
that sample; it does not prove factuality, prose coherence, or usefulness.

## Perplexity and bits per character

Per-token perplexity is \(\exp(\mathcal L)\). Loss `2.0` nats means perplexity `7.39`, the
geometric-mean inverse target probability. Different tokenizers define different prediction units,
so their token loss and perplexity are not directly comparable.

For total NLL \(S\) over \(C_s\) represented source characters,

\[
\mathrm{BPC}=\frac{S}{C_s\ln 2}.
\]

`693.147` nats over `1,000` characters is `1.0` bit/character. Numerator and denominator must
cover the same targets. A prefix loss multiplied by whole-corpus counts is not exact BPC.

## Failure modes and checks

- Averaging batch means equally despite unequal target counts.
- Comparing perplexity across tokenizers.
- Selecting checkpoints on test rather than validation evidence.
- Inferring generation quality from NLL alone.

Explain why BPC is comparable across tokenizations only when preprocessing, source-character
coverage, and held-out text are identical.
