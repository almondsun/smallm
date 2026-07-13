# Reproducibility, Artifacts, Security, and Experiment Design

## Evidence chain

```mermaid
flowchart TD
    A[Raw source] -->|normalize + hash| B[Prepared corpus + manifest]
    B -->|verify + split| C[Training and validation text]
    C -->|fit on train only| D[Tokenizer]
    D --> E[Config-driven training]
    E --> F[Metrics + final/best checkpoints]
    F --> G[Controlled generation]
    G --> H[Experiment report]
```

A professional result is a chain of identities and transformations. A filename is not provenance;
a checksum is. A config is not enough if package versions or preprocessing code differ. A sample
is not evidence unless decoding controls and checkpoint identity are known.

## Artifact contracts

Each completed run contains a config snapshot, JSONL metrics, summary, final checkpoint, optional
best checkpoint, sample, and copied manifest. Schema versions permit additive evolution and make
incompatible interpretation explicit. Legacy readers accept older character tokenizers; new
writers include an unknown token and schema number.

Atomic writes serialize to a sibling temporary file, flush it, and replace the destination. This
prevents readers from observing a partially written checkpoint or JSON document. A run becomes
discoverable only once `summary.json` exists, so a failed orchestration cannot become `latest`.

## Trust boundaries

Python pickle can execute constructors during deserialization. PyTorch checkpoints are therefore
loaded with `weights_only=True` and validated as mappings containing model state and model config.
This narrows, but does not erase, the artifact trust boundary. Corpora can contain private text;
samples may memorize fragments; paths and manifests can point at local data. Runtime artifacts
remain ignored and must not be attached casually to issues.

## Controlled experiments

A useful report states hypothesis, independent variable, controls, corpus identity, seeds,
commands, quantitative outputs, qualitative samples, limitations, and a conclusion proportional
to evidence. Multiple comparisons on one validation set gradually overfit researcher decisions.
The test set, if introduced, should remain sealed until a final choice is made.

Experiments 016–017 fitted tokenizers on full text and derived BPC from prefix loss with whole-set
counts. Their raw observations remain historical, but corrected metrics supersede their headline
interpretation. This is not an embarrassment: preserving an erratum is stronger scientific
behavior than silently rewriting evidence.

## Project boundaries

smaLLM intentionally omits distributed training, mixed precision, resume state, production
tokenizers, model families, and remote tracking. Adding them is justified only when a concrete
experiment or reliability contract needs them. Small, inspectable boundaries are a design feature.

Implementation: [`artifacts.py`](../src/smallm/training/artifacts.py),
[`checkpoints.py`](../src/smallm/training/checkpoints.py), and
[`runs.py`](../src/smallm/training/runs.py).

Checks: reconstruct a result from its summary; identify which failures occur before any run
directory exists; explain why old and corrected BPC values cannot be compared as one series.
### Seed ensembles and descriptive uncertainty

A random seed fixes initialization, minibatch order, dropout masks, and sampling streams; it is an
experimental condition, not a hyperparameter to optimize. For preregistered seeds
(s_1,\ldots,s_n) and metric (x_i), report every observation plus

\[
\bar{x}=\frac1n\sum_{i=1}^n x_i,
\qquad
\sigma_{\mathrm{pop}}=\sqrt{\frac1n\sum_{i=1}^n(x_i-\bar{x})^2}.
\]

smaLLM uses population standard deviation because the report describes the complete, explicitly
chosen seed set; it does not pretend three seeds estimate a universal sampling distribution. The
minimum and maximum expose asymmetry that a mean and deviation can hide. Stop step is itself a
random outcome under early stopping and must be summarized alongside model quality.

Never discard a completed seed because it weakens the conclusion, and never report the best seed as
the expected result. A hyperparameter difference much smaller than seed-to-seed spread is not robust
evidence. Decoding randomness is held fixed when comparing training seeds so observed generation
variation comes from model training rather than a second uncontrolled random stream.

### External validity and corpus-by-seed interactions

A random split of one book estimates performance on held-out text from essentially the same source;
it does not establish robustness to a new author, style, vocabulary, or document structure. A new
book changes the empirical distribution \(P_C(X)\), so a tokenizer comparison can be modeled as

\[
y_{tcs}=\mu+\alpha_t+\beta_c+(\alpha\beta)_{tc}+u_s+\varepsilon_{tcs},
\]

where \(t\) is tokenizer, \(c\) corpus, \(s\) seed, \(\alpha_t\) is the tokenizer effect,
\(\beta_c\) is corpus difficulty, \((\alpha\beta)_{tc}\) is the tokenizer-by-corpus interaction,
and \(u_s\) captures training randomness. One seed on a second corpus observes a cell but cannot
separate interaction from seed noise. A balanced corpus-by-seed matrix can.

Matching corpus length controls the amount of text, not its entropy, lexical diversity, boundary
statistics, or chronological-tail difficulty. Matching token context also requires converting to
characters:

\[
L_{\mathrm{chars}}=L_{\mathrm{tokens}}\frac{N_{\mathrm{train\ chars}}}{N_{\mathrm{train\ tokens}}}.
\]

This is an average receptive-field proxy; individual ByteBPE sequences still vary in character
length. BPC makes held-out likelihood comparable across lossless tokenizers,

\[
\operatorname{BPC}=\frac{-\log_2 p(x_{1:n})}{n},
\]

but comparable metrics do not remove design confounds such as vocabulary-dependent parameter
counts or checkpoint selection on the same validation tail.

External-validity claims should therefore form a ladder: same split across seeds; deterministic
alternative splits; a second source; multiple source families; finally, a sealed test distribution.
Each rung supports a wider claim, and none licenses the next one automatically. Milestone 024
reaches the second-source rung: it replicates ByteBPE512's direction on Peter Pan, while its narrow
margin and single seed leave the interaction term unresolved.
