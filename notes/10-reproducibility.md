# 10 — Reproducible Experiments and Honest Evidence

A model can run correctly while an experiment answers the wrong question. This chapter separates
software outputs from justified claims. The core idea is an evidence chain: identify inputs,
declare which choices may use which data, preserve transformations and outcomes, and interpret only
what the design supports.

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
## Seed ensembles and descriptive uncertainty

A random seed fixes initialization, minibatch order, dropout masks, and sampling streams; it is an
experimental condition, not a hyperparameter to optimize. For preregistered seeds
(s_1,\ldots,s_n) and metric (x_i), report every observation plus

$$
\bar{x}=\frac1n\sum_{i=1}^n x_i,
\qquad
\sigma_{\mathrm{pop}}=\sqrt{\frac1n\sum_{i=1}^n(x_i-\bar{x})^2}.
$$

smaLLM uses population standard deviation because the report describes the complete, explicitly
chosen seed set; it does not pretend three seeds estimate a universal sampling distribution. The
minimum and maximum expose asymmetry that a mean and deviation can hide. Stop step is itself a
random outcome under early stopping and must be summarized alongside model quality.

Never discard a completed seed because it weakens the conclusion, and never report the best seed as
the expected result. A hyperparameter difference much smaller than seed-to-seed spread is not robust
evidence. Decoding randomness is held fixed when comparing training seeds so observed generation
variation comes from model training rather than a second uncontrolled random stream.

## External validity and corpus-by-seed interactions

A random split of one book estimates performance on held-out text from essentially the same source;
it does not establish robustness to a new author, style, vocabulary, or document structure. A new
book changes the empirical distribution $P_C(X)$, so a tokenizer comparison can be modeled as

$$
y_{tcs}=\mu+\alpha_t+\beta_c+(\alpha\beta)_{tc}+u_s+\varepsilon_{tcs},
$$

where $t$ is tokenizer, $c$ corpus, $s$ seed, $\alpha_t$ is the tokenizer effect,
$\beta_c$ is corpus difficulty, $(\alpha\beta)_{tc}$ is the tokenizer-by-corpus interaction,
and $u_s$ captures training randomness. One seed on a second corpus observes a cell but cannot
separate interaction from seed noise. A balanced corpus-by-seed matrix can.

Matching corpus length controls the amount of text, not its entropy, lexical diversity, boundary
statistics, or chronological-tail difficulty. Matching token context also requires converting to
characters:

$$
L_{\mathrm{chars}}=L_{\mathrm{tokens}}\frac{N_{\mathrm{train\ chars}}}{N_{\mathrm{train\ tokens}}}.
$$

This is an average receptive-field proxy; individual ByteBPE sequences still vary in character
length. BPC makes held-out likelihood comparable across lossless tokenizers,

$$
\operatorname{BPC}=\frac{-\log_2 p(x_{1:n})}{n},
$$

but comparable metrics do not remove design confounds such as vocabulary-dependent parameter
counts or checkpoint selection on the same validation tail.

External-validity claims should therefore form a ladder: same split across seeds; deterministic
alternative splits; a second source; multiple source families; finally, a sealed test distribution.
Each rung supports a wider claim, and none licenses the next one automatically. Milestone 024
reached the second-source rung with one seed. Milestone 025 fills a balanced 2 × 2 × 3 matrix and
observes a negative ByteBPE512-minus-character contrast in every same-seed pair. Its positive
corpus interaction shows why a replicated direction is not a universal effect size: Peter Pan
attenuates the advantage even though it does not reverse it.

For paired seeds, define

$$
d_{cs}=y_{\mathrm{Byte},c,s}-y_{\mathrm{char},c,s},
\qquad
I_s=d_{\mathrm{PeterPan},s}-d_{\mathrm{Alice},s}.
$$

Pairing subtracts seed-specific difficulty shared by both tokenizers within a corpus. The
difference-in-differences $I_s$ measures how much the tokenizer contrast changes between corpora.
It is descriptive here: three seeds and two related books do not justify asymptotic inference, and
the paired observations are not mutually independent across corpus because a seed is reused.

## Validation selection and a sealed test set

Early stopping chooses

$$
\hat{k}=\arg\min_{k\in\mathcal{K}}\widehat{L}_{\mathrm{val}}(\theta_k),
$$

so the reported validation minimum is an order statistic selected from repeated looks. Even when
each evaluation is unbiased for a fixed checkpoint, the minimum is optimistically biased as an
estimate of future performance. Reusing the same validation tail to choose tokenizers, context,
regularization, and follow-up questions compounds researcher degrees of freedom.

A three-way split separates roles: training fits parameters and tokenizer merges; validation drives
early stopping and configuration choice; a sealed test segment is read once after the decision rule
is frozen. For chronological text, the ordering must remain explicit because a terminal test block
measures forward generalization, not exchangeable random-split performance. Test metrics must never
flow back into checkpoint choice, hyperparameter tuning, or seed selection.

## The test set as an information firewall

Let the research process before test access produce a decision
$D=f(X_{\mathrm{train}},X_{\mathrm{val}},R)$, where $R$ includes seeds, code, and declared
rules. A confirmatory test estimate is interpretable because $X_{\mathrm{test}}$ is absent from
$f$. Once its result $T(D,X_{\mathrm{test}})$ is observed, any later decision
$D'=g(D,T)$ is statistically coupled to the test set. The same examples have become validation
data in practice, regardless of their filename.

An information firewall therefore requires both a data boundary and a workflow boundary. smaLLM's
training path computes split indices and records the terminal character count, but does not encode
the terminal text into tokens. The separate evaluator verifies the copied manifest, prepared-text
hash, checkpoint hash, and checkpoint step before evaluating full coverage. Recording these
identities makes accidental substitution detectable.

The no-overwrite artifact rule prevents a common accidental second look, but cannot prove secrecy:
a user controls the local filesystem and can delete outputs or modify code. Reproducibility tooling
can make violations conspicuous; it cannot replace research discipline or an external data
custodian. After milestone 026, the Alice and Peter Pan terminal segments are permanently consumed
for this research line.

The observed generalization gap is

$$
G=\operatorname{BPC}_{\mathrm{test}}-\operatorname{BPC}_{\mathrm{val}}.
$$

A positive $G$ does not by itself prove overfitting: a chronological terminal segment may have
higher irreducible entropy or a shifted style. Define the tokenizer contrast as

$$
\Delta_c=\operatorname{BPC}_{\mathrm{Byte},c}-\operatorname{BPC}_{\mathrm{char},c}.
$$

Comparing this contrast across validation and test asks whether the frozen decision preserves its direction
under forward distribution shift. Milestone 026 finds $G>0$ for every model while
$\Delta_c<0$ on both sealed tests.

## Preregistration and distributional replication

Preregistration separates a hypothesis from its outcome in time. A useful repository-native record
commits the source identity, extraction rule, split, seeds, configurations, stopping policy, primary
metric, directional contrast, and test-access rule before data or model access. Git history then
makes later changes observable. It does not prevent misconduct, but it sharply reduces ambiguity
about which choices preceded the result.

External-distribution replication asks whether a fixed decision transfers when the data-generating
process changes. Let $c$ index corpora and define

$$
\Delta_c=\operatorname{BPC}_{\mathrm{candidate},c}
-\operatorname{BPC}_{\mathrm{control},c}.
$$

Repeatedly observing
$\operatorname{sign}(\Delta_c)<0$ supports directional robustness. It does not imply a common
effect size: heterogeneity in $|\Delta_c|$ is evidence that the tokenizer interacts with corpus
structure.

The chronological generalization gap is

$$
G_c=\operatorname{BPC}_{\mathrm{test},c}-\operatorname{BPC}_{\mathrm{val},c}.
$$

It is likewise not a model-only property. Milestone 026 finds $G_c>0$ for Alice and Peter Pan, while preregistered
milestone 027 finds $G_c<0$ for both Hamlet models. Textual position can alter speaker mix,
chapter or scene structure, punctuation, verse density, and intrinsic entropy. A professional
interpretation therefore reports both the model contrast and the regional difficulty shift rather
than labeling every positive gap "overfitting" or every negative gap "improved generalization."

## Balanced confirmatory panels and analysis contracts

A multi-corpus confirmatory panel adds two forms of blocking. Same-seed pairing within corpus
removes the seed component shared by candidate and control, while repeating the pair across corpus
families exposes tokenizer-by-corpus heterogeneity. For corpora
$c\in\{1,\ldots,N_c\}$, tokenizers $t\in\{A,B\}$, and fixed seeds
$s\in\{1,\ldots,N_s\}$, completeness is part of the estimand:

$$
\mathcal{D}=\{y_{tcs}:t\in\{A,B\},c\in\{1,\ldots,N_c\},s\in\{1,\ldots,N_s\}\}.
$$

Silently dropping a failed, inconvenient, or missing cell changes $\mathcal{D}$ after outcomes
are partially known. A professional aggregator therefore validates the exact Cartesian product,
unique paths, common seed sets, artifact schema, checkpoint kind and hash, corpus hash, and full
coverage before calculating any effect. Validation failure is an outcome of the analysis contract,
not an invitation to weaken it.

Comparability fingerprints must encode causes of the likelihood result—data, model, optimizer,
training budget, and selection policy—but exclude downstream presentation controls such as sample
prompt, decoding temperature, and generation seed. Including an irrelevant sample seed creates a
false mismatch; excluding a learning rate or split creates a false match. Milestone 028 encountered
the former after test access, repaired it transparently without changing artifacts, and added a
regression test. This illustrates why preregistering the analysis rule is necessary but executable
schema checks are still fallible software that require audit trails.

Milestone 028's complete 2 × 2 × 3 panel finds all six ByteBPE512-minus-character test contrasts
negative. The result strengthens a directional claim, not a universal effect-size claim: the mean
interaction differs across Art of War and Lincoln, the corpora are fixed rather than sampled, and
three population-SD summaries are descriptive. Every accessed terminal region is now historical
evidence and cannot serve as a clean test set for a parameter-matched follow-up.

## Capacity matching and the final panel

Tokenizer vocabulary changes model capacity because smaLLM has both a `V×C` input embedding and an
independent `C×V` output head. A ByteBPE512 comparison with an 80-ish-character vocabulary therefore
changes tokenization and roughly `2C(512-80)` learned weights at once.

Milestones 029–030 preregister and complete a 3-corpus × 3-arm × 3-seed panel with a wider char136
control matched within 0.40% of ByteBPE512 parameters. ByteBPE512 wins eight of nine matched sealed
pairs and has a negative mean contrast on each corpus, while Douglass seed 4242 reverses by
`+0.0018` BPC. The correct conclusion is not “BPE always wins.” It is that the fixed panel's
advantage generally survives the measured capacity confound, with a small corpus-dependent effect
that can reverse in one seed.

This is the repository's final modeling evidence. Its value comes partly from stopping as declared:
consumed test regions do not become fresh selection data merely because another idea is interesting.

## Chapter checkpoint

1. Distinguish training, validation selection, and sealed-test confirmation.
2. Why is a random seed an experimental condition rather than a result to optimize?
3. What does pairing the same seed across arms remove from a contrast?
4. Why can corpus means differ without either analysis being wrong?
5. How did vocabulary size become a parameter-count confound in this architecture?
6. Why is preserving a reversal stronger evidence practice than hiding it?

Next: [End-to-end code walkthrough](11-code-walkthrough.md).
