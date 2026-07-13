# Notation, Glossary, Derivations, and References

## Shape table

| Symbol | Meaning |
| --- | --- |
| \(B\) | batch size |
| \(T\) | current sequence length |
| \(V\) | vocabulary size |
| \(C\) | embedding width |
| \(H\) | attention heads |
| \(D=C/H\) | per-head width |
| \(X\in\mathbb R^{B\times T\times C}\) | hidden states |
| \(Q,K,V_a\in\mathbb R^{B\times H\times T\times D}\) | attention tensors |
| \(Z\in\mathbb R^{B\times T\times V}\) | output logits |

## Worked weighted-loss example

Suppose validation blocks contain `128`, `128`, and `17` targets with mean NLL `1.5`, `1.7`, and
`2.0`. Total NLL is `128(1.5)+128(1.7)+17(2.0)=443.6`; mean NLL is `443.6/273=1.6249`, not the
unweighted `1.7333`. If those targets represent `400` source characters, BPC is
`443.6/(400 ln 2)=1.6000`.

## Glossary

- **Causal:** position `t` cannot depend on tokens after `t`.
- **Checkpoint:** serialized model configuration, weights, tokenizer state, and run metadata.
- **Cross-entropy:** expected negative log probability assigned to the target.
- **Held-out:** excluded from fitting, including learned preprocessing.
- **Leakage:** information from evaluation data influences fitting or selection improperly.
- **Logit:** unnormalized log-score before softmax.
- **Nat:** information unit from a natural logarithm; one bit is `ln(2)` nats.
- **Perplexity:** exponential of mean token NLL.
- **Provenance:** evidence identifying inputs, transformations, environment, and outputs.
- **Residual connection:** identity path plus a learned transformation.
- **Token:** vocabulary unit predicted by the model; not necessarily a character or word.

## Primary references

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018.
- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), 2016.
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.
- Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016.
- Hendrycks and Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), 2016.
- Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html), 2014.
- Dodge et al., [Show Your Work: Improved Reporting of Experimental Results](https://aclanthology.org/D19-1224/), 2019.

## Further navigation

Return to the [handbook index](README.md), inspect the
[architecture](../docs/architecture.md), or follow the chronological
[experiment index](../docs/experiments.md).
