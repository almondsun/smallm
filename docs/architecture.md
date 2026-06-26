# Architecture

smaLLM is organized as a small PyTorch package with thin scripts at the edge.

- `smallm.data` owns tokenization and token block datasets.
- `smallm.model` owns GPT-style Transformer modules.
- `smallm.training` owns training orchestration and checkpoints.
- `smallm.generation` owns sampling.
- `smallm.utils` contains shared runtime helpers.
