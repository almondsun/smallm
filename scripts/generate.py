from __future__ import annotations

import argparse

import torch

from smallm.data import CharTokenizer
from smallm.generation import generate, generation_diagnostics
from smallm.model import GPT, GPTConfig
from smallm.training import load_checkpoint
from smallm.training.runs import checkpoint_path_for_run, resolve_run_path
from smallm.utils.device import default_device


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--run")
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if args.run is not None:
        run_dir = resolve_run_path(args.run, run_name=args.run_name, runs_dir=args.runs_dir)
        checkpoint_path = checkpoint_path_for_run(run_dir)

    checkpoint = load_checkpoint(checkpoint_path)
    if "tokenizer" in checkpoint:
        tokenizer = CharTokenizer(checkpoint["tokenizer"]["stoi"])
    else:
        tokenizer = CharTokenizer.load(checkpoint["tokenizer_path"])
    device = default_device()
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(
        model,
        prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        greedy=args.greedy,
    )
    text = tokenizer.decode(output[0].tolist())
    print(text)
    if args.diagnostics:
        print()
        print("diagnostics:")
        for key, value in generation_diagnostics(text).items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
