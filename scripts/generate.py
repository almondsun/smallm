from __future__ import annotations

import argparse

import torch

from smallm.data import CharTokenizer
from smallm.generation import generate
from smallm.model import GPT, GPTConfig
from smallm.training import load_checkpoint
from smallm.utils.device import default_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = CharTokenizer.load(checkpoint["tokenizer_path"])
    device = default_device()
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(model, prompt, args.max_new_tokens)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
