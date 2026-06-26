from __future__ import annotations

import argparse

from smallm.config import load_config
from smallm.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    checkpoint_path = train(load_config(args.config))
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
