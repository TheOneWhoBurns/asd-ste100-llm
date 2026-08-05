#!/usr/bin/env python3
"""Generate a text-only baseline from the pinned local model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()

    lock = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))
    model_path = ROOT / lock["destination"]
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writing assistant. Preserve the technical "
                "meaning. Return only the rewritten text."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
    print(response)


if __name__ == "__main__":
    main()

