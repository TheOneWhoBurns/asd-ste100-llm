#!/usr/bin/env python3
"""Run a small repeatable throughput benchmark for the reward function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ste_reward import STERewardFunction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    "Open the door.",
    "The door is open.",
    "The language model has training data.",
    "Open the door and close the door.",
    "The door must be opened.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    completions = SAMPLES * args.repeats
    reward = STERewardFunction(
        glossary_path=PROJECT_ROOT / "policy" / "glossary.yaml"
    )
    reward(SAMPLES)
    started = perf_counter()
    values = reward(completions)
    elapsed = perf_counter() - started
    print(
        json.dumps(
            {
                "samples": len(values),
                "elapsed_seconds": round(elapsed, 6),
                "samples_per_second": round(len(values) / elapsed, 2),
                "positive": sum(value > 0 for value in values),
                "non_positive": sum(value <= 0 for value in values),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
