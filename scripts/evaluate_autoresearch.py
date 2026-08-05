#!/usr/bin/env python3
"""Evaluate the fixed positive and adversarial reward sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ste_reward import score_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, default=PROJECT_ROOT / "data/eval/gpt5_positive.jsonl")
    parser.add_argument("--adversarial", type=Path, default=PROJECT_ROOT / "data/eval/adversarial.jsonl")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data/eval/command_validation.jsonl")
    args = parser.parse_args()

    positive = read_jsonl(args.positive)
    adversarial = read_jsonl(args.adversarial)
    validation = read_jsonl(args.validation)
    started = perf_counter()
    positive_results = [
        score_text(
            row["output"],
            text_type=row["text_type"],
            glossary_path=PROJECT_ROOT / "policy/glossary.yaml",
            allowed_terms=row.get("technical_terms"),
        )
        for row in positive
    ]
    adversarial_results = [
        score_text(
            row["output"],
            text_type=row["text_type"],
            glossary_path=PROJECT_ROOT / "policy/glossary.yaml",
        )
        for row in adversarial
    ]
    validation_results = [
        score_text(
            row["output"],
            text_type=row["text_type"],
            glossary_path=PROJECT_ROOT / "policy/glossary.yaml",
            allowed_terms=row.get("technical_terms"),
        )
        for row in validation
    ]
    elapsed = perf_counter() - started
    accepted = sum(result.strict_compliant for result in positive_results)
    rejected = sum(not result.strict_compliant for result in adversarial_results)
    validation_pass = sum(result.strict_compliant for result in validation_results)
    payload = {
        "positive_pass": accepted,
        "positive_total": len(positive),
        "positive_rate": round(accepted / len(positive), 6),
        "adversarial_rejected": rejected,
        "adversarial_total": len(adversarial),
        "adversarial_rejection_rate": round(rejected / len(adversarial), 6),
        "validation_pass": validation_pass,
        "validation_total": len(validation),
        "validation_rate": round(validation_pass / len(validation), 6),
        "score": round(accepted / len(positive), 6),
        "elapsed_seconds": round(elapsed, 6),
    }
    print(json.dumps(payload, indent=2))
    if rejected != len(adversarial):
        raise SystemExit("adversarial rejection regressed")


if __name__ == "__main__":
    main()
