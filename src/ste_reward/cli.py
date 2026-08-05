"""Command-line interface for the STE reward harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from ste_reward.scorer import RewardConfig, score_batch, score_text


def _read_text(argument: str | None) -> str:
    if argument is not None:
        return argument
    return sys.stdin.read()


def _open_input(value: str) -> TextIO:
    return sys.stdin if value == "-" else Path(value).open(encoding="utf-8")


def _open_output(value: str) -> TextIO:
    return sys.stdout if value == "-" else Path(value).open("w", encoding="utf-8")


def _config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        strict=args.strict,
        allow_all_caps=args.allow_all_caps,
    )


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text-type",
        choices=("auto", "procedure", "description"),
        default="auto",
    )
    parser.add_argument("--glossary", type=Path)
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Allow one project technical term. Repeat this option as necessary.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat warnings as failure for the compliance gate.",
    )
    parser.add_argument(
        "--allow-all-caps",
        action="store_true",
        help="Allow mostly uppercase output. Disabled by default to prevent reward bypass.",
    )


def _check(args: argparse.Namespace) -> int:
    result = score_text(
        _read_text(args.text),
        text_type=args.text_type,
        glossary_path=args.glossary,
        allowed_terms=args.term,
        config=_config(args),
    )
    json.dump(result.to_dict(), sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    passed = result.strict_compliant if args.strict else result.compliant
    return 0 if passed else 1


def _batch(args: argparse.Namespace) -> int:
    input_stream = _open_input(args.input)
    output_stream = _open_output(args.output)
    close_input = input_stream is not sys.stdin
    close_output = output_stream is not sys.stdout
    try:
        rows = [
            json.loads(line)
            for line in input_stream
            if line.strip()
        ]
        results = score_batch(
            rows,
            output_field=args.output_field,
            default_text_type=args.text_type,
            glossary_path=args.glossary,
            default_allowed_terms=args.term,
            config=_config(args),
        )
        for result in results:
            output_stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    finally:
        if close_input:
            input_stream.close()
        if close_output:
            output_stream.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ste-reward")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Score one model output.")
    check.add_argument("--text", help="Text to check. Reads stdin when omitted.")
    check.add_argument("--pretty", action="store_true")
    _add_shared_options(check)
    check.set_defaults(handler=_check)

    batch = subparsers.add_parser("batch", help="Score JSONL model outputs.")
    batch.add_argument("--input", default="-", help="JSONL path or '-' for stdin.")
    batch.add_argument("--output", default="-", help="JSONL path or '-' for stdout.")
    batch.add_argument("--output-field", default="output")
    _add_shared_options(batch)
    batch.set_defaults(handler=_batch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
