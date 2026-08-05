#!/usr/bin/env python3
"""Interactive local chat with ASD-style checking and repair attempts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from mlx_lm import generate, load
from ste100.core.analyzer import get_nlp

from ste_reward import score_text


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "You are a technical language assistant. Answer in concise ASD-STE100 style. "
    "Preserve the user's meaning. Use active voice and short sentences. "
    "Use one instruction per sentence. Do not use semicolons or contractions. "
    "Avoid vague pronouns such as it, this, and they when the noun is unclear. "
    "Return only the answer. Do not include analysis, labels, or commentary."
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_VISIBLE_REASONING = re.compile(
    r"\A\s*(?:thinking process|analysis|reasoning)\s*:.*?(?:\n\s*(?:final answer|answer)\s*:\s*)",
    re.IGNORECASE | re.DOTALL,
)
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def clean_response(text: str) -> str:
    """Remove reasoning blocks if a chat template exposes them in the answer."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    cleaned = _VISIBLE_REASONING.sub("", cleaned).strip()
    return cleaned


def prompt_terms(text: str) -> list[str]:
    """Preserve user-specific nouns and labels as task technical terms."""
    return _WORD.findall(text)


def task_terms(user_text: str, candidate: str) -> list[str]:
    """Keep user terms and newly introduced technical nouns available to STE."""
    terms = prompt_terms(user_text)
    for token in get_nlp()(candidate):
        if token.pos_ in {"NOUN", "PROPN"} and any(char.isalpha() for char in token.text):
            terms.append(token.text)
    return terms


def chat_score(
    candidate: str,
    user_text: str,
    *,
    text_type: str,
    glossary_path: Path,
):
    """Score chat text while allowing open-class terms introduced by the answer."""
    terms = task_terms(user_text, candidate)
    result = score_text(
        candidate,
        text_type=text_type,
        glossary_path=glossary_path,
        allowed_terms=terms,
    )
    generated_terms = [
        finding.get("evidence", {}).get("word")
        for finding in result.findings
        if finding.get("rule_id") == "STE-VOCAB-UNAPPROVED"
    ]
    generated_terms = [term for term in generated_terms if isinstance(term, str)]
    if generated_terms:
        result = score_text(
            candidate,
            text_type=text_type,
            glossary_path=glossary_path,
            allowed_terms=[*terms, *generated_terms],
        )
    return result


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_once(model, tokenizer, messages: list[dict[str, str]], max_tokens: int) -> str:
    response = generate(
        model,
        tokenizer,
        prompt=render_prompt(tokenizer, messages),
        max_tokens=max_tokens,
        verbose=False,
    )
    return clean_response(response)


def answer(
    model,
    tokenizer,
    user_text: str,
    *,
    text_type: str,
    glossary_path: Path,
    max_tokens: int,
    repair_attempts: int,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, object]:
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": user_text},
    ]
    candidate = generate_once(model, tokenizer, base, max_tokens)
    result = chat_score(
        candidate,
        user_text,
        text_type=text_type,
        glossary_path=glossary_path,
    )
    best = (candidate, result)

    for _ in range(repair_attempts):
        if result.strict_compliant:
            break
        findings = "\n".join(
            f"- {finding['message']}"
            for finding in result.findings[:12]
        )
        repair = [
            *base,
            {"role": "assistant", "content": candidate},
            {
                "role": "user",
                "content": (
                    "Rewrite your answer. Preserve all technical meaning. "
                    "Fix these style findings:\n"
                    f"{findings}\n"
                    "Return only the corrected answer."
                ),
            },
        ]
        candidate = generate_once(model, tokenizer, repair, max_tokens)
        result = chat_score(
            candidate,
            user_text,
            text_type=text_type,
            glossary_path=glossary_path,
        )
        if result.strict_compliant or result.reward > best[1].reward:
            best = (candidate, result)

    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument(
        "--text-type",
        choices=("auto", "procedure", "description"),
        default="auto",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=ROOT / "policy" / "glossary.yaml",
    )
    args = parser.parse_args()

    lock = (ROOT / "model.lock.json").read_text(encoding="utf-8")
    import json

    model_lock = json.loads(lock)
    model_path = ROOT / model_lock["destination"]
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    print("Ready. Type /quit to exit.")
    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.casefold() in {"/quit", "/exit", "quit", "exit"}:
            break
        response, result = answer(
            model,
            tokenizer,
            user_text,
            text_type=args.text_type,
            glossary_path=args.glossary,
            max_tokens=args.max_tokens,
            repair_attempts=args.repair_attempts,
            history=history,
        )
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response},
            ]
        )
        print(f"\nModel> {response}")
        if not result.strict_compliant:
            print(
                f"[checker: reward={result.reward:+.2f}; "
                f"rules={','.join(result.rule_counts) or 'none'}]"
            )


if __name__ == "__main__":
    main()
