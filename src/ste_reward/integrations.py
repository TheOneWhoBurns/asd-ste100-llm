"""Trainer-facing adapters for raw strings and chat completion payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ste_reward.scorer import RewardConfig, score_text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content,
        (str, bytes, bytearray),
    ):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise TypeError("completion content must be text or a sequence of text parts")


def completion_text(completion: Any) -> str:
    """Extract assistant text from a raw or chat-shaped completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        if "content" in completion:
            return _content_text(completion["content"])
        if "text" in completion and isinstance(completion["text"], str):
            return completion["text"]
    if isinstance(completion, Sequence) and not isinstance(
        completion,
        (str, bytes, bytearray),
    ):
        for message in reversed(completion):
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "assistant" and "content" in message:
                return _content_text(message["content"])
        if completion and isinstance(completion[-1], Mapping):
            last = completion[-1]
            if "content" in last:
                return _content_text(last["content"])
    raise TypeError("completion must contain assistant text")


@dataclass(frozen=True)
class STERewardFunction:
    """Picklable batch reward function for reinforcement-learning trainers."""

    text_type: str = "auto"
    glossary_path: str | Path | None = None
    config: RewardConfig = field(default_factory=RewardConfig)

    def __call__(
        self,
        completions: Sequence[Any],
        **columns: Any,
    ) -> list[float]:
        row_types = columns.get("text_type")
        if row_types is None:
            text_types = [self.text_type] * len(completions)
        elif isinstance(row_types, str):
            text_types = [row_types] * len(completions)
        else:
            text_types = list(row_types)
            if len(text_types) != len(completions):
                raise ValueError(
                    "text_type column length must match the completions batch"
                )

        row_terms = columns.get("technical_terms")
        if row_terms is None:
            technical_terms: list[Any] = [None] * len(completions)
        else:
            technical_terms = list(row_terms)
            if len(technical_terms) != len(completions):
                raise ValueError(
                    "technical_terms column length must match the completions batch"
                )

        return [
            score_text(
                completion_text(completion),
                text_type=str(row_text_type),
                glossary_path=self.glossary_path,
                allowed_terms=(
                    [row_allowed_terms]
                    if isinstance(row_allowed_terms, str)
                    else row_allowed_terms
                ),
                config=self.config,
            ).reward
            for completion, row_text_type, row_allowed_terms in zip(
                completions,
                text_types,
                technical_terms,
                strict=True,
            )
        ]
