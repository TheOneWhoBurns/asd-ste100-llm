"""Convert deterministic STE findings into dense reinforcement-learning rewards."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from ste100.core.analyzer import analyze
from ste100.dictionary.engine import DictionaryEngine


ENGINE_REVISION = "sourdough-bread/asd-ste100-checker@e193ecdd66b09ce81b7c611f1c841efd8ba84cc7"

_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)?(?=\s|$)")
_CONTRACTION = re.compile(
    r"\b(?:[A-Za-z]+n['’]t|[A-Za-z]+['’](?:d|ll|m|re|s|ve))\b",
    re.IGNORECASE,
)

DEFAULT_RULE_WEIGHTS: dict[str, float] = {
    "HARNESS-EMPTY": 4.0,
    "HARNESS-NO-LEXICAL-CONTENT": 4.0,
    "HARNESS-ALL-CAPS-BYPASS": 3.0,
    "STE-VOCAB-FORBIDDEN": 3.0,
    "STE-VOCAB-UNAPPROVED": 1.5,
    "STE-SENTENCE-LENGTH": 2.0,
    "STE-PASSIVE": 2.0,
    "STE-IMPERATIVE": 2.0,
    "STE-VERB-FORM": 1.75,
    "STE-NOUN-CLUSTER": 1.5,
    "STE-POS-MISMATCH": 1.25,
    "STE-SEMICOLON": 1.5,
    "STE-CONTRACTION": 1.5,
    "STE-PARAGRAPH-LENGTH": 1.5,
    "STE-ONE-INSTRUCTION": 1.0,
    "STE-PRONOUN-AMBIG": 0.75,
    "STE-TOPIC-SENTENCE": 0.5,
    "STE-UNITS-FORMAT": 0.1,
}

_SEVERITY_MULTIPLIER = {
    "error": 1.0,
    "warning": 0.5,
    "info": 0.1,
}

_PROCEDURE_NOUN_VERB_EXCEPTIONS = frozenset({"start"})


@dataclass(frozen=True)
class RewardConfig:
    """Policy for converting findings into one bounded scalar."""

    strict: bool = True
    allow_all_caps: bool = False
    rule_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RULE_WEIGHTS)
    )
    unknown_rule_weight: float = 1.0


@dataclass(frozen=True)
class RewardResult:
    """Serializable checker and reward output for one model response."""

    text: str
    text_type: str
    compliant: bool
    strict_compliant: bool
    reward: float
    weighted_penalty: float
    sentence_count: int
    word_count: int
    summary: dict[str, int]
    rule_counts: dict[str, int]
    findings: list[dict[str, Any]]
    engine_revision: str = ENGINE_REVISION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surface_finding(
    *,
    rule_id: str,
    severity: str,
    message: str,
    start: int,
    end: int,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "start": start,
        "end": end,
        "sentence": None,
        "evidence": evidence or {},
        "suggestions": [],
    }


def _sentence_count(text: str) -> int:
    count = len(_SENTENCE_END.findall(text))
    if count:
        return count
    return 1 if text.strip() else 0


def _paragraph_sentence_counts(text: str) -> list[tuple[int, int, int]]:
    """Return (start, end, sentence_count) for non-empty paragraphs."""
    output: list[tuple[int, int, int]] = []
    cursor = 0
    for block in re.split(r"\n\s*\n", text):
        start = text.find(block, cursor)
        if start < 0:
            start = cursor
        end = start + len(block)
        cursor = end
        if block.strip():
            output.append((start, end, _sentence_count(block)))
    return output


def _surface_findings(
    text: str,
    *,
    text_type: str,
    allow_all_caps: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    stripped = text.strip()
    words = _WORD.findall(text)

    if not stripped:
        return [
            _surface_finding(
                rule_id="HARNESS-EMPTY",
                severity="error",
                message="The model output is empty.",
                start=0,
                end=0,
            )
        ]

    if not words:
        findings.append(
            _surface_finding(
                rule_id="HARNESS-NO-LEXICAL-CONTENT",
                severity="error",
                message="The model output has no English lexical content.",
                start=0,
                end=len(text),
            )
        )

    if not allow_all_caps and words:
        uppercase = sum(1 for word in words if word.isupper())
        if uppercase / len(words) >= 0.8:
            findings.append(
                _surface_finding(
                    rule_id="HARNESS-ALL-CAPS-BYPASS",
                    severity="error",
                    message=(
                        "Most words are uppercase. This can bypass dictionary checks "
                        "that allow uppercase labels."
                    ),
                    start=0,
                    end=len(text),
                    evidence={
                        "uppercase_words": uppercase,
                        "word_count": len(words),
                    },
                )
            )

    for match in re.finditer(";", text):
        findings.append(
            _surface_finding(
                rule_id="STE-SEMICOLON",
                severity="error",
                message="Do not use a semicolon (Rule 8.1).",
                start=match.start(),
                end=match.end(),
                evidence={"rule_ref": "Rule 8.1"},
            )
        )

    for match in _CONTRACTION.finditer(text):
        findings.append(
            _surface_finding(
                rule_id="STE-CONTRACTION",
                severity="error",
                message=f"Do not use the contraction '{match.group(0)}' (Rule 4.2).",
                start=match.start(),
                end=match.end(),
                evidence={"rule_ref": "Rule 4.2", "word": match.group(0)},
            )
        )

    if text_type == "description":
        for start, end, count in _paragraph_sentence_counts(text):
            if count > 6:
                findings.append(
                    _surface_finding(
                        rule_id="STE-PARAGRAPH-LENGTH",
                        severity="error",
                        message=(
                            f"This paragraph has {count} sentences. Use no more "
                            "than 6 sentences (Rule 6.6)."
                        ),
                        start=start,
                        end=end,
                        evidence={
                            "rule_ref": "Rule 6.6",
                            "sentence_count": count,
                            "limit": 6,
                        },
                    )
                )

    return findings


def _severity_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _allowed_term_tokens(terms: Iterable[str] | None) -> set[str]:
    if terms is None:
        return set()
    if isinstance(terms, str):
        terms = [terms]
    tokens: set[str] = set()
    for term in terms:
        if not isinstance(term, str):
            raise TypeError("each allowed technical term must be a string")
        for word in _WORD.findall(term):
            token = word.casefold()
            tokens.add(token)
            if token.endswith(("s", "x", "z", "ch", "sh")):
                tokens.add(token + "es")
            elif token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
                tokens.add(token[:-1] + "ies")
            else:
                tokens.add(token + "s")
    return tokens


def _is_allowed_technical_term(
    finding: Mapping[str, Any],
    allowed_tokens: set[str],
) -> bool:
    if finding.get("rule_id") != "STE-VOCAB-UNAPPROVED":
        return False
    evidence = finding.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    word = evidence.get("word")
    return isinstance(word, str) and word.casefold() in allowed_tokens


def _is_parser_false_positive(
    finding: Mapping[str, Any],
    text: str,
    text_type: str,
) -> bool:
    """Ignore two narrow spaCy/dictionary collisions seen in procedure text."""
    if finding.get("rule_id") != "STE-POS-MISMATCH":
        return False
    evidence = finding.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    word = evidence.get("word")
    if not isinstance(word, str):
        return False

    # A capitalized label after a command can make spaCy tag the command as a
    # proper-noun compound. The checker reports this as a low-confidence warning.
    if (
        text_type == "procedure"
        and evidence.get("confidence", 1.0) <= 0.5
        and "NNP/compound" in str(evidence.get("parse_cue", ""))
    ):
        return True

    # The controlled dictionary contains common command words as nouns, but a
    # procedure commonly uses them as imperative verbs. Limit this exception to
    # a sentence/list boundary so an ordinary noun in running prose is not
    # hidden.
    prefix = text[: int(finding.get("start", 0))].rstrip()
    at_command_boundary = not prefix or bool(re.search(r"[.!?]\s*$", prefix))
    return (
        text_type == "procedure"
        and (
            word.casefold() in _PROCEDURE_NOUN_VERB_EXCEPTIONS
            or at_command_boundary
        )
        and evidence.get("observed_pos") == "verb"
        and evidence.get("approved_pos") == "noun"
    )


@lru_cache(maxsize=16)
def _load_glossary_engine(
    path: str,
    modified_ns: int,
    size: int,
) -> DictionaryEngine:
    """Load and cache a glossary-aware dictionary.

    The metadata arguments invalidate the cache after a glossary edit. They
    otherwise exist only as cache-key material.
    """
    del modified_ns, size
    engine = DictionaryEngine().load()
    engine.merge_glossary(path)
    return engine


def _dictionary_for_glossary(path: str | Path) -> DictionaryEngine:
    glossary = Path(path).expanduser().absolute()
    metadata = glossary.stat()
    return _load_glossary_engine(
        str(glossary),
        metadata.st_mtime_ns,
        metadata.st_size,
    )


def score_text(
    text: str,
    *,
    text_type: str = "auto",
    glossary_path: str | Path | None = None,
    allowed_terms: Iterable[str] | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Check one output and return a bounded dense reward in [-1, 1].

    The style reward is intentionally separate from task correctness. A trainer
    must combine it with a task or meaning-preservation reward to prevent short,
    content-free outputs from becoming an optimum.
    """
    policy = config or RewardConfig()

    if text.strip():
        dictionary = (
            _dictionary_for_glossary(glossary_path) if glossary_path else None
        )
        analysis = analyze(
            text,
            text_type=text_type,
            dictionary=dictionary,
        )
        resolved_type = analysis.text_type.value
        findings = [finding.model_dump(mode="json") for finding in analysis.findings]
        allowed_tokens = _allowed_term_tokens(allowed_terms)
        if allowed_tokens:
            findings = [
                finding
                for finding in findings
                if not _is_allowed_technical_term(finding, allowed_tokens)
            ]
        findings = [
            finding
            for finding in findings
            if not _is_parser_false_positive(finding, text, resolved_type)
        ]
    else:
        analysis = None
        resolved_type = "description" if text_type == "auto" else text_type
        findings = []

    findings.extend(
        _surface_findings(
            text,
            text_type=resolved_type,
            allow_all_caps=policy.allow_all_caps,
        )
    )
    findings.sort(key=lambda item: (item["start"], item["end"], item["rule_id"]))

    rule_counts: dict[str, int] = {}
    summary = {"total": len(findings), "error": 0, "warning": 0, "info": 0}
    weighted_penalty = 0.0
    for finding in findings:
        rule_id = str(finding["rule_id"])
        severity = _severity_value(finding["severity"])
        rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        summary[severity] = summary.get(severity, 0) + 1
        base_weight = policy.rule_weights.get(rule_id, policy.unknown_rule_weight)
        weighted_penalty += base_weight * _SEVERITY_MULTIPLIER.get(severity, 1.0)

    sentences = _sentence_count(text)
    words = len(_WORD.findall(text))
    compliant = summary["error"] == 0
    strict_compliant = compliant and summary["warning"] == 0
    if policy.strict:
        selected_compliant = strict_compliant
    else:
        selected_compliant = compliant

    # Compliance is lexicographic: no output with a blocking finding can earn a
    # positive style reward. This prevents a long answer from diluting one
    # violation and looking better than a clean answer.
    if selected_compliant:
        reward = max(0.0, 1.0 - min(1.0, weighted_penalty / 2.0))
    else:
        blocking_penalty = 0.0
        for finding in findings:
            severity = _severity_value(finding["severity"])
            if severity == "error" or (policy.strict and severity == "warning"):
                rule_id = str(finding["rule_id"])
                base_weight = policy.rule_weights.get(
                    rule_id,
                    policy.unknown_rule_weight,
                )
                blocking_penalty += base_weight * _SEVERITY_MULTIPLIER.get(
                    severity,
                    1.0,
                )
        reward = -min(1.0, blocking_penalty / 2.0)

    if not text.strip() or words == 0:
        reward = -1.0

    if not math.isfinite(reward):
        raise RuntimeError("reward is not finite")

    return RewardResult(
        text=text,
        text_type=resolved_type,
        compliant=compliant,
        strict_compliant=strict_compliant,
        reward=round(reward, 6),
        weighted_penalty=round(weighted_penalty, 6),
        sentence_count=sentences,
        word_count=words,
        summary=summary,
        rule_counts=dict(sorted(rule_counts.items())),
        findings=findings,
    )


def score_batch(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_field: str = "output",
    default_text_type: str = "auto",
    glossary_path: str | Path | None = None,
    default_allowed_terms: Iterable[str] | None = None,
    config: RewardConfig | None = None,
) -> list[dict[str, Any]]:
    """Score JSON-like rows and preserve their original fields."""
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if output_field not in row:
            raise KeyError(f"row {index} does not contain output field {output_field!r}")
        text = row[output_field]
        if not isinstance(text, str):
            raise TypeError(f"row {index} field {output_field!r} must be a string")
        row_text_type = str(row.get("text_type", default_text_type))
        row_allowed_terms = row.get("technical_terms", default_allowed_terms)
        if isinstance(row_allowed_terms, str):
            row_allowed_terms = [row_allowed_terms]
        result = score_text(
            text,
            text_type=row_text_type,
            glossary_path=glossary_path,
            allowed_terms=row_allowed_terms,
            config=config,
        )
        merged = dict(row)
        merged["ste_reward"] = result.to_dict()
        scored.append(merged)
    return scored
