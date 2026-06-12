"""Lightweight, deterministic prompt-injection detector for input guardrails.

Adapted from "Formalizing and Benchmarking Prompt Injection Attacks and
Defenses" (Liu et al., https://arxiv.org/abs/2310.12815). That paper unifies
the scattered prompt-injection literature into a small taxonomy of attack
components: naive instruction injection, *escape characters*, *context
ignoring*, *fake completion*, and the *combined attack* that stacks them.

We do not reproduce the paper's LLM-based known-answer detection (which needs a
model call). Instead we deliver the paper's core insight as a pure, offline
string-in / struct-out detector keyed on that taxonomy. It drops straight into
the SDK's ``@input_guardrail`` contract (string in, ``tripwire_triggered`` out)
and runs with no network access, so it is cheap to call on every turn and easy
to test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Each taxonomy entry maps an attack component from the paper to the regexes
# that betray it in a raw user/data string. Patterns are intentionally
# conservative so that benign questions ("ignore the typo above") rarely trip,
# while the canonical attack phrasings are caught.
_ATTACK_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    # "Context ignoring": text that tells the model to discard prior context.
    "context_ignoring": (
        re.compile(
            r"\b(?:ignore|disregard|forget|overlook)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|preceding|all|any)\b"
            r"[^.\n]{0,40}\b(?:instruction|instructions|prompt|prompts|"
            r"context|message|messages|direction|directions|rule|rules)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:do not|don't|never)\b[^.\n]{0,20}"
            r"\b(?:follow|obey|listen to)\b[^.\n]{0,30}"
            r"\b(?:previous|prior|above|system)\b",
            re.IGNORECASE,
        ),
    ),
    # "Fake completion": the attacker fabricates an answer/closing marker so the
    # model believes the legitimate task is finished before the injected one.
    "fake_completion": (
        re.compile(
            r"\b(?:task|request|job|instruction)\b[^.\n]{0,20}"
            r"\b(?:completed|complete|done|finished|accomplished)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|\n)\s*(?:answer|response|result|output)\s*:\s*\S",
            re.IGNORECASE,
        ),
    ),
    # "Escape characters": literal control-character sequences used to break out
    # of the surrounding data context (e.g. a literal "\n" smuggled in text).
    "escape_characters": (re.compile(r"(?:\\n|\\r|\\t){1,}"),),
    # Injected role / chat-template markers that impersonate the system or a new
    # turn, a common carrier for the components above.
    "role_impersonation": (
        re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*:", re.IGNORECASE),
        re.compile(
            r"<\|im_(?:start|end)\|>|\[/?INST\]|###\s*(?:instruction|system)", re.IGNORECASE
        ),
    ),
    # Naive instruction override: an explicit "new task / instead do X" cue.
    "instruction_override": (
        re.compile(
            r"\b(?:new|updated|real|actual)\b[^.\n]{0,15}"
            r"\b(?:instruction|instructions|task|prompt|goal)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\binstead\b[^.\n]{0,20}\b(?:do|say|print|output|reply|respond|tell|send|reveal)\b",
            re.IGNORECASE,
        ),
    ),
}

_TOTAL_CATEGORIES = len(_ATTACK_PATTERNS)


@dataclass(frozen=True)
class InjectionDetection:
    """Structured result of a prompt-injection scan.

    Attributes:
        detected: Whether the tripwire should fire for this input.
        attack_types: The taxonomy categories that matched, sorted.
        matched_patterns: A short human-readable snippet per matched category.
        score: Fraction of taxonomy categories that matched, in ``[0, 1]``.
        combined: True when two or more categories matched. The paper singles
            out the "combined attack" as the strongest variant, so this flag
            lets callers escalate on stacked components.
    """

    detected: bool
    attack_types: tuple[str, ...]
    matched_patterns: tuple[str, ...]
    score: float
    combined: bool


def extract_input_text(input: str | Iterable[Any]) -> str:
    """Normalize a guardrail ``input`` into a single scannable string.

    Accepts either a raw string or the SDK's ``list[TResponseInputItem]`` shape
    (dict-like items with a ``content`` field, which may itself be a list of
    content parts). Anything we cannot interpret is coerced via ``str``.
    """

    if isinstance(input, str):
        return input

    parts: list[str] = []
    for item in input:
        if isinstance(item, str):
            parts.append(item)
            continue
        content = item.get("content") if isinstance(item, dict) else None
        if content is None:
            parts.append(str(item))
        elif isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Iterable):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
        else:
            parts.append(str(content))
    return "\n".join(parts)


def detect_prompt_injection(
    text: str,
    *,
    min_attack_types: int = 1,
) -> InjectionDetection:
    """Scan ``text`` for the prompt-injection components formalized in the paper.

    Args:
        text: The candidate user/data string to inspect.
        min_attack_types: How many distinct taxonomy categories must match
            before the tripwire fires. Raise this to trade recall for precision
            (e.g. ``2`` only flags the paper's "combined attack").

    Returns:
        An :class:`InjectionDetection` describing what was found.
    """

    matched_types: list[str] = []
    snippets: list[str] = []
    for attack_type, patterns in _ATTACK_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                matched_types.append(attack_type)
                snippets.append(match.group(0).strip())
                break

    matched_types.sort()
    score = len(matched_types) / _TOTAL_CATEGORIES
    return InjectionDetection(
        detected=len(matched_types) >= min_attack_types,
        attack_types=tuple(matched_types),
        matched_patterns=tuple(snippets),
        score=score,
        combined=len(matched_types) >= 2,
    )
