"""Small, dependency-free text utilities for normalisation and lexicon matching.

Used by feature extraction to detect *evidence* (phrases) in free text. We keep
this simple and transparent on purpose: every match is auditable, which is what
lets the reasoning layer cite specific facts without hallucinating.
"""
from __future__ import annotations

import re
from typing import Iterable

_WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Lower-case and collapse whitespace; pad with spaces so word-boundary-ish
    lexicon entries like ``" e5 "`` or ``"map "`` match reliably."""
    if not text:
        return " "
    t = text.lower().replace("/", " / ").replace("-", "-")
    t = _WS.sub(" ", t)
    return f" {t} "


def count_hits(text: str, phrases: Iterable[str]) -> int:
    """Number of distinct phrases from ``phrases`` that occur in ``text``.

    ``text`` is assumed already normalised (see :func:`normalize`).
    """
    return sum(1 for p in phrases if p in text)


def matched(text: str, phrases: Iterable[str]) -> list[str]:
    """Return the list of phrases that occur in ``text`` (already normalised)."""
    return [p.strip() for p in phrases if p in text]


def any_hit(text: str, phrases: Iterable[str]) -> bool:
    return any(p in text for p in phrases)


def saturating(count: int, k: float = 2.0) -> float:
    """Map a non-negative count to (0,1) with diminishing returns: count/(count+k).

    ``k`` controls how quickly evidence saturates — a couple of strong hits
    should already be convincing, so we don't reward keyword spamming linearly.
    """
    if count <= 0:
        return 0.0
    return count / (count + k)
