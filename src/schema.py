"""Candidate I/O and the canonical 'career narrative' text builder.

Candidates are kept as plain dicts (cheaper than dataclasses across 100K
records). This module provides streaming loaders and the single source of truth
for the *text we embed*.

Design note — what we embed matters: we build the dense narrative from the
headline, summary and per-role *descriptions* only. We deliberately EXCLUDE the
raw skills list. Keyword-stuffers inflate the skills array, but their actual
career narrative is about marketing / HR / accounting — so embedding the
narrative (not the skill bag) makes the dense signal naturally robust to the
dataset's keyword-stuffing trap.
"""
from __future__ import annotations

import gzip
import json
from datetime import date
from typing import Iterator

from . import config


def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """Yield candidate dicts from a (optionally gzipped) JSONL file."""
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: str) -> list[dict]:
    return list(iter_candidates(path))


def reference_date(candidates: list[dict]) -> date:
    """The deterministic 'now' for recency = max last_active_date in the pool."""
    mx = ""
    for c in candidates:
        la = c.get("redrob_signals", {}).get("last_active_date", "")
        if la > mx:
            mx = la
    try:
        return date.fromisoformat(mx) if mx else date.fromisoformat(config.FALLBACK_REFERENCE_DATE)
    except ValueError:
        return date.fromisoformat(config.FALLBACK_REFERENCE_DATE)


def career_narrative(cand: dict) -> str:
    """The text we embed for the dense component (see module docstring)."""
    p = cand.get("profile", {})
    parts: list[str] = []
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])
    for role in cand.get("career_history", []):
        title = role.get("title", "")
        company = role.get("company", "")
        industry = role.get("industry", "")
        desc = role.get("description", "")
        header = f"{title} at {company} ({industry})".strip()
        parts.append(f"{header}. {desc}".strip())
    return "  ".join(parts)


def full_text(cand: dict) -> str:
    """Everything as lexical-search fodder (narrative + skills + certs). Used for
    *evidence* detection, not for the dense embedding."""
    parts = [career_narrative(cand)]
    for s in cand.get("skills", []):
        parts.append(s.get("name", ""))
    for c in cand.get("certifications", []):
        parts.append(c.get("name", ""))
        parts.append(c.get("issuer", ""))
    return "  ".join(p for p in parts if p)
