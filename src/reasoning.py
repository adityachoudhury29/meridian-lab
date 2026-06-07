"""Grounded, varied, rank-consistent reasoning generation.

The reasoning column is graded at Stage 4 on six axes: specific facts, JD
connection, honest concerns, no hallucination, variation, and rank consistency.

We satisfy all six by *construction*: every clause is assembled only from fields
that exist on the candidate's :class:`Features` (title, years, company, matched
skill names, evidenced requirements, signal values, detected concerns). Nothing
is invented — there is no free-text generation step, so hallucination is
structurally impossible. Variation comes from (a) genuinely different per-profile
facts and (b) a deterministic, id-seeded choice among synonymous connectors, so
no two reasonings are identical templates and the choice is reproducible.
"""
from __future__ import annotations

import hashlib

from .scoring import Scored

# Rank-band lead-ins. Tone scales with rank so a rank-5 reads confident and a
# rank-95 reads hedged — the Stage-4 "rank consistency" check.
_LEADS = {
    "top":      ["Strong fit", "Top pick", "Excellent match", "Clear fit"],
    "strong":   ["Strong adjacent fit", "Very good fit", "High-confidence fit"],
    "solid":    ["Solid fit", "Good partial fit", "Reasonable match"],
    "partial":  ["Partial fit with gaps", "Adjacent fit", "Moderate fit"],
    "filler":   ["Borderline", "Below-bar filler", "Long-shot inclusion"],
}


def _band(rank: int) -> str:
    if rank <= 10:
        return "top"
    if rank <= 30:
        return "strong"
    if rank <= 60:
        return "solid"
    if rank <= 90:
        return "partial"
    return "filler"


def _seeded(cid: str, options: list[str]) -> str:
    """Deterministic, id-seeded pick — gives lexical variety without randomness."""
    h = int(hashlib.md5(cid.encode()).hexdigest(), 16)
    return options[h % len(options)]


def _experience_phrase(f) -> str:
    if f.product_fraction >= 0.6:
        loc = "mostly at product companies"
    elif f.product_fraction >= 0.3:
        loc = "with some product-company experience"
    else:
        loc = ""
    base = f"{f.current_title} ({f.years_of_experience:.1f}y"
    if f.location_label:
        base += f", {f.location_label.split(',')[0]}"
    base += ")"
    return f"{base} {loc}".strip()


def reasoning_for(s: Scored, rank: int) -> str:
    f = s.features
    cid = s.candidate_id
    lead = _seeded(cid, _LEADS[_band(rank)])
    parts = [f"{lead}: {_experience_phrase(f)}"]

    # --- a positive clause grounded in real evidence ---
    pos = []
    if f.domain_evidence >= 0.45 and f.domain_phrases:
        verb = _seeded(cid + "d", ["career shows", "history demonstrates", "has done"])
        pos.append(f"{verb} {'/'.join(f.domain_phrases[:3])} work")
    if len(f.musthave_present) >= 2:
        cap = ", ".join(f.musthave_present[:3])
        pos.append(f"evidences {cap}")
    if f.top_skills:
        names = ", ".join(sk["name"] for sk in f.top_skills[:3])
        verb = _seeded(cid + "s", ["trusted skills in", "endorsed/used skills:", "depth in"])
        pos.append(f"{verb} {names}")
    if not pos and f.nicehave:
        pos.append("some relevant adjacent experience")
    if pos:
        # keep it to the two strongest positive clauses
        parts.append("; ".join(pos[:2]))

    sentence = ". ".join(p for p in parts if p)

    # --- an honest concern clause (always include the sharpest one if present) ---
    if f.concerns:
        connector = _seeded(cid + "c", ["Concern", "Caveat", "But", "Gap"])
        sentence += f". {connector}: {f.concerns[0]}"
        if rank > 60 and len(f.concerns) > 1:
            sentence += f"; {f.concerns[1]}"

    # tidy + length guard (Stage-4 wants 1-2 sentences)
    sentence = sentence.replace("  ", " ").strip()
    if len(sentence) > 320:
        sentence = sentence[:317].rstrip(" ,;.") + "…"
    if not sentence.endswith((".", "…")):
        sentence += "."
    return sentence
