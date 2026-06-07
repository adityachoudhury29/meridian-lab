"""Profile consistency / honeypot detection.

The spec plants ~80 honeypots with *subtly impossible* profiles ("8 years of
experience at a company founded 3 years ago"; "'expert' proficiency in 10 skills
with 0 years used"). They are forced to relevance tier 0; ranking them in the
top 100 hurts the honeypot-rate Stage-3 filter.

We do NOT special-case honeypots by id — we detect *internal contradictions*
that no real profile would contain. The bar is deliberately high (unambiguous
timeline / arithmetic impossibilities, or egregious mastery-without-usage
stuffing) so that ordinary noisy-but-plausible profiles are never flagged.
"""
from __future__ import annotations

from datetime import date

from . import config


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _parse(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def consistency_check(cand: dict, ref: date | None = None) -> tuple[bool, list[str]]:
    """Return ``(is_honeypot, reasons)``. ``is_honeypot`` is True only on an
    unambiguous internal contradiction."""
    if ref is None:
        ref = date.fromisoformat(config.FALLBACK_REFERENCE_DATE)

    reasons: list[str] = []
    p = cand.get("profile", {})
    yoe = float(p.get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12.0
    roles = cand.get("career_history", [])

    # --- timeline contradictions per role ---
    for r in roles:
        dur = r.get("duration_months", 0) or 0
        start = _parse(r.get("start_date"))
        end = _parse(r.get("end_date")) or (ref if r.get("is_current") else None)
        if start and end:
            if end < start:
                reasons.append(f"role '{r.get('title','?')}' ends before it starts")
            window = _months_between(start, end)
            # claims substantially more tenure than the calendar window allows
            if dur - window > 18:
                reasons.append(
                    f"role '{r.get('title','?')}' claims {dur} months but its dates span only ~{window}")
        # a single role longer than the entire stated career (+2y slack)
        if dur - yoe_months > 24:
            reasons.append(
                f"role '{r.get('title','?')}' is {dur} months but total experience is only {yoe:.1f}y")

    # earliest start implies far more career than stated YOE
    starts = [s for s in (_parse(r.get("start_date")) for r in roles) if s]
    if starts:
        span = _months_between(min(starts), ref)
        if span - yoe_months > 36:
            reasons.append(f"first role began ~{span} months ago but only {yoe:.1f}y experience stated")

    # --- education timeline ---
    for e in cand.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            reasons.append("education ends before it starts")

    # --- mastery-without-usage stuffing ---
    skills = cand.get("skills", [])
    adv_zero = sum(1 for s in skills
                   if s.get("proficiency") in ("expert", "advanced")
                   and (s.get("duration_months", 0) or 0) == 0)
    if adv_zero >= 5:
        reasons.append(f"{adv_zero} skills claimed expert/advanced with 0 months of usage")

    # skills used far longer than the person has worked (needs to be egregious + plural)
    skill_over = sum(1 for s in skills
                     if (s.get("duration_months", 0) or 0) - yoe_months > 24)
    if skill_over >= 4 and yoe < 5:
        reasons.append(f"{skill_over} skills used longer than the stated {yoe:.1f}y career")

    return (len(reasons) > 0, reasons)
