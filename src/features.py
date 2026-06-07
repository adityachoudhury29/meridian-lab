"""Feature extraction: raw candidate profile -> recruiter-legible signals.

Each candidate becomes a :class:`Features` object holding (a) normalised 0..1
component sub-scores, (b) multiplicative modifiers (experience, location,
behavioural availability, disqualifier penalties, consistency), and (c) the raw
facts + matched-evidence the reasoning layer cites. Nothing here is stochastic;
every signal is auditable, which is what lets us generate grounded, non-
hallucinated reasoning downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import config, text as T
from .jd import JobRequirements
from .schema import career_narrative
from .validation import consistency_check


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _seniority_rank(title: str) -> int:
    t = title.lower()
    if any(k in t for k in ("principal", "staff", "director", "vp", "head of", "distinguished")):
        return 4
    if "lead" in t or "senior" in t or "sr." in t or "sr " in t:
        return 3
    if "junior" in t or "jr" in t or "associate" in t or "intern" in t or "trainee" in t:
        return 1
    return 2


def _experience_fit(yoe: float) -> float:
    lo_p, hi_p = config.EXPERIENCE_BAND["plateau"]
    floor = config.EXPERIENCE_BAND["floor"]
    if lo_p <= yoe <= hi_p:
        return 1.0
    if yoe < lo_p:
        # 5->0.85, 4->0.6, 3->0.4, 2->0.28, <=1 -> floor
        pts = [(lo_p, 1.0), (5.0, 0.85), (4.0, 0.6), (3.0, 0.42), (2.0, 0.3), (1.0, floor), (0.0, floor)]
    else:
        # 9->0.85, 10->0.7, 11->0.55, 13->0.38, >=15 -> floor
        pts = [(hi_p, 1.0), (9.0, 0.85), (10.0, 0.72), (11.0, 0.58), (13.0, 0.4), (15.0, floor), (50.0, floor)]
    pts.sort()
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= yoe <= x1:
            if x1 == x0:
                return y0
            r = (yoe - x0) / (x1 - x0)
            return y0 + r * (y1 - y0)
    return floor


def _education_score(education: list[dict]) -> float:
    table = {"tier_1": 1.0, "tier_2": 0.78, "tier_3": 0.55, "tier_4": 0.4, "unknown": 0.45}
    if not education:
        return 0.4
    return max(table.get(e.get("tier", "unknown"), 0.45) for e in education)


def _skill_trust(skill: dict) -> float:
    """Trust in a *claimed* skill from endorsements, duration and proficiency
    consistency. 'expert'/'advanced' with 0 months used is the canonical
    keyword-stuffing tell and is trust-near-zero."""
    end = skill.get("endorsements", 0) or 0
    dur = skill.get("duration_months", 0) or 0
    prof = skill.get("proficiency", "intermediate")
    trust = 0.15 + 0.35 * min(end / 20.0, 1.0) + 0.35 * min(dur / 24.0, 1.0)
    if prof in ("expert", "advanced") and dur == 0:
        trust *= 0.25            # claimed mastery with zero usage -> deeply suspect
    if prof in ("expert", "advanced") and end == 0 and dur < 6:
        trust *= 0.5
    return max(0.0, min(1.0, trust))


def _location_bucket(profile: dict, willing_relocate: bool) -> str:
    country = (profile.get("country") or "").strip().lower()
    city = (profile.get("location") or "").split(",")[0].strip().lower()
    if country == "india":
        if city in config.NOIDA_PUNE:
            return "noida_pune"
        if city in config.NCR_METROS:
            return "ncr_metro"
        return "other_india"
    return "abroad_relocate" if willing_relocate else "abroad_static"


def _behavioral_modifier(sig: dict, ref: date) -> tuple[float, int]:
    """Availability modifier + days-since-active. A perfect-on-paper candidate who
    is dormant and unresponsive is, per the JD, not actually hireable."""
    # recency
    days = 9999
    la = sig.get("last_active_date")
    if la:
        try:
            days = (ref - date.fromisoformat(la)).days
        except ValueError:
            pass
    if days <= 30:
        recency = 1.0
    elif days <= 90:
        recency = 0.95
    elif days <= 180:
        recency = 0.85
    elif days <= 270:
        recency = 0.72
    else:
        recency = 0.6

    resp = sig.get("recruiter_response_rate", 0.4) or 0.0
    resp_f = 0.6 + 0.4 * max(0.0, min(1.0, resp))
    open_f = 1.05 if sig.get("open_to_work_flag") else 0.95
    icr = sig.get("interview_completion_rate", 0.5)
    icr = 0.5 if icr is None else icr
    icr_f = 0.92 + 0.13 * max(0.0, min(1.0, icr))
    comp = sig.get("profile_completeness_score", 50) or 0
    comp_f = 0.93 + 0.07 * min(comp / 100.0, 1.0)

    mod = recency * resp_f * open_f * icr_f * comp_f
    lo, hi = config.BEHAVIORAL_BOUNDS
    return max(lo, min(hi, mod)), days


# --------------------------------------------------------------------------- #
# the feature object
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    candidate_id: str
    # component sub-scores (0..1)
    title_fit: float
    domain_evidence: float
    musthave: float
    nicehave: float
    experience_fit: float
    education: float
    # multipliers
    location_modifier: float
    behavioral_modifier: float
    disqualifier_multiplier: float
    consistency_multiplier: float
    # the four hard reqs, evidenced or missing
    musthave_present: list[str]
    musthave_missing: list[str]
    # raw facts (for reasoning)
    years_of_experience: float
    current_title: str
    current_company: str
    location_label: str
    location_bucket: str
    top_skills: list[dict]            # trusted, relevant skills [{name,proficiency,trust}]
    ai_core_skill_count: int
    domain_phrases: list[str]
    product_fraction: float
    last_active_days: int
    response_rate: float
    notice_period_days: int
    open_to_work: bool
    github_activity_score: float
    # diagnostics
    disqualifiers: list[str]
    concerns: list[str]
    strengths: list[str]
    is_honeypot: bool
    honeypot_reasons: list[str] = field(default_factory=list)


def extract_features(cand: dict, ref: date, jd: JobRequirements) -> Features:
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    roles = cand.get("career_history", [])
    skills = cand.get("skills", [])
    yoe = float(p.get("years_of_experience", 0) or 0)

    narrative = T.normalize(career_narrative(cand))
    skills_text = T.normalize("  ".join(s.get("name", "") for s in skills))
    blob = narrative + skills_text

    # ---- titles ----
    cur_title = p.get("current_title", "")
    titles = [r.get("title", "") for r in roles] + [cur_title]
    tl = [t.lower() for t in titles]
    strong_now = T.any_hit(T.normalize(cur_title), config.STRONG_AI_TITLE_TERMS)
    strong_ever = any(T.any_hit(T.normalize(t), config.STRONG_AI_TITLE_TERMS) for t in titles)
    adjacent_ever = any(T.any_hit(T.normalize(t), config.ADJACENT_TITLE_TERMS) for t in titles)
    nontech_now = cur_title.strip().lower() in config.NONTECH_TITLES
    if strong_now:
        title_fit = 1.0
    elif strong_ever:
        title_fit = 0.72
    elif nontech_now:
        title_fit = 0.05
    elif adjacent_ever:
        title_fit = 0.45
    else:
        title_fit = 0.18

    # ---- skill trust over relevant skills ----
    relevant_skills = []
    for s in skills:
        nm = s.get("name", "").lower()
        is_core = nm in config.AI_CORE_SKILLS
        is_domain = T.any_hit(f" {nm} ", config.DOMAIN_CORE)
        if is_core or is_domain:
            relevant_skills.append({
                "name": s.get("name", ""), "proficiency": s.get("proficiency", ""),
                "endorsements": s.get("endorsements", 0), "duration_months": s.get("duration_months", 0),
                "trust": _skill_trust(s),
            })
    relevant_skills.sort(key=lambda x: x["trust"], reverse=True)
    avg_skill_trust = (sum(s["trust"] for s in relevant_skills) / len(relevant_skills)) if relevant_skills else 0.4
    ai_core_skill_count = sum(1 for s in skills if s.get("name", "").lower() in config.AI_CORE_SKILLS)

    # ---- domain evidence (career narrative is hard to fake; skills are trust-discounted) ----
    core_career = T.count_hits(narrative, config.DOMAIN_CORE)
    core_skill = T.count_hits(skills_text, config.DOMAIN_CORE)
    domain_phrases = T.matched(narrative, config.DOMAIN_CORE)[:6]
    domain_evidence = T.saturating(core_career * 1.0 + core_skill * 0.4 * avg_skill_trust, k=2.5)

    # ---- must-have coverage (career evidence full, skill evidence trust-discounted) ----
    mh_weights = {"embeddings_retrieval": 0.30, "vector_db_hybrid": 0.30,
                  "ranking_eval": 0.25, "strong_python": 0.15}
    mh_label = {"embeddings_retrieval": "embeddings-based retrieval",
                "vector_db_hybrid": "vector DB / hybrid search",
                "ranking_eval": "ranking evaluation (NDCG/MRR/MAP/A-B)",
                "strong_python": "strong Python"}
    musthave = 0.0
    present, missing = [], []
    tech_titled = strong_now or strong_ever or adjacent_ever
    for key, lex in config.MUSTHAVE_LEXICONS.items():
        in_career = T.any_hit(narrative, lex)
        in_skill = T.any_hit(skills_text, lex)
        cov = 1.0 if in_career else (0.55 * avg_skill_trust if in_skill else 0.0)
        # Python is a given for any genuine engineering/ML role — don't treat its
        # literal absence from the text as a gap (it reads as not understanding the
        # candidate). Only the *rare, discriminating* must-haves should drive concerns.
        if key == "strong_python" and cov < 0.9 and (tech_titled or domain_evidence >= 0.3):
            cov = max(cov, 0.9)
        musthave += mh_weights[key] * cov
        (present if cov >= 0.5 else missing).append(mh_label[key])

    # ---- nice-to-haves ----
    nice_matched = []
    for key, lex in config.NICEHAVE_LEXICONS.items():
        if T.any_hit(blob, lex):
            nice_matched.append(key)
    nicehave = min(len(nice_matched) / 3.0, 1.0)

    experience_fit = _experience_fit(yoe)
    education = _education_score(cand.get("education", []))

    # ---- product vs services / consulting ----
    n_roles = max(1, len(roles))
    product_roles = services_roles = consulting_named = 0
    for r in roles:
        ind = (r.get("industry") or "").strip().lower()
        comp = (r.get("company") or "").strip().lower()
        named = any(f in comp for f in config.CONSULTING_FIRMS)
        if named:
            consulting_named += 1
        if ind in config.PRODUCT_INDUSTRIES:
            product_roles += 1
        elif ind in config.SERVICES_INDUSTRIES or named:
            services_roles += 1
    product_fraction = product_roles / n_roles
    has_product = product_roles > 0
    consulting_only = (not has_product) and (services_roles + consulting_named) >= n_roles and n_roles >= 1 \
        and (services_roles >= 1 or consulting_named >= 1)

    # ---- NLP/IR vs CV/speech/robotics ----
    nlp_hits = T.count_hits(blob, config.NLP_IR_TERMS)
    cv_hits = T.count_hits(blob, config.CV_SPEECH_ROBOTICS_TERMS)
    cv_primary = cv_hits >= 3 and cv_hits > (nlp_hits * 1.5 + 1) and nlp_hits < 2

    # ---- research-only / framework / recency disqualifiers ----
    research_hits = T.count_hits(blob, config.RESEARCH_TERMS)
    prod_hits = T.count_hits(blob, config.PRODUCTION_TERMS)
    pure_research = research_hits >= 2 and prod_hits == 0 and not has_product
    framework_hits = T.count_hits(blob, config.FRAMEWORK_WRAPPER_TERMS)
    framework_enthusiast = framework_hits >= 3 and domain_evidence < 0.28 and not T.any_hit(narrative, config.MUSTHAVE_LEXICONS["ranking_eval"])
    ai_recent_langchain = (yoe < 4.0 and framework_hits >= 2 and core_career == 0
                           and not T.any_hit(narrative, config.MUSTHAVE_LEXICONS["embeddings_retrieval"]))

    # ---- no production code in 18 months (moved to architecture / management) ----
    mgmt_terms = ("architect", "engineering manager", "tech lead", "director", "vp ",
                  "head of", "principal architect")
    cur_role = next((r for r in roles if r.get("is_current")), None)
    no_prod_code = (any(m in cur_title.lower() for m in mgmt_terms)
                    and cur_role is not None and (cur_role.get("duration_months", 0) or 0) >= 18)

    # ---- title-chaser ----
    dated = [r for r in roles if r.get("start_date")]
    dated.sort(key=lambda r: r.get("start_date", ""))
    title_chaser = False
    if len(dated) >= 3:
        durs = [r.get("duration_months", 0) or 0 for r in dated]
        median_dur = sorted(durs)[len(durs) // 2]
        distinct_co = len({(r.get("company") or "").lower() for r in dated})
        ranks = [_seniority_rank(r.get("title", "")) for r in dated]
        ascending = ranks[-1] > ranks[0] and all(b >= a for a, b in zip(ranks, ranks[1:]))
        title_chaser = median_dur <= 18 and distinct_co >= 3 and ascending

    # the dataset's signature keyword-stuffer trap
    keyword_stuffer_mismatch = nontech_now and core_career == 0 and ai_core_skill_count >= 4

    disq = []
    if keyword_stuffer_mismatch:
        disq.append("keyword_stuffer_mismatch")
    if consulting_only:
        disq.append("consulting_only_no_product")
    if pure_research:
        disq.append("pure_research_no_production")
    if ai_recent_langchain:
        disq.append("ai_is_recent_langchain_only")
    if no_prod_code:
        disq.append("no_production_code_18mo")
    if title_chaser:
        disq.append("title_chaser")
    if framework_enthusiast:
        disq.append("framework_enthusiast")
    if cv_primary:
        disq.append("cv_speech_robotics_primary")
    disq_mult = 1.0
    for d in disq:
        disq_mult *= config.DISQUALIFIER_MULTIPLIERS.get(d, 1.0)
    disq_mult = max(config.DISQUALIFIER_FLOOR, disq_mult)

    # ---- location & behaviour ----
    willing = bool(sig.get("willing_to_relocate"))
    loc_bucket = _location_bucket(p, willing)
    location_modifier = config.LOCATION_MODIFIER[loc_bucket]
    beh_mod, last_active_days = _behavioral_modifier(sig, ref)

    # ---- consistency / honeypot ----
    is_hp, hp_reasons = consistency_check(cand, ref)
    consistency_multiplier = config.CONSISTENCY_HONEYPOT if is_hp else 1.0

    # ---- diagnostics for reasoning ----
    loc_label = p.get("location", "")
    notice = sig.get("notice_period_days", 0) or 0
    resp = sig.get("recruiter_response_rate", 0.0) or 0.0
    gh = sig.get("github_activity_score", -1)

    strengths, concerns = [], []
    if strong_now:
        strengths.append(f"current role is {cur_title}")
    if domain_evidence >= 0.5 and domain_phrases:
        strengths.append("career history shows " + "/".join(domain_phrases[:3]) + " work")
    if product_fraction >= 0.5:
        strengths.append("product-company experience")
    if location_modifier >= 1.02:
        strengths.append(f"based in {loc_label}")
    if len(present) >= 3:
        strengths.append(f"{len(present)}/4 hard requirements evidenced")

    if nontech_now and ai_core_skill_count >= 5:
        concerns.append(f"{cur_title} listing {ai_core_skill_count} AI skills but no AI/ML career history (keyword-stuffer pattern)")
    if missing:
        concerns.append("no evidence of " + "; ".join(missing[:2]))
    if yoe < jd.ideal_experience_band[0]:
        concerns.append(f"{yoe:.1f}y experience is below the 5-9y band")
    elif yoe > jd.ideal_experience_band[1] + 2:
        concerns.append(f"{yoe:.1f}y experience is above the band")
    if loc_bucket == "abroad_static":
        concerns.append("outside India and not open to relocation (no visa sponsorship)")
    elif loc_bucket == "abroad_relocate":
        concerns.append("outside India but willing to relocate")
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if last_active_days > 150:
        concerns.append(f"last active ~{last_active_days} days ago")
    if resp < 0.2:
        concerns.append(f"low recruiter response rate ({resp:.0%})")
    _disq_concern = {
        "consulting_only_no_product": "entire career at services/consulting firms, no product company",
        "pure_research_no_production": "research-heavy profile with no production-deployment evidence",
        "ai_is_recent_langchain_only": "AI experience appears to be recent LangChain/OpenAI work only",
        "no_production_code_18mo": "moved into architecture/management; may not be writing production code",
        "title_chaser": "short tenures with ascending titles (title-chaser pattern)",
        "framework_enthusiast": "framework-tutorial profile rather than systems work",
        "cv_speech_robotics_primary": "primarily computer-vision/speech/robotics, little NLP/IR",
    }
    for d in disq:
        msg = _disq_concern.get(d)   # keyword_stuffer_mismatch already has a dedicated concern above
        if msg:
            concerns.append(msg)
    if is_hp:
        concerns.append("profile is internally inconsistent (honeypot): " + "; ".join(hp_reasons[:2]))

    return Features(
        candidate_id=cand["candidate_id"],
        title_fit=title_fit, domain_evidence=domain_evidence, musthave=musthave,
        nicehave=nicehave, experience_fit=experience_fit, education=education,
        location_modifier=location_modifier, behavioral_modifier=beh_mod,
        disqualifier_multiplier=disq_mult, consistency_multiplier=consistency_multiplier,
        musthave_present=present, musthave_missing=missing,
        years_of_experience=yoe, current_title=cur_title,
        current_company=p.get("current_company", ""), location_label=loc_label,
        location_bucket=loc_bucket, top_skills=relevant_skills[:6],
        ai_core_skill_count=ai_core_skill_count, domain_phrases=domain_phrases,
        product_fraction=product_fraction, last_active_days=last_active_days,
        response_rate=resp, notice_period_days=notice, open_to_work=bool(sig.get("open_to_work_flag")),
        github_activity_score=gh, disqualifiers=disq, concerns=concerns, strengths=strengths,
        is_honeypot=is_hp, honeypot_reasons=hp_reasons,
    )
