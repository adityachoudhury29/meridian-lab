"""Test helpers: a builder that produces schema-valid candidate dicts so we can
construct precise archetypes (ideal fit, keyword-stuffer trap, honeypot, …) and
assert how the ranker orders them.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REF = date(2026, 5, 27)


def skill(name, proficiency="advanced", endorsements=20, duration_months=30):
    return {"name": name, "proficiency": proficiency,
            "endorsements": endorsements, "duration_months": duration_months}


def role(title, company, industry, months, start="2021-01-01", end=None, current=False,
         size="201-500", desc=""):
    return {"company": company, "title": title, "start_date": start, "end_date": end,
            "duration_months": months, "is_current": current, "industry": industry,
            "company_size": size, "description": desc}


def candidate(cid, title, yoe, country="India", city="Pune, Maharashtra",
              roles=None, skills=None, education=None, signals=None, summary="", headline=""):
    sig = {
        "profile_completeness_score": 85, "signup_date": "2023-01-01",
        "last_active_date": "2026-05-20", "open_to_work_flag": True,
        "profile_views_received_30d": 30, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.7, "avg_response_time_hours": 5.0,
        "skill_assessment_scores": {}, "connection_count": 300, "endorsements_received": 100,
        "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 40, "search_appearance_30d": 20, "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.9, "offer_acceptance_rate": 0.5,
        "verified_email": True, "verified_phone": True, "linkedin_connected": True,
    }
    if signals:
        sig.update(signals)
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Person", "headline": headline or title,
            "summary": summary, "location": city, "country": country,
            "years_of_experience": yoe, "current_title": title,
            "current_company": (roles[0]["company"] if roles else "Acme"),
            "current_company_size": "201-500", "current_industry":
                (roles[0]["industry"] if roles else "Software"),
        },
        "career_history": roles or [role(title, "Acme", "Software", int(yoe * 12), current=True)],
        "education": education or [{"institution": "IIT", "degree": "B.Tech",
                                    "field_of_study": "CS", "start_year": 2014,
                                    "end_year": 2018, "grade": "8.5", "tier": "tier_1"}],
        "skills": skills or [],
        "redrob_signals": sig,
    }


@pytest.fixture
def ref():
    return REF
