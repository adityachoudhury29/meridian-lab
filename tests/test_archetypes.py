"""Archetype ordering tests — the behavioural contract of the ranker.

These encode, as executable assertions, the exact judgments the JD asks for:
the keyword-stuffer must lose to the person who *built* the system, behavioural
dormancy must cost, consulting-only / CV-primary / abroad-static must be
penalised, juniors and the over-experienced sit below the band, and honeypots
are crushed. Run with the dense component disabled (semantic=0) so they are
deterministic and model-free.
"""
from __future__ import annotations

from conftest import REF, candidate, role, skill

from src.features import extract_features
from src.jd import build_jd
from src.scoring import score_one

JD = build_jd()

RETRIEVAL_DESC = (
    "Built and deployed an embeddings-based semantic retrieval and ranking system "
    "with sentence-transformers and FAISS serving real users in production. Owned "
    "hybrid search relevance and a learning-to-rank recommender; evaluated with NDCG, "
    "MRR and online A/B tests. Strong Python."
)
AI_SKILLS = ["Embeddings", "FAISS", "NLP", "PyTorch", "Learning to Rank", "BERT",
             "Semantic Search", "Information Retrieval", "NDCG"]


def s(cand):
    f = extract_features(cand, REF, JD)
    return score_one(f, 0.0).score   # semantic disabled -> deterministic


def ideal():
    return candidate(
        "CAND_0000001", "Machine Learning Engineer", 7.0, city="Pune, Maharashtra",
        roles=[role("Machine Learning Engineer", "Swiggy", "Food Delivery", 84,
                    start="2019-05-01", current=True, desc=RETRIEVAL_DESC)],
        skills=[skill(n) for n in AI_SKILLS], summary=RETRIEVAL_DESC)


def recsys_no_buzzwords():
    # Built a recommender at a product company but lists NO AI skills, plain title.
    return candidate(
        "CAND_0000002", "Software Engineer", 6.5, city="Bangalore, Karnataka",
        roles=[role("Software Engineer", "Flipkart", "E-commerce", 78, start="2019-11-01",
                    current=True,
                    desc=("Built the product recommendation engine and search ranking that "
                          "serves millions of users; improved relevance and click-through with "
                          "a learning-to-rank model evaluated offline and via A/B tests."))],
        skills=[skill("Python"), skill("Java"), skill("Spark")],
        summary="Backend engineer who builds search and recommendation systems at scale.")


def keyword_stuffer():
    # HR Manager listing many AI skills (expert, never used) — the dataset's trap.
    return candidate(
        "CAND_0000003", "HR Manager", 7.0, city="Pune, Maharashtra",
        roles=[role("HR Manager", "Conglomerate Ltd", "Conglomerate", 84, start="2019-05-01",
                    current=True,
                    desc="Led talent acquisition, payroll and employee engagement programs.")],
        # realistic endorsements/duration so this is NOT a honeypot — a pure
        # title/career-vs-skills mismatch, which is the trap we must catch.
        skills=[skill(n, proficiency="advanced", endorsements=8, duration_months=18) for n in AI_SKILLS],
        summary="Human resources leader specialising in hiring and people operations.")


def honeypot():
    return candidate(
        "CAND_0000004", "Machine Learning Engineer", 3.0, city="Pune, Maharashtra",
        roles=[role("Machine Learning Engineer", "Acme", "Software", 120,  # impossible vs 3y
                    start="2024-01-01", current=True, desc=RETRIEVAL_DESC)],
        skills=[skill(n, proficiency="expert", endorsements=0, duration_months=0) for n in AI_SKILLS],
        summary=RETRIEVAL_DESC)


def inactive_strong():
    c = ideal()
    c["candidate_id"] = "CAND_0000005"
    c["redrob_signals"].update({"last_active_date": "2025-07-01", "recruiter_response_rate": 0.03,
                                "open_to_work_flag": False, "interview_completion_rate": 0.2})
    return c


def consulting_only():
    return candidate(
        "CAND_0000006", "Software Engineer", 7.0, city="Pune, Maharashtra",
        roles=[role("Software Engineer", "Infosys", "IT Services", 50, start="2019-05-01"),
               role("Senior Software Engineer", "TCS", "IT Services", 34, start="2023-07-01",
                    current=True, desc=RETRIEVAL_DESC)],
        skills=[skill(n) for n in AI_SKILLS], summary=RETRIEVAL_DESC)


def cv_primary():
    desc = ("Computer vision engineer: object detection, image segmentation and OCR pipelines "
            "for autonomous driving; trained CNNs and diffusion models for image generation.")
    return candidate(
        "CAND_0000007", "Computer Vision Engineer", 7.0, city="Pune, Maharashtra",
        roles=[role("Computer Vision Engineer", "Acme Vision", "Software", 84, start="2019-05-01",
                    current=True, desc=desc)],
        skills=[skill("Object Detection"), skill("Image Classification"), skill("OCR"),
                skill("PyTorch")], summary=desc)


def abroad_static():
    c = ideal()
    c["candidate_id"] = "CAND_0000008"
    c["profile"]["country"] = "USA"
    c["profile"]["location"] = "San Francisco"
    c["redrob_signals"]["willing_to_relocate"] = False
    return c


def junior():
    c = ideal()
    c["candidate_id"] = "CAND_0000009"
    c["profile"]["current_title"] = "Junior ML Engineer"
    c["profile"]["years_of_experience"] = 2.0
    c["career_history"][0]["title"] = "Junior ML Engineer"
    c["career_history"][0]["duration_months"] = 24
    return c


# --------------------------------------------------------------------------- #
def test_keyword_stuffer_loses_to_real_builder():
    """The single most important judgment in the JD."""
    assert s(recsys_no_buzzwords()) > s(keyword_stuffer())
    assert s(ideal()) > s(keyword_stuffer())


def test_plain_recsys_builder_is_relevant():
    """A product-company recommender-builder with no buzzwords should still rank well.
    Threshold is conservative because these tests disable the dense component
    (semantic=0), which in production adds a further ~0.1 to such a candidate."""
    assert s(recsys_no_buzzwords()) > 0.40


def test_honeypot_is_crushed():
    # A honeypot looks like a strong ML profile but is internally impossible; it
    # must be crushed far below real fits (both real candidates and the trap).
    assert s(honeypot()) < s(ideal())
    assert s(honeypot()) < s(recsys_no_buzzwords())
    assert s(honeypot()) < 0.1
    assert extract_features(honeypot(), REF, JD).is_honeypot


def test_keyword_stuffer_is_suppressed():
    # Non-tech title + AI skills + no career evidence -> heavily penalised.
    assert s(keyword_stuffer()) < 0.30
    assert s(ideal()) > 2 * s(keyword_stuffer())


def test_behavioural_dormancy_costs():
    assert s(ideal()) > s(inactive_strong())


def test_consulting_only_penalised():
    assert s(ideal()) > s(consulting_only())


def test_cv_primary_penalised():
    assert s(ideal()) > s(cv_primary())


def test_location_preference():
    assert s(ideal()) > s(abroad_static())


def test_experience_band():
    assert s(ideal()) > s(junior())


def test_full_ordering_is_sane():
    order = {
        "ideal": s(ideal()),
        "recsys": s(recsys_no_buzzwords()),
        "stuffer": s(keyword_stuffer()),
        "honeypot": s(honeypot()),
    }
    assert order["ideal"] >= order["recsys"] > order["stuffer"]
    assert order["honeypot"] < order["recsys"]      # honeypot crushed below real fit
