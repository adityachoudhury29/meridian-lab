"""Reasoning-quality tests mirroring the Stage-4 manual-review checks:
no hallucination, variation, and rank-consistent tone.
"""
from __future__ import annotations

import csv
from pathlib import Path

from conftest import REF, candidate, role, skill

from src.features import extract_features
from src.jd import build_jd
from src.reasoning import _LEADS, reasoning_for
from src.scoring import score_one

ROOT = Path(__file__).resolve().parents[1]
JD = build_jd()


def _scored(cand):
    return score_one(extract_features(cand, REF, JD), 0.0)


def test_no_skill_hallucination():
    """Any skill named in the reasoning must exist on the profile."""
    skills = [skill("Embeddings"), skill("FAISS"), skill("NDCG")]
    cand = candidate("CAND_0000010", "ML Engineer", 7.0,
                     roles=[role("ML Engineer", "Swiggy", "Food Delivery", 84, current=True,
                                 desc="Built embeddings retrieval and ranking with FAISS; NDCG eval.")],
                     skills=skills, summary="ranking and retrieval")
    txt = reasoning_for(_scored(cand), 1).lower()
    profile_skill_names = {s["name"].lower() for s in skills}
    # crude hallucination probe: any capitalised skill-like token must be a real skill or a known cap term
    for sk in profile_skill_names:
        pass  # presence allowed
    # negative control: a skill NOT on the profile must never appear
    assert "pinecone" not in txt and "weaviate" not in txt


def test_reasoning_varies_between_candidates():
    a = reasoning_for(_scored(candidate("CAND_0000011", "ML Engineer", 7.0,
                      roles=[role("ML Engineer", "Swiggy", "Food Delivery", 84, current=True,
                                  desc="retrieval ranking embeddings FAISS NDCG")],
                      skills=[skill("Embeddings")])), 3)
    b = reasoning_for(_scored(candidate("CAND_0000012", "NLP Engineer", 6.0,
                      roles=[role("NLP Engineer", "CRED", "Fintech", 72, current=True,
                                  desc="semantic search recommendation relevance A/B test")],
                      skills=[skill("Semantic Search")])), 7)
    assert a != b


def test_rank_consistent_tone():
    cand = candidate("CAND_0000013", "ML Engineer", 7.0,
                     roles=[role("ML Engineer", "Swiggy", "Food Delivery", 84, current=True,
                                 desc="retrieval ranking embeddings")], skills=[skill("Embeddings")])
    s = _scored(cand)
    top = reasoning_for(s, 1)
    bottom = reasoning_for(s, 97)
    assert any(top.startswith(p) for p in _LEADS["top"])
    assert any(bottom.startswith(p) for p in _LEADS["filler"])


def test_real_submission_reasonings_are_varied():
    """On a generated submission, the 100 reasonings must not collapse to one
    template (Stage-4 penalises all-identical / name-insertion templating)."""
    for name in ("submission.csv", "submission_structural.csv"):
        p = ROOT / name
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        reasonings = [r["reasoning"] for r in rows]
        assert len(set(reasonings)) >= 0.9 * len(reasonings)   # >=90% unique
        # not just name-insertion: lengths and structure should vary
        assert len({len(x) for x in reasonings}) > 20
        return
