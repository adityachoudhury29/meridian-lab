"""Hybrid scoring engine.

Final score = (weighted sum of seven 0..1 component scores) × four multiplicative
modifiers. The additive base captures *fit*; the multipliers capture *reachability
and trust*:

    base   = Σ wᵢ · componentᵢ           (title, domain-evidence, must-haves,
                                           semantic, experience, nice-to-haves, education)
    score  = base
             × location_modifier          (Noida/Pune > NCR metros > India > abroad)
             × behavioral_modifier         (active & responsive vs dormant)
             × disqualifier_multiplier      (consulting-only, title-chaser, …)
             × consistency_multiplier       (honeypots crushed to the floor)

This is a transparent, learning-to-rank-style linear blend over precomputed
features — chosen over a per-candidate LLM call precisely because the spec
forbids GPUs/network and demands a sub-5-minute CPU budget over 100K profiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .features import Features


@dataclass
class Scored:
    candidate_id: str
    score: float
    base: float
    components: dict[str, float]   # weighted contributions, for explainability
    semantic: float
    features: Features


def _base(f: Features, semantic_0_1: float) -> tuple[float, dict[str, float]]:
    w = config.WEIGHTS
    comp = {
        "title_fit": w["title_fit"] * f.title_fit,
        "domain_evidence": w["domain_evidence"] * f.domain_evidence,
        "musthave": w["musthave"] * f.musthave,
        "semantic": w["semantic"] * semantic_0_1,
        "experience": w["experience"] * f.experience_fit,
        "nicehave": w["nicehave"] * f.nicehave,
        "education": w["education"] * f.education,
    }
    return sum(comp.values()), comp


def score_one(f: Features, semantic_0_1: float) -> Scored:
    base, comp = _base(f, semantic_0_1)
    score = (base
             * f.location_modifier
             * f.behavioral_modifier
             * f.disqualifier_multiplier
             * f.consistency_multiplier)
    return Scored(candidate_id=f.candidate_id, score=float(score), base=float(base),
                  components=comp, semantic=float(semantic_0_1), features=f)


def score_pool(features: list[Features], semantic_0_1: np.ndarray) -> list[Scored]:
    """Score every candidate. ``semantic_0_1`` is the rescaled dense similarity,
    aligned 1:1 with ``features``."""
    return [score_one(f, float(semantic_0_1[i])) for i, f in enumerate(features)]


def rank(scored: list[Scored], top_k: int = 100) -> list[Scored]:
    """Sort best-first. Ties broken by candidate_id ascending so the output
    satisfies the validator's equal-score tie rule deterministically."""
    return sorted(scored, key=lambda s: (-s.score, s.candidate_id))[:top_k]


def normalized_output_scores(ranked: list[Scored]) -> list[float]:
    """Scale the ranked scores to (0,1] (divide by the top score) so the score
    column reads like a confidence and is strictly non-increasing."""
    if not ranked:
        return []
    top = max(ranked[0].score, 1e-9)
    return [round(s.score / top, 6) for s in ranked]
