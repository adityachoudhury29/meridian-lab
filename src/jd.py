"""JD understanding.

The challenge ships exactly one job description, so we encode it as a structured
``JobRequirements`` object rather than parsing it stochastically at runtime. The
structure below is a faithful, line-by-line reading of ``job_description.md`` —
its "Things you absolutely need", "would like", "explicitly do NOT want", the
ideal-candidate paragraph, and the location/experience/notice logic.

Crucially, this is where we capture *intent over keywords*: the role is the
intelligence layer (retrieval, ranking, matching) of a recruiting product, the
hard disqualifiers are encoded as first-class signals, and the dense-retrieval
query describes the role's *meaning* so a candidate who "built a recommendation
system" scores highly even without the buzzwords.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobRequirements:
    title: str
    # The four explicit hard requirements ("Things you absolutely need").
    must_haves: list[str]
    # "Things we'd like you to have but won't reject you for."
    nice_to_haves: list[str]
    # "Things we explicitly do NOT want" + the disqualifiers section.
    disqualifiers: list[str]
    # The "ideal candidate" paragraph, as machine-usable preferences.
    ideal_experience_band: tuple[float, float]
    ideal_experience_plateau: tuple[float, float]
    preferred_locations: list[str]
    welcome_locations: list[str]
    requires_india_or_relocate: bool
    preferred_notice_days: int
    # A distilled, technical "what this role actually needs" query used for the
    # dense (embedding) similarity component. Deliberately excludes culture prose
    # so the cosine reflects role meaning, not vibes.
    dense_query: str = ""
    dense_aspect_queries: list[str] = field(default_factory=list)


def build_jd() -> JobRequirements:
    """Return the structured requirements for the released Senior AI Engineer JD."""
    return JobRequirements(
        title="Senior AI Engineer — Founding Team (the company)",
        must_haves=[
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers / OpenAI embeddings / BGE / E5) deployed to real users",
            "Production experience with vector databases or hybrid search "
            "(Pinecone / Weaviate / Qdrant / Milvus / OpenSearch / Elasticsearch / FAISS)",
            "Strong Python and genuine attention to code quality",
            "Hands-on design of evaluation frameworks for ranking systems "
            "(NDCG / MRR / MAP / offline-to-online correlation / A/B test interpretation)",
        ],
        nice_to_haves=[
            "LLM fine-tuning (LoRA / QLoRA / PEFT)",
            "Learning-to-rank models (XGBoost-based or neural)",
            "HR-tech / recruiting-tech / marketplace product experience",
            "Distributed systems or large-scale inference optimisation",
            "Open-source contributions in the AI/ML space",
        ],
        disqualifiers=[
            "Pure-research career (academic / research-only) with no production deployment",
            "AI experience is only recent (<12 months) LangChain-on-OpenAI work with no pre-LLM ML production",
            "Senior who has not written production code in 18 months (moved into architecture / tech-lead)",
            "Title-chaser: switching companies ~every 1.5 years for title bumps",
            "Framework enthusiast: GitHub of LangChain tutorials / 'how I used hot framework' demos",
            "Entire career at consulting / services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) with no product-company role",
            "Primary expertise in computer vision / speech / robotics without significant NLP/IR exposure",
            "Closed-source proprietary work for 5+ years with no external validation",
        ],
        ideal_experience_band=(5.0, 9.0),
        ideal_experience_plateau=(6.0, 8.0),
        preferred_locations=["noida", "pune"],
        welcome_locations=["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                           "bangalore", "bengaluru"],
        requires_india_or_relocate=True,
        preferred_notice_days=30,
        dense_query=(
            "Senior AI/ML engineer with six to eight years of experience building production "
            "embeddings-based retrieval, hybrid search, ranking and recommendation systems for "
            "real users at product companies. Owns the intelligence and ranking layer of a "
            "recruiting marketplace: dense and sparse retrieval, vector databases such as FAISS, "
            "Pinecone, Qdrant, Weaviate, OpenSearch and Elasticsearch, strong Python, and rigorous "
            "ranking evaluation with NDCG, MRR, MAP, offline-to-online correlation and A/B testing. "
            "Ships fast, has shipped end-to-end search or recommendation systems at scale, strong "
            "NLP and information-retrieval background, applied machine learning rather than pure "
            "research, prefers product engineering over framework demos."
        ),
        dense_aspect_queries=[
            "built and deployed an embeddings-based semantic retrieval system with sentence transformers to production",
            "built a recommendation or ranking system that serves real users at scale at a product company",
            "vector database and hybrid search infrastructure: FAISS, Pinecone, Qdrant, Elasticsearch, OpenSearch",
            "evaluated a ranking system with NDCG, MRR, MAP and online A/B testing",
        ],
    )
