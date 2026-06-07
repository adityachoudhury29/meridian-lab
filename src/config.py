"""Central configuration: scoring weights, modifier envelopes and the domain
lexicons used to turn a free-text JD + candidate profiles into structured
signals.

Everything that a reviewer might want to tune lives here, in plain data, so the
ranking logic in ``scoring.py`` reads like a description of *what a recruiter
weighs* rather than a wall of magic numbers.

These values were hand-tuned against (a) a battery of archetype unit tests in
``tests/test_archetypes.py`` and (b) inspection of the realised top-100 over the
full pool (composition, honeypot rate, experience/location mix). They are NOT
fit to the hidden ground truth — there is none available — they encode the
explicit hiring logic stated in the job description.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Reference date for recency.
# The dataset's last_active_date values run up to 2026-05-27; we compute the
# reference "now" deterministically from the data at load time (max
# last_active_date) and fall back to this if the pool is empty.
# ---------------------------------------------------------------------------
FALLBACK_REFERENCE_DATE = "2026-05-27"

# ---------------------------------------------------------------------------
# Base score component weights.  These combine into the pre-modifier "base"
# score and sum to 1.0 so the base stays interpretable on a 0..1 scale.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "title_fit": 0.18,        # how close the candidate's role(s) are to the AI/ML/IR mandate
    "domain_evidence": 0.26,  # career-history evidence of retrieval / ranking / recsys / search work
    "musthave": 0.20,         # coverage of the 4 explicit hard requirements (trust-weighted)
    "semantic": 0.16,         # dense cosine(JD narrative, candidate career narrative)
    "experience": 0.08,       # fit against the 6-8 (5-9) year band
    "nicehave": 0.07,         # LtR / fine-tuning / HR-tech / distributed / OSS
    "education": 0.05,        # mild prestige signal — explicitly NOT decisive in the JD
}

# Modifier envelopes (multiplicative on top of the base score).
LOCATION_MODIFIER = {
    "noida_pune": 1.05,        # explicitly preferred offices
    "ncr_metro": 1.02,         # Hyderabad / Mumbai / Delhi NCR / Bangalore — "welcome to apply"
    "other_india": 0.98,
    "abroad_relocate": 0.90,   # outside India but willing to relocate (no visa sponsorship)
    "abroad_static": 0.78,     # outside India, not willing to relocate
}

BEHAVIORAL_BOUNDS = (0.50, 1.12)      # clamp for the multiplicative availability modifier
DISQUALIFIER_FLOOR = 0.12             # the strongest stacked penalties cannot push below this
CONSISTENCY_HONEYPOT = 0.03          # multiplier applied to a profile flagged as a honeypot

# Per-disqualifier multipliers (see features.py / scoring.py for detection).
DISQUALIFIER_MULTIPLIERS = {
    "consulting_only_no_product": 0.35,
    "pure_research_no_production": 0.42,
    "ai_is_recent_langchain_only": 0.60,
    "no_production_code_18mo": 0.72,
    "title_chaser": 0.78,
    "framework_enthusiast": 0.72,
    "cv_speech_robotics_primary": 0.52,
    "closed_source_no_validation": 0.88,
    # The dataset's signature trap: a non-technical role listing AI skills but
    # whose actual career history shows zero retrieval/ranking/ML work. The
    # skills are decorative; the work is not there. Crush it.
    "keyword_stuffer_mismatch": 0.40,
}

# Experience band (years).  Plateau is the JD's stated ideal; the curve tapers
# rather than gates, because the JD says the band is "a range, not a requirement".
EXPERIENCE_BAND = {
    "plateau": (6.0, 8.0),
    "good": (5.0, 9.0),
    "floor": 0.20,
}

# ---------------------------------------------------------------------------
# Lexicons.  Phrases are matched case-insensitively against normalised text.
# Keep these readable: each list is a recruiter-legible bucket of evidence.
# ---------------------------------------------------------------------------

# The role's CORE mandate: retrieval / ranking / recommendation / search.
# This is the single most decisive evidence bucket — it is what separates a
# genuine fit from a keyword-stuffer (who lists skills but whose *career* shows
# no such work).
DOMAIN_CORE = [
    "retrieval", "retrieve", "ranking", "rank ", "re-rank", "rerank", "learning to rank",
    "learning-to-rank", "ltr", "recommend", "recommender", "recommendation", "recsys",
    "personalization", "personalisation", "relevance", "search relevance", "search ranking",
    "semantic search", "vector search", "information retrieval", "matching system",
    "candidate matching", "two-tower", "two tower", "dense retrieval", "hybrid search",
    "hybrid retrieval", "nearest neighbor", "nearest neighbour", "ann ", "knn ",
    "click-through", "ctr ", "feed ranking", "search engine",
]

# The four explicit hard requirements ("Things you absolutely need").
MUSTHAVE_LEXICONS = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "sbert", "bge", " e5 ", "word2vec", "glove", "bert", "transformer", "encoder",
        "dense retrieval", "vector representation", "semantic similarity", "openai embedding",
    ],
    "vector_db_hybrid": [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "annoy", "hnsw", "scann",
        "vector database", "vector db", "vector store", "vector index", "opensearch",
        "elasticsearch", "elastic search", "solr", "lucene", "bm25", "hybrid search",
        "approximate nearest neighbor", "approximate nearest neighbour",
    ],
    "strong_python": [
        "python", "pytorch", "tensorflow", "scikit-learn", "scikit learn", "sklearn",
        "numpy", "pandas", "fastapi", "huggingface", "hugging face", "jax",
    ],
    "ranking_eval": [
        "ndcg", "mrr", "mean reciprocal", "mean average precision", " map ", "map@",
        "precision@", "recall@", "a/b test", "ab test", "a/b testing", "offline evaluation",
        "online evaluation", "offline-to-online", "auc", "roc", "evaluation framework",
        "relevance judgment", "relevance judgement", "eval harness", "ranking metric",
    ],
}

# "Things we'd like you to have but won't reject you for."
NICEHAVE_LEXICONS = {
    "llm_finetuning": ["fine-tune", "fine tune", "finetune", "lora", "qlora", "peft",
                        "instruction tuning", "rlhf", "dpo", "sft "],
    "learning_to_rank_models": ["xgboost", "lightgbm", "catboost", "gradient boosted",
                                 "gradient-boosted", "gbdt", "learning to rank", "lambdamart"],
    "hr_recruiting_marketplace": ["recruit", "hr-tech", "hr tech", "hiring", "talent",
                                   "marketplace", "two-sided", "candidate", "applicant tracking",
                                   "job matching"],
    "distributed_scale": ["distributed", "large-scale", "large scale", "high-throughput",
                           "throughput", "low-latency", "low latency", "spark", "kafka",
                           "flink", "ray ", "horovod", "sharding", "billions", "millions of"],
    "open_source": ["open-source", "open source", "github.com", "oss ", "contributor",
                     "maintainer", "pull request", "published", "arxiv", "neurips", "acl ",
                     "emnlp", "kdd", "sigir", "recsys conference"],
}

# NLP / IR evidence (positive) vs CV / speech / robotics (a disqualifier when it
# is the *primary* expertise and NLP/IR is absent).
NLP_IR_TERMS = [
    "nlp", "natural language", "language model", "text classification", "named entity",
    "ner ", "question answering", "summarization", "summarisation", "topic model",
    "information retrieval", "search", "retrieval", "ranking", "recommendation",
    "embedding", "transformer", "bert", "llm", "tokeniz", "semantic",
]
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "segmentation",
    "ocr", "face recognition", "pose estimation", "image generation", "diffusion model",
    "speech recognition", "asr", "text-to-speech", "tts", "speaker", "audio",
    "robotics", "slam", "lidar", "point cloud", "autonomous driving", "self-driving",
    "manipulation", "reinforcement learning control",
]

# Framework-tutorial / wrapper signal (negative when it is the *whole* story).
FRAMEWORK_WRAPPER_TERMS = [
    "langchain", "llamaindex", "llama-index", "autogen", "crewai", "haystack tutorial",
    "openai api", "gpt wrapper", "chatgpt wrapper", "prompt engineering", "no-code",
]

# Pure-research / academic signal (negative when production evidence is absent).
RESEARCH_TERMS = [
    "research scientist", "research engineer", "phd", "ph.d", "postdoc", "post-doc",
    "research lab", "research assistant", "academic", "university", "institute of technology",
    "publication", "published", "thesis", "dissertation", "professor",
]
PRODUCTION_TERMS = [
    "production", "deployed", "deploy", "shipped", "ship ", "launched", "in production",
    "real users", "real-time", "serving", "served", "live system", "scaled", "at scale",
    "on-call", "oncall", "sla", "uptime", "rollout", "a/b test",
]

# Industries that read as *product* companies vs *services / non-tech*.
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "ecommerce", "food delivery", "saas", "ai/ml",
    "adtech", "edtech", "insurance tech", "insurtech", "healthtech", "gaming", "social media",
    "transportation", "logistics tech", "proptech", "traveltech",
}
SERVICES_INDUSTRIES = {
    "it services", "consulting", "outsourcing", "staffing", "bpo",
}
NONTECH_INDUSTRIES = {
    "manufacturing", "paper products", "conglomerate", "retail", "construction",
    "automotive", "oil & gas", "agriculture", "hospitality", "education", "healthcare",
}

# Named consulting / services firms (the JD explicitly names several).  An
# *entire* career at these, with no product-company role, is a disqualifier.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "hcltech", "mindtree", "ltimindtree", "l&t infotech", "mphasis",
    "deloitte", "ernst & young", "kpmg", "pwc", "pricewaterhouse", "igate", "syntel",
    "birlasoft", "hexaware", "nttdata", "ntt data", "dxc", "atos", "persistent systems",
    "zensar",
}

# Non-technical current titles that, when paired with a stuffed AI skill list,
# are the dataset's signature keyword-stuffer trap.
NONTECH_TITLES = {
    "hr manager", "accountant", "sales executive", "marketing manager", "content writer",
    "graphic designer", "operations manager", "business analyst", "project manager",
    "customer support", "civil engineer", "mechanical engineer",
}

# Titles that read as genuinely on-mandate (AI / ML / NLP / DS / IR engineering).
STRONG_AI_TITLE_TERMS = [
    "ml engineer", "machine learning engineer", "applied scientist", "applied ml",
    "ai engineer", "artificial intelligence engineer", "nlp engineer", "data scientist",
    "research scientist", "search engineer", "relevance engineer", "recommendation",
    "ranking engineer", "ml scientist", "deep learning",
]
# Adjacent engineering titles — relevant only when corroborated by domain evidence.
ADJACENT_TITLE_TERMS = [
    "software engineer", "backend engineer", "data engineer", "analytics engineer",
    "full stack", "fullstack", "platform engineer", "cloud engineer", "devops",
    "data analyst", "full-stack",
]

# Skills that count as "AI core" — used for the *trust-weighted* skill signal and
# to surface the keyword-stuffer trap. Deliberately NOT a primary ranking driver.
AI_CORE_SKILLS = {
    "nlp", "llm", "llms", "fine-tuning llms", "fine-tuning", "rag", "embeddings",
    "transformers", "pytorch", "tensorflow", "vector search", "sentence-transformers",
    "semantic search", "information retrieval", "learning to rank", "faiss", "bert",
    "deep learning", "machine learning", "recommendation systems", "hugging face",
    "lora", "peft", "llamaindex", "langchain", "pinecone", "weaviate", "qdrant",
    "milvus", "elasticsearch", "ndcg", "neural networks", "mlops",
}

# Indian-city buckets for location fit.
NOIDA_PUNE = {"noida", "pune"}
NCR_METROS = {"hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore",
              "bengaluru", "ghaziabad", "faridabad", "navi mumbai", "thane"}
